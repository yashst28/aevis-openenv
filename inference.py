"""
AEVIS OpenEnv — inference.py
Runs an AI agent through all 3 task levels and produces scores.

Usage:
    export API_BASE_URL=https://api.groq.com/openai/v1
    export MODEL_NAME=llama-3.3-70b-versatile
    export HF_TOKEN=gsk_your_groq_key_here
    python inference.py

Runtime: < 20 minutes (uses 20 cases per task level = 60 total)
"""

import os
import json
import time
import random
import sys
from openai import OpenAI
from env.aevis_env import AEVISEnv
from env.patient_loader import PATIENT_CASES

# ── Config ────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
SEED         = 42
CASES_PER_TASK = min(20, len(PATIENT_CASES))

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "sk-placeholder")

# ── Agent prompts ─────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI-powered retinal screening assistant for the AEVIS system.
You analyze descriptions of retinal fundus images and provide clinical assessments.
You must always respond with valid JSON only — no markdown, no explanation outside the JSON.

Disease labels available: normal, mild_dr, moderate_dr, severe_dr, proliferative_dr, glaucoma_suspect, amd
Urgency levels: monitor, refer, urgent
"""

TASK_PROMPTS = {
    "easy": """Analyze this retinal scan and decide: normal or refer?

Patient info:
{patient_info}

Retinal findings:
{image_description}

Respond ONLY with this JSON:
{{"screening_result": "normal" or "refer"}}""",

    "medium": """Classify the retinal disease severity from this scan description.

Patient info:
{patient_info}

Retinal findings:
{image_description}

Respond ONLY with this JSON:
{{"disease_label": "<one of: normal, mild_dr, moderate_dr, severe_dr, proliferative_dr, glaucoma_suspect, amd>", "confidence": <0.0 to 1.0>}}""",

    "hard": """Perform a full clinical workflow assessment for this retinal scan.

Patient info:
{patient_info}

Retinal findings:
{image_description}

Respond ONLY with this JSON:
{{
  "disease_label": "<classification>",
  "urgency": "<monitor | refer | urgent>",
  "recommendation": "<what should happen next for this patient>",
  "report_summary": "<brief clinical report summary>"
}}""",
}


def build_patient_info(obs: dict) -> str:
    age = obs["patient_age"]
    diabetes = obs["diabetes_years"]
    if diabetes is not None and diabetes > 0:
        dm_str = f"Diabetic for {diabetes} years"
    elif diabetes == 0:
        dm_str = "Newly diagnosed diabetic"
    else:
        dm_str = "No diabetes history"
    return f"Age {age}, {dm_str}"


def call_agent(task_level: str, obs: dict) -> dict:
    """Call the LLM agent and parse its JSON response."""
    patient_info = build_patient_info(obs)
    prompt = TASK_PROMPTS[task_level].format(
        patient_info=patient_info,
        image_description=obs["image_description"],
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON parse error: {e} | Raw: {raw[:100]}")
        return {}
    except Exception as e:
        print(f"  [ERROR] API call failed: {e}")
        return {}


def run_task(task_level: str, n_cases: int, seed: int) -> dict:
    """Run the agent through n_cases for a given task level."""
    print(f"\n{'='*60}")
    print(f"  Task: {task_level.upper()}  |  Cases: {n_cases}  |  Model: {MODEL_NAME}")
    print(f"{'='*60}")

    env = AEVISEnv(task_level=task_level, seed=seed)
    scores = []
    results = []

    for i in range(n_cases):
        obs = env.reset()
        patient_id = obs["patient_id"]
        print(f"  [{i+1:02d}/{n_cases}] Patient {patient_id} ... ", end="", flush=True)

        action = call_agent(task_level, obs)

        if not action:
            print("FAILED (empty action) → score: 0.000")
            scores.append(0.0)
            results.append({"patient_id": patient_id, "score": 0.0, "action": {}, "feedback": "Agent returned empty response."})
            continue

        result = env.step(action)
        reward = result["reward"]
        score = reward["score"]
        scores.append(score)

        print(f"score: {score:.3f}")
        if score < 0.5:
            print(f"         Feedback: {reward['feedback'][:120]}")

        results.append({
            "patient_id": patient_id,
            "score": score,
            "action": action,
            "feedback": reward["feedback"],
            "breakdown": reward.get("breakdown", {}),
        })

        time.sleep(0.3)  # rate limit buffer

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\n  Average score ({task_level}): {avg:.3f}")
    print(f"  Min: {min(scores):.3f}  |  Max: {max(scores):.3f}")

    return {
        "task_level": task_level,
        "n_cases": n_cases,
        "average_score": round(avg, 4),
        "min_score": round(min(scores), 4),
        "max_score": round(max(scores), 4),
        "scores": scores,
        "results": results,
    }


def main():
    print("\n" + "="*60)
    print("  AEVIS OpenEnv — Inference Script")
    print(f"  Model:    {MODEL_NAME}")
    print(f"  Endpoint: {API_BASE_URL}")
    print(f"  Seed:     {SEED}")
    print("="*60)

    all_results = {}

    for task in ("easy", "medium", "hard"):
        task_result = run_task(task, CASES_PER_TASK, SEED)
        all_results[task] = task_result

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL BASELINE SCORES")
    print("="*60)
    total_scores = []
    for task, res in all_results.items():
        avg = res["average_score"]
        total_scores.append(avg)
        bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
        print(f"  {task.upper():8s}  {bar}  {avg:.4f}")

    overall = sum(total_scores) / len(total_scores)
    print(f"\n  OVERALL AVERAGE:  {overall:.4f}")
    print("="*60)

    # Save results
    output = {
        "model": MODEL_NAME,
        "endpoint": API_BASE_URL,
        "seed": SEED,
        "cases_per_task": CASES_PER_TASK,
        "results": all_results,
        "summary": {
            "easy_avg": all_results["easy"]["average_score"],
            "medium_avg": all_results["medium"]["average_score"],
            "hard_avg": all_results["hard"]["average_score"],
            "overall_avg": round(overall, 4),
        },
    }

    with open("baseline_scores.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n  Results saved to baseline_scores.json")


if __name__ == "__main__":
    main()
