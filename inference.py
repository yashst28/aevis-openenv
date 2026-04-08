"""
AEVIS OpenEnv — inference.py
Runs an AI agent through all 3 task levels and produces scores.

Usage:
    export API_BASE_URL=https://api.groq.com/openai/v1
    export MODEL_NAME=llama-3.3-70b-versatile
    export HF_TOKEN=gsk_your_groq_key_here
    python inference.py

Runtime: < 20 minutes (uses 20 cases per task level = 60 total)

STDOUT FORMAT (mandatory):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import json
import time
import sys
from typing import Optional, List
from openai import OpenAI
from env.aevis_env import AEVISEnv
from env.patient_loader import PATIENT_CASES

# ── Config ────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
SEED         = 42
CASES_PER_TASK = min(20, len(PATIENT_CASES))
BENCHMARK    = "aevis-openenv"
SUCCESS_THRESHOLD = 0.5

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "sk-placeholder")

# ── Mandatory log functions ───────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

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


def call_agent(task_level: str, obs: dict) -> tuple:
    """Call the LLM agent and parse its JSON response. Returns (action_dict, error_str)."""
    patient_info = build_patient_info(obs)
    prompt = TASK_PROMPTS[task_level].format(
        patient_info=patient_info,
        image_description=obs["image_description"],
    )
    raw = ""
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
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw), None
    except json.JSONDecodeError as e:
        return {}, f"JSON parse error: {e}"
    except Exception as e:
        return {}, f"API error: {e}"


def run_task(task_level: str, n_cases: int, seed: int) -> dict:
    """Run the agent through n_cases for a given task level with mandatory logging."""
    env = AEVISEnv(task_level=task_level, seed=seed)
    all_rewards = []
    all_scores = []

    for i in range(n_cases):
        rewards_this_ep = []
        obs = env.reset()
        patient_id = obs["patient_id"]
        task_name = f"{task_level}-{patient_id}"

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        action, error = call_agent(task_level, obs)

        if not action:
            # Failed to get action
            log_step(step=1, action="null", reward=0.00, done=True, error=error or "empty action")
            log_end(success=False, steps=1, score=0.000, rewards=[0.00])
            all_rewards.append(0.0)
            all_scores.append(0.0)
            continue

        try:
            result = env.step(action)
            reward_val = result["reward"]["score"]
            done = result["done"]
            step_error = None
        except Exception as e:
            reward_val = 0.0
            done = True
            step_error = str(e)

        rewards_this_ep.append(reward_val)

        # Sanitize action for single-line logging
        action_str = json.dumps(action).replace("\n", " ")

        log_step(
            step=1,
            action=action_str,
            reward=reward_val,
            done=done,
            error=step_error,
        )

        episode_score = reward_val
        success = episode_score >= SUCCESS_THRESHOLD

        log_end(
            success=success,
            steps=1,
            score=episode_score,
            rewards=rewards_this_ep,
        )

        all_rewards.append(reward_val)
        all_scores.append(episode_score)

        time.sleep(0.3)  # rate limit buffer

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return {
        "task_level": task_level,
        "n_cases": n_cases,
        "average_score": round(avg, 4),
        "min_score": round(min(all_scores), 4) if all_scores else 0.0,
        "max_score": round(max(all_scores), 4) if all_scores else 0.0,
        "scores": all_scores,
    }


def main():
    all_results = {}

    for task in ("easy", "medium", "hard"):
        task_result = run_task(task, CASES_PER_TASK, SEED)
        all_results[task] = task_result

    # ── Summary ───────────────────────────────────────────────────
    total_scores = [r["average_score"] for r in all_results.values()]
    overall = sum(total_scores) / len(total_scores)

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


if __name__ == "__main__":
    main()
