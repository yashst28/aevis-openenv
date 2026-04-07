"""
AEVIS OpenEnv — Reward function.
Provides step-by-step feedback, rewards progress, penalises bad actions.
NOT a simple +1/-1 system.
"""

from env.models import Action, Reward
from env.graders import grade_task1, grade_task2, grade_task3


def compute_reward(
    task_level: str,
    action: Action,
    ground_truth: dict,
    step_number: int,
    max_steps: int = 3,
) -> Reward:
    """
    Compute reward for the given action in the given task.

    Reward components:
      - Accuracy score from grader (main signal)
      - Step efficiency bonus (faster correct answer = slightly higher reward)
      - Safety penalty (dangerous misses on urgent cases)
      - Completeness bonus (Task 3 only)
    """

    breakdown = {}
    feedback_parts = []

    # ── 1. Grader score ───────────────────────────────────────────
    if task_level == "easy":
        grader_score, grader_feedback = grade_task1(action, ground_truth)
    elif task_level == "medium":
        grader_score, grader_feedback = grade_task2(action, ground_truth)
    else:  # hard
        grader_score, grader_feedback = grade_task3(action, ground_truth)

    breakdown["accuracy"] = round(grader_score, 3)
    feedback_parts.append(grader_feedback)

    # ── 2. Step efficiency bonus ──────────────────────────────────
    # Reward agents that get it right quickly
    efficiency_bonus = 0.0
    if grader_score >= 0.8 and step_number == 1:
        efficiency_bonus = 0.05
        feedback_parts.append("Efficiency bonus: correct on first step (+0.05).")
    elif grader_score >= 0.8 and step_number == 2:
        efficiency_bonus = 0.02
        feedback_parts.append("Efficiency bonus: correct on second step (+0.02).")
    breakdown["efficiency_bonus"] = efficiency_bonus

    # ── 3. Safety penalty ─────────────────────────────────────────
    # Heavy penalty for dangerous misses on urgent cases
    safety_penalty = 0.0
    gt_urgency = ground_truth.get("ground_truth_urgency", "monitor")
    gt_label = ground_truth.get("ground_truth_label", "normal")

    if gt_urgency == "urgent":
        # Agent said normal or low urgency on an urgent case
        if task_level == "easy" and action.screening_result == "normal":
            safety_penalty = 0.2
            feedback_parts.append(
                "Safety penalty: urgent case missed as normal (-0.20). "
                "In clinical use this could lead to irreversible vision loss."
            )
        elif task_level in ("medium", "hard") and action.disease_label == "normal":
            safety_penalty = 0.2
            feedback_parts.append(
                "Safety penalty: severe pathology labelled normal (-0.20). High clinical risk."
            )
        elif task_level == "hard" and action.urgency == "monitor":
            safety_penalty = 0.15
            feedback_parts.append(
                "Safety penalty: urgent case triaged as monitor only (-0.15)."
            )

    breakdown["safety_penalty"] = -safety_penalty

    # ── 4. Completeness bonus (Task 3 only) ───────────────────────
    completeness_bonus = 0.0
    if task_level == "hard":
        fields_provided = sum([
            action.disease_label is not None,
            action.urgency is not None,
            action.recommendation is not None and len(action.recommendation) > 20,
            action.report_summary is not None and len(action.report_summary) > 30,
        ])
        if fields_provided == 4:
            completeness_bonus = 0.05
            feedback_parts.append("Completeness bonus: all workflow fields provided (+0.05).")
        elif fields_provided == 3:
            completeness_bonus = 0.02
        elif fields_provided <= 1:
            completeness_bonus = -0.05
            feedback_parts.append("Incompleteness penalty: too few fields for a full workflow (-0.05).")
        breakdown["completeness_bonus"] = completeness_bonus

    # ── Final score ───────────────────────────────────────────────
    raw_score = grader_score + efficiency_bonus - safety_penalty + completeness_bonus
    final_score = max(0.0, min(1.0, round(raw_score, 3)))

    breakdown["final_score"] = final_score
    full_feedback = " | ".join(feedback_parts)

    return Reward(
        score=final_score,
        breakdown=breakdown,
        feedback=full_feedback,
        is_terminal=True,
    )
