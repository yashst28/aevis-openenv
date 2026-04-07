"""
AEVIS OpenEnv — Grader functions for Tasks 1, 2, 3.
All graders return a float between 0.0 and 1.0.
Graders use partial credit — they never return the same value for all inputs.
"""

from env.models import Action, DiseaseLabel

# Severity ordering for partial credit
SEVERITY_ORDER = [
    "normal",
    "mild_dr",
    "moderate_dr",
    "severe_dr",
    "proliferative_dr",
]

# Non-DR labels treated as separate category
NON_DR_LABELS = {"glaucoma_suspect", "amd"}


def grade_task1(action: Action, ground_truth: dict) -> tuple[float, str]:
    """
    Task 1 — Binary screening: normal vs refer.
    Returns (score 0.0–1.0, feedback string).
    """
    gt_label = ground_truth["ground_truth_label"]
    gt_urgency = ground_truth["ground_truth_urgency"]

    # Ground truth: should the patient be referred?
    should_refer = gt_label != "normal"
    agent_refers = action.screening_result == "refer"

    if action.screening_result is None:
        return 0.0, "No screening result provided. Must be 'normal' or 'refer'."

    if action.screening_result not in ("normal", "refer"):
        return 0.0, f"Invalid screening result '{action.screening_result}'. Use 'normal' or 'refer'."

    if agent_refers == should_refer:
        # Correct decision
        if gt_urgency == "urgent" and agent_refers:
            return 1.0, "Correct. Urgent case correctly flagged for referral."
        elif not should_refer:
            return 1.0, "Correct. Normal retina correctly identified — no referral needed."
        else:
            return 1.0, "Correct. Pathology detected, referral recommended."
    else:
        if should_refer and not agent_refers:
            # False negative — dangerous miss
            if gt_urgency == "urgent":
                return 0.0, "Critical miss. Urgent pathology labelled as normal. Patient at risk of blindness."
            else:
                return 0.1, "Missed pathology. Disease present but labelled normal."
        else:
            # False positive — over-referral
            return 0.3, "False positive. Normal retina referred unnecessarily. Acceptable but inefficient."


def grade_task2(action: Action, ground_truth: dict) -> tuple[float, str]:
    """
    Task 2 — DR severity grading (5-class + glaucoma + AMD).
    Partial credit: exact match = 1.0, off-by-one = 0.6, off-by-two = 0.3, wrong category = 0.0.
    """
    gt_label = ground_truth["ground_truth_label"]

    if action.disease_label is None:
        return 0.0, "No disease label provided."

    pred = action.disease_label.lower().strip()

    if pred == gt_label:
        score = 1.0
        feedback = f"Perfect classification. Correctly identified as '{gt_label}'."
    elif gt_label in SEVERITY_ORDER and pred in SEVERITY_ORDER:
        # Both are DR labels — partial credit based on severity distance
        gt_idx = SEVERITY_ORDER.index(gt_label)
        pred_idx = SEVERITY_ORDER.index(pred)
        distance = abs(gt_idx - pred_idx)
        if distance == 1:
            score = 0.6
            feedback = f"Off by one severity grade. True: '{gt_label}', Predicted: '{pred}'."
        elif distance == 2:
            score = 0.3
            feedback = f"Off by two severity grades. True: '{gt_label}', Predicted: '{pred}'."
        else:
            score = 0.1
            feedback = f"Severity grade far off. True: '{gt_label}', Predicted: '{pred}'."
    elif gt_label in NON_DR_LABELS and pred in NON_DR_LABELS:
        # Both non-DR but wrong specific condition
        score = 0.4
        feedback = f"Correct disease category but wrong specific condition. True: '{gt_label}', Predicted: '{pred}'."
    elif gt_label == "normal" and pred != "normal":
        score = 0.0
        feedback = f"False positive. Normal retina labelled as '{pred}'."
    elif gt_label != "normal" and pred == "normal":
        score = 0.0
        feedback = f"False negative. '{gt_label}' missed and labelled normal."
    else:
        # Wrong disease category entirely
        score = 0.0
        feedback = f"Wrong disease category. True: '{gt_label}', Predicted: '{pred}'."

    # Confidence bonus/penalty
    if action.confidence is not None and score > 0:
        if score == 1.0 and action.confidence >= 0.8:
            score = min(1.0, score + 0.0)  # already max
            feedback += f" High confidence ({action.confidence:.0%}) on correct prediction."
        elif score == 1.0 and action.confidence < 0.5:
            score = 0.85
            feedback += f" Correct but low confidence ({action.confidence:.0%}) — penalised slightly."
        elif score < 0.5 and action.confidence > 0.8:
            score = max(0.0, score - 0.1)
            feedback += f" Overconfident ({action.confidence:.0%}) on incorrect prediction."

    return round(score, 3), feedback


def grade_task3(action: Action, ground_truth: dict) -> tuple[float, str]:
    """
    Task 3 — Full patient workflow: label + urgency + recommendation + report.
    Weighted scoring across all components.
    """
    gt_label = ground_truth["ground_truth_label"]
    gt_urgency = ground_truth["ground_truth_urgency"]

    scores = {}
    feedbacks = []

    # Component 1: Disease classification (40% weight)
    label_score, label_fb = grade_task2(action, ground_truth)
    scores["classification"] = label_score * 0.40
    feedbacks.append(f"[Classification] {label_fb}")

    # Component 2: Urgency decision (30% weight)
    urgency_score = _grade_urgency(action.urgency, gt_urgency)
    scores["urgency"] = urgency_score * 0.30
    feedbacks.append(f"[Urgency] {_urgency_feedback(action.urgency, gt_urgency)}")

    # Component 3: Recommendation quality (20% weight)
    rec_score = _grade_recommendation(action.recommendation, gt_label, gt_urgency)
    scores["recommendation"] = rec_score * 0.20
    feedbacks.append(f"[Recommendation] {_recommendation_feedback(rec_score)}")

    # Component 4: Report summary (10% weight)
    report_score = _grade_report(action.report_summary)
    scores["report"] = report_score * 0.10
    feedbacks.append(f"[Report] {_report_feedback(report_score)}")

    total = round(sum(scores.values()), 3)
    breakdown_feedback = " | ".join(feedbacks)
    final_feedback = (
        f"Total score: {total:.3f} | "
        f"Classification: {label_score:.2f} (×0.4) | "
        f"Urgency: {urgency_score:.2f} (×0.3) | "
        f"Recommendation: {rec_score:.2f} (×0.2) | "
        f"Report: {report_score:.2f} (×0.1)"
    )

    return total, final_feedback


# ── Helpers ───────────────────────────────────────────────────────────────────

URGENCY_ORDER = ["monitor", "refer", "urgent"]


def _grade_urgency(predicted: str, ground_truth: str) -> float:
    if predicted is None:
        return 0.0
    pred = predicted.lower().strip()
    if pred == ground_truth:
        return 1.0
    if pred not in URGENCY_ORDER or ground_truth not in URGENCY_ORDER:
        return 0.0
    distance = abs(URGENCY_ORDER.index(pred) - URGENCY_ORDER.index(ground_truth))
    if distance == 1:
        return 0.5
    return 0.0


def _urgency_feedback(predicted: str, ground_truth: str) -> str:
    if predicted is None:
        return "No urgency provided."
    if predicted == ground_truth:
        return f"Correct urgency: '{ground_truth}'."
    if ground_truth == "urgent" and predicted != "urgent":
        return f"Under-triaged. Should be 'urgent', got '{predicted}'. Patient safety risk."
    if ground_truth == "monitor" and predicted == "urgent":
        return f"Over-triaged. 'monitor' case escalated to 'urgent'. Wastes resources."
    return f"Urgency off by one level. True: '{ground_truth}', Predicted: '{predicted}'."


def _grade_recommendation(recommendation: str, gt_label: str, gt_urgency: str) -> float:
    if not recommendation or len(recommendation.strip()) < 10:
        return 0.0
    rec = recommendation.lower()
    score = 0.3  # base for having any recommendation

    # Check for relevant keywords
    if gt_urgency == "urgent":
        if any(w in rec for w in ["urgent", "immediate", "emergency", "same day", "today"]):
            score += 0.5
        if any(w in rec for w in ["ophthalmolog", "specialist", "hospital"]):
            score += 0.2
    elif gt_urgency == "refer":
        if any(w in rec for w in ["refer", "ophthalmolog", "follow", "appointment"]):
            score += 0.5
        if any(w in rec for w in ["month", "week", "soon"]):
            score += 0.2
    else:  # monitor
        if any(w in rec for w in ["monitor", "annual", "yearly", "follow", "screen"]):
            score += 0.5
        if any(w in rec for w in ["lifestyle", "diet", "sugar", "blood"]):
            score += 0.2

    return min(1.0, score)


def _recommendation_feedback(score: float) -> str:
    if score >= 0.9:
        return "Excellent recommendation with correct urgency and specific guidance."
    elif score >= 0.6:
        return "Good recommendation, partially appropriate."
    elif score >= 0.3:
        return "Basic recommendation provided but missing key urgency details."
    else:
        return "Inadequate or missing recommendation."


def _grade_report(report: str) -> float:
    if not report or len(report.strip()) < 20:
        return 0.0
    if len(report.strip()) < 50:
        return 0.4
    if len(report.strip()) < 100:
        return 0.7
    return 1.0


def _report_feedback(score: float) -> str:
    if score >= 0.9:
        return "Detailed report summary provided."
    elif score >= 0.6:
        return "Adequate report summary."
    elif score >= 0.3:
        return "Report too brief."
    else:
        return "No report provided."
