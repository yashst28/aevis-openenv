---
title: AEVIS OpenEnv
emoji: 👁️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
tags:
  - openenv
---

# AEVIS OpenEnv — AI Retinal Disease Screening Environment

An OpenEnv-compatible environment for benchmarking AI agents on real-world retinal disease screening tasks, simulating the AEVIS portable fundus camera system used in rural and underserved clinical settings.

## Motivation

Diabetic retinopathy, glaucoma, and AMD are leading causes of preventable blindness. Early AI-assisted screening can save sight — but AI agents must be rigorously evaluated before deployment. AEVIS OpenEnv provides a reproducible benchmark for exactly this.

## Action Space

| Task | Fields | Values |
|------|--------|--------|
| Easy | `screening_result` | `normal`, `refer` |
| Medium | `disease_label`, `confidence` | 7 disease classes, 0.0–1.0 |
| Hard | `disease_label`, `urgency`, `recommendation`, `report_summary` | Full clinical workflow |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_level` | string | easy / medium / hard |
| `patient_id` | string | Unique patient ID |
| `patient_age` | int | Age in years |
| `diabetes_years` | int or null | Years since diabetes diagnosis |
| `image_description` | string | Retinal fundus findings (text) |
| `previous_actions` | list | History of actions this episode |
| `step_number` | int | Current step |
| `task_complete` | bool | Episode done flag |

## Tasks

### Task 1 — Binary Screening (Easy)
Classify the retinal scan as `normal` or `refer`. Tests basic pathology detection ability. Grader penalises false negatives on urgent cases heavily.

### Task 2 — DR Severity Grading (Medium)
Classify into one of 7 categories: `normal`, `mild_dr`, `moderate_dr`, `severe_dr`, `proliferative_dr`, `glaucoma_suspect`, `amd`. Partial credit for near-miss severity grades.

### Task 3 — Full Clinical Workflow (Hard)
Provide all four fields: disease classification, urgency triage, clinical recommendation, and patient report summary. Weighted scoring across all components.

## Reward Function

Multi-component reward (0.0–1.0), never binary:
- **Accuracy** (main signal): grader score based on correctness
- **Efficiency bonus**: +0.05 for correct answer on first step
- **Safety penalty**: up to -0.20 for missing urgent pathology
- **Completeness bonus**: +0.05 for all workflow fields on hard task

## Setup

```bash
git clone https://huggingface.co/spaces/Yash0028/aevis-openenv
cd aevis-openenv
pip install -r requirements.txt
```

## Running the API

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Endpoints: `GET /health`, `POST /reset`, `POST /step`, `GET /state`, `GET /tasks`

## Running Inference

```bash
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
export HF_TOKEN=gsk_your_key_here
python inference.py
```

## Docker

```bash
docker build -t aevis-openenv .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.groq.com/openai/v1 \
  -e MODEL_NAME=llama-3.3-70b-versatile \
  -e HF_TOKEN=your_key \
  aevis-openenv
```

## Baseline Scores

| Task | Score |
|------|-------|
| Easy | 1.0000 |
| Medium | 0.9877 |
| Hard | 0.9877 |
| **Overall** | **0.9918** |

Model: `llama-3.3-70b-versatile` via Groq · Seed: 42 · 20 cases per task
