# Periospot Social: Predicting High-Performance Posts (Binary Classification)

**Goal**  
Given only features known *before publishing* (network, post type, content type, account, posting time, caption length, etc.), predict whether a post will be a **High Performer** (threshold chosen *empirically during EDA*).

**Why it matters**  
This enables proactive content planning: schedule, format, and creative choices that raise the odds of high engagement.

## Dataset

- Source: Sprout Social exports (5+ years)
- Files:
  - `data/raw/post_performance.csv`
  - `data/raw/profile_performance.csv`

## Methods

- EDA → choose a defensible high/low threshold from the empirical distribution.
- Feature engineering (pre-posting only; strict anti-leakage).
- Models: Logistic (baseline), RandomForest, XGBoost (tuned).
- Validation: **temporal split** (train ≤2024-12, test ≥2025-01) to simulate out-of-time generalization.
- Metrics: ROC-AUC, PR-AUC, Recall/Precision, business-focused threshold tuning (maximize recall at acceptable precision).

## Repro

1. `python -m venv .venv && source .venv/bin/activate` (or use Conda)
2. `pip install -r requirements.txt`
3. Open `notebooks/01_post_performance_binary.ipynb` and run cells top-to-bottom.

