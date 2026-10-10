# Evaluation protocol

This file freezes the rules for every experiment in this repository. A rule
changes only through a dated entry in the change log at the bottom, with the
reason written down. Nothing is tuned on the test split.

## Data and split

- Dataset: AI4I 2020 (`data/raw/ai4i2020.csv`, git-ignored), 10,000 rows.
- Split: `src/data/splits.py::stratified_split`, stratified on `Machine failure`,
  70/15/15, `seed=42`. Rows are independent, so a row split is leakage-free;
  duplicate `UDI` values are rejected.
- Features: `Type` plus the five sensor/process columns and three engineered
  columns (`temp_diff`, `power_w`, `strain`). `UDI`, `Product ID` and the failure
  flags `TWF`, `HDF`, `PWF`, `OSF`, `RNF` are never features.

## Model selection and thresholds

- Preprocessing (encoding, scaling) and class weights are fit on train only.
- Hyperparameters are fixed in code. No search runs on the test split.
- The decision threshold is the one that maximises F1 on validation.
- Primary metric: PR-AUC. Also reported: ROC-AUC, F1 at the validation
  threshold, recall at precision >= 0.8.

## Seeds and repeats

- Classical models: one run each, `seed=42`.
- Neural models: five seeds (42, 43, 44, 45, 46), reported as mean and standard
  deviation. The seed list is fixed here before those runs.

## Test-set use

- The test split is a one-time scoring set for each model and each seed.
- Analyses that need predictions (bootstrap intervals, failure-mode recall,
  calibration) use saved predictions. They do not re-score the model.
- Any re-run that scores test again must be logged below, with the reason.

## Uncertainty

- Bootstrap intervals (2,000 resamples of the test rows, stratified by class)
  are reported for every test metric. Model differences are reported as paired
  bootstrap differences.

## Reinforcement learning costs (proposed, frozen before Phase 4)

- Cost per missed failure: 10.
- Cost per false alarm (inspection without failure): 1.
- Cost per planned replacement: 3.
- A sensitivity sweep over the missed-failure cost (5, 10, 20) is reported.
  These values are chosen before any RL run and must not be changed afterwards.

## Test-use log

| Date | Model / seed | Scored on test | Recorded in results |
|---|---|---|---|
| 2026-10-10 | classical: dummy, logistic, random_forest, xgboost | once each | `results/metrics_classical.json` |
| 2026-10-10 | PyTorch MLP, seed 42 | once | `results/metrics_deep.json` |
| 2026-10-10 | Keras MLP, seed 42 | once | **not recorded** (overwrite bug, see README) |

## Change log

- 2026-10-10: protocol written at the start of the two-week plan. Keras test
  result is not recorded and is not re-scored.
