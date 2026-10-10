# Industrial Predictive Maintenance

[![CI](https://github.com/RezaManzour/industrial-predictive-maintenance/actions/workflows/ci.yml/badge.svg)](https://github.com/RezaManzour/industrial-predictive-maintenance/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: AI4I 2020](https://img.shields.io/badge/dataset-UCI%20AI4I%202020-orange.svg)](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

Predicting machine failures on the AI4I 2020 dataset, then using those
predictions to choose **when to maintain**. One industrial story in three
parts: classical ML, deep learning (PyTorch, with a TensorFlow/Keras mirror),
and a small reinforcement-learning maintenance policy, all CPU-only.

> **Status:** Phase 3 (deep learning) complete, with one recorded limitation (see Results). Every number in this README
> comes from a script in this repo, with the command to reproduce it.

## Problem

- **Data:** 10,000 rows, 6 sensor/process features plus product `Type` (L/M/H).
- **Target:** `Machine failure` (binary, about 3.4% positive).
- **Why it is hard:** strong class imbalance, and the costs are asymmetric: a
  missed failure (unplanned downtime) costs far more than a planned stop.

## Phases

| Phase | Scope | Status |
|---|---|---|
| 0 | Repo scaffolding, leakage-safe split, metrics, CI | ✅ Done |
| 1 | EDA: imbalance, temperature correlation, tool-wear drift, failure modes | ✅ Done |
| 2 | Classical ML: Dummy / LogReg → RandomForest → XGBoost | ✅ Done |
| 3 | Deep learning: PyTorch MLP with `Type` embedding + Keras mirror | ✅ Done (Keras test result not recorded) |
| 4 | RL: `MaintenanceEnv` + tabular Q-learning vs threshold vs random | ⏳ Planned |
| 5 | Results table, limitations, future work | ⏳ Planned |

## Evaluation protocol

- One stratified **70/15/15** train/val/test split, `seed=42`, shared by every
  model. AI4I rows are independent (`UDI` is a row id, not a machine id), so a
  stratified row split is leakage-free; `src/data/splits.py` verifies the
  splits are disjoint.
- Thresholds and hyperparameters are chosen on **val**; **test** is used once
  per model, for the final table. The full rules, seeds and test-use log are in
  [PROTOCOL.md](PROTOCOL.md).
- Primary metric: **PR-AUC**. Also ROC-AUC, F1 and recall at fixed precision
  (`src/utils/metrics.py`).
- Failure-mode flags (TWF, HDF, PWF, OSF, RNF) are excluded from the features
  because they define the target. `UDI` and `Product ID` are identifiers and
  are excluded too.

## Results

### Phase 2: classical models (test set, 51 positives)

| Model | PR-AUC | ROC-AUC | F1 @ val threshold | Recall @ Precision ≥ 0.8 | Threshold (from val) |
|---|---|---|---|---|---|
| Dummy (most frequent) | 0.034 | 0.500 | 0.066 | 0.000 | 0.000 |
| Dummy (stratified) | 0.034 | 0.491 | 0.066 | 0.000 | 0.000 |
| Logistic Regression | 0.443 | 0.937 | 0.444 | 0.020 | 0.906 |
| **Random Forest** | **0.910** | **0.988** | **0.894** | **0.882** | 0.467 |
| XGBoost | 0.872 | 0.982 | 0.857 | 0.824 | 0.752 |

Reading these numbers:

- A dummy classifier's PR-AUC equals the positive rate (about 0.034), so it is
  the floor. The most-frequent dummy predicts "failure" for every row, so its
  threshold is 0.
- Random Forest and XGBoost are close. With 51 positives in the test set, a
  gap of about 0.04 PR-AUC is within noise, so no winner is claimed.
- Recall at precision 0.8 is a single point on the precision-recall curve
  and is noisy at this sample size.
- Hyperparameters were fixed before evaluation and not tuned, so these are
  untuned baselines.

Reproduce: `python -m src.train_classical` (writes `results/metrics_classical.json`).

### Phase 3: deep models (test set, 51 positives)

| Model | Framework | PR-AUC | ROC-AUC | F1 @ val threshold | Recall @ Precision ≥ 0.8 | Threshold (from val) |
|---|---|---|---|---|---|---|
| PyTorch MLP (`Type` embedding) | PyTorch | 0.720 | 0.981 | 0.684 | 0.471 | 0.895 |
| Keras MLP (`Type` one-hot) | TensorFlow | not recorded | not recorded | not recorded | not recorded | not recorded |

Validation PR-AUC: PyTorch 0.770, Keras 0.761.

Limitations and disclosures:

- **Keras test result not recorded.** The Keras run was executed with a version of
  `src/train_deep.py` that overwrote `results/metrics_deep.json` and dropped its
  row. The Keras test set was scored during that run, but its numbers were not saved.
  Its validation PR-AUC was recorded. The test set is not scored again, to keep the
  one-use protocol. The fix is in commit `123ba9f`.
- The PyTorch model's test PR-AUC (0.720) is lower than its validation PR-AUC (0.770).
  With 51 test positives, this gap is within noise and is not tuned away.
- Phase 2's Random Forest remains the best classical model on test (PR-AUC 0.910).
  The deep models do not beat it on this dataset.

Reproduce the PyTorch result: `python -m src.train_deep --frameworks torch`.

RL (Phase 4) results will be added here when they run.

## Repository layout

```
src/
  data/splits.py          stratified, leakage-checked split
  features/engineering.py physics-motivated features (temp_diff, power_w, strain)
  models/classical.py     Dummy, LogReg, RandomForest, XGBoost pipelines
  train_classical.py      Phase 2 training and evaluation CLI
  utils/metrics.py        PR-AUC, ROC-AUC, F1@threshold, recall@precision
notebooks/                EDA (Phase 1): 01_eda.ipynb and its script twin
reports/figures/          saved EDA figures
results/                  metrics JSON written by training scripts
tests/                    offline tests on synthetic data
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest tests/ -v
```

Download `ai4i2020.csv` from
[UCI](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
into `data/raw/` (the `data/` folder is git-ignored). Then:

```bash
python -m src.train_classical
```

## Dataset

S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance
Applications," *2020 Third International Conference on Artificial Intelligence
for Industries (AI4I)*, 2020, pp. 69–74. Dataset: UCI Machine Learning
Repository, CC BY 4.0.

## License

MIT
