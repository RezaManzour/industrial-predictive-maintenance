# Industrial Predictive Maintenance

[![CI](https://github.com/RezaManzour/industrial-predictive-maintenance/actions/workflows/ci.yml/badge.svg)](https://github.com/RezaManzour/industrial-predictive-maintenance/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: AI4I 2020](https://img.shields.io/badge/dataset-UCI%20AI4I%202020-orange.svg)](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

Predicting machine failures on the AI4I 2020 dataset, then using those
predictions to choose **when to maintain**. One industrial story in three
parts: classical ML, deep learning (PyTorch, with a TensorFlow/Keras mirror),
and a small reinforcement-learning maintenance policy, all CPU-only.

> **Status:** Phase 0 (scaffolding). No results yet. Every number in this
> README will come from a script in this repo, with the command to reproduce it.

## Problem

- **Data:** 10,000 rows, 6 sensor/process features plus product `Type` (L/M/H).
- **Target:** `Machine failure` (binary, about 3.4% positive).
- **Why it is hard:** strong class imbalance, and the costs are asymmetric: a
  missed failure (unplanned downtime) costs far more than a planned stop.

## Phases

| Phase | Scope | Status |
|---|---|---|
| 0 | Repo scaffolding, leakage-safe split, metrics, CI | ✅ In Progress |
| 1 | EDA: imbalance, temperature correlation, tool-wear drift, failure modes | ⏳ Planned |
| 2 | Classical ML: Dummy / LogReg → RandomForest → XGBoost | ⏳ Planned |
| 3 | Deep learning: PyTorch MLP with `Type` embedding + Keras mirror | ⏳ Planned |
| 4 | RL: `MaintenanceEnv` + tabular Q-learning vs threshold vs random | ⏳ Planned |
| 5 | Results table, limitations, future work | ⏳ Planned |

## Evaluation protocol

- One stratified **70/15/15** train/val/test split, `seed=42`, shared by every
  model. AI4I rows are independent (`UDI` is a row id, not a machine id), so a
  stratified row split is leakage-free; `src/data/splits.py` verifies the
  splits are disjoint.
- Thresholds and hyperparameters are chosen on **val**; **test** is used once
  per model, for the final table.
- Primary metric: **PR-AUC**. Also ROC-AUC, F1 and recall at fixed precision
  (`src/utils/metrics.py`).
- Failure-mode flags (TWF, HDF, PWF, OSF, RNF) are excluded from the features
  because they define the target.

## Repository layout

src/
data/splits.py stratified, leakage-checked split
features/engineering.py physics-motivated features (Phase 2)
utils/metrics.py PR-AUC, ROC-AUC, F1@threshold, recall@precision
notebooks/ EDA (Phase 1)
reports/figures/ saved figures
results/ metrics JSON written by training scripts
tests/ offline tests on synthetic data

## Reproduce

```bash  
python -m venv .venv && source .venv/bin/activate  
pip install -r requirements-ci.txt  
pytest tests/ -v  
Download ai4i2020.csv from
UCI
into data/raw/ (the data/ folder is git-ignored). Training commands will
be added per phase.

Results
To be filled from real runs in Phases 2–4.

Dataset
S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance
Applications," 2020 Third International Conference on Artificial Intelligence
for Industries (AI4I), 2020, pp. 69–74. Dataset: UCI Machine Learning
Repository, CC BY 4.0.

License
MIT
