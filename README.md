# Industrial Predictive Maintenance

Predicting machine failures on the [AI4I 2020 Predictive Maintenance dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
(10,000 rows, about 3.4% failures). The project compares classical machine learning with deep models
(PyTorch) and adds a small reinforcement-learning maintenance policy. Everything runs on CPU.

## Status

Phases 0-3 done: scaffold, leakage-safe split, metrics, EDA. Phase 4 adds classical baselines (code in
`src/ipm/classical.py`). Model numbers are added to this README only after they are produced by code in
this repo; see `results/` for the raw outputs.

## Data findings (from `results/eda_summary.json`)

- 10,000 rows, 339 failures (3.39%), no missing values, no duplicate rows.
- Failure rate by product type: L 3.92%, M 2.77%, H 2.09%.
- Linear correlation with failure is weak (torque 0.19, tool wear 0.11), so relationships are non-linear.
- Air and process temperature are strongly correlated with each other; torque and rotational speed are
  strongly negatively correlated.
- Label quirks: 9 failures have no failure-mode flag, and 18 rows have a mode flag but no recorded failure.
- The dataset is synthetic and its failures follow fixed rules. High scores here show that a model can recover
  those rules, not that it would transfer to real plant data.

## Evaluation protocol

- Stratified 70/15/15 train/validation/test split, fixed seed (42).
- Failure-mode flags (TWF, HDF, PWF, OSF, RNF) and identifiers are never used as features, since they leak the label.
- Primary metric: PR-AUC. Also precision, recall and F1 at a threshold chosen on validation data only.
- The test set is used once per final model.

## Roadmap

1. EDA and baselines
2. Classical models (logistic regression, random forest, gradient boosting), with and without engineered features
3. Deep models (MLP and variants) with the same protocol
4. Cost-aware threshold selection and a small RL maintenance policy
5. Final report and honest comparison

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Download the data and run the EDA:

```bash
python scripts/download_data.py
python -m ipm.eda
python -m ipm.classical
```

## Layout

```
src/ipm/        data loading, splits, metrics
tests/          unit tests
notebooks/      exploration
configs/        experiment settings
results/        saved metrics
reports/        figures and write-ups
```

## License

MIT
