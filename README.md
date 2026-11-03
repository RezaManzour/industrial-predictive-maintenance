# Industrial Predictive Maintenance

Predicting machine failures on the [AI4I 2020 Predictive Maintenance dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
(10,000 rows, about 3.4% failures). The project compares classical machine learning with deep models
(PyTorch) and adds a small reinforcement-learning maintenance policy. Everything runs on CPU.

## Status

Phase 0: project scaffold, leakage-safe data split, imbalance-aware metrics, tests and CI.
No model results yet. Numbers will be added to this README only after they are produced by code in this repo.

## Evaluation protocol

- Stratified 70/15/15 train/validation/test split, fixed seed (42).
- Failure-mode flags (TWF, HDF, PWF, OSF, RNF) and identifiers are never used as features, since they leak the label.
- Primary metric: PR-AUC. Also precision, recall and F1 at a threshold chosen on validation data only.
- The test set is used once per final model.

## Roadmap

1. EDA and baselines
2. Classical models (logistic regression, random forest, gradient boosting) with tuning on validation
3. Deep models (MLP and variants) with the same protocol
4. Cost-aware threshold selection and a small RL maintenance policy
5. Final report and honest comparison

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Download `ai4i2020.csv` from the UCI page above into `data/raw/`.

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
