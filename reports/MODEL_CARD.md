# Model card: gradient boosting with engineered features

**Model:** scikit-learn `HistGradientBoostingClassifier` with balanced class weights, on five sensor readings, product
type and three engineered features (mechanical power, process-air temperature gap, tool wear x torque).

**Intended use:** a reference model for studying failure prediction and cost-aware decisions on the AI4I 2020
synthetic dataset. It is a research and portfolio artifact.

**Not intended for:** decisions about real equipment, safety-relevant maintenance, or any setting whose failure
mechanisms differ from this dataset's generating rules.

## Data
AI4I 2020 (see `DATA_CARD.md`). Stratified 70/15/15 split. Identifiers, the label and the five failure-mode flags are
excluded from the features.

## Performance (test split, 5 split seeds, mean ± std)
- PR-AUC 0.879 ± 0.051 and F1 0.850 ± 0.072, with the threshold chosen on validation data. Source: `results/metrics_multiseed.json`.
- Without the engineered features: PR-AUC 0.788 ± 0.043.
- A single split (seed 42) gave about 0.94, which was a favourable draw; the multi-seed mean is the figure to use.

## Limitations and risks
- The engineered features mirror the rules that generated the labels, so the gain shows the rules can be recovered,
  not that the model generalises to real machines.
- Each test split has about 50 failures, so precision, recall and thresholds are noisy.
- Scores are nearly two-valued and not calibrated probabilities: on the seed-42 test split, 1,383 of 1,500 rows score
  below 0.01 (observed failure rate 0.0014) and 48 score above 0.5 (mean predicted 0.985, observed 0.958); in the 0.2-0.5
  bin none of 19 rows failed while about 6 were expected. Use the scores for ranking; recalibration was not done.
- Hyperparameters were not tuned.
- Costs used in the decision experiments are assumptions.
