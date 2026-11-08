# Data card: AI4I 2020 Predictive Maintenance Dataset

**Source:** UCI Machine Learning Repository, dataset 601, described in S. Matzka, "Explainable Artificial
Intelligence for Predictive Maintenance Applications", 2020 Third International Conference on Artificial
Intelligence for Industries (AI4I). Check the UCI page for the current licence terms. The data is not redistributed in
this repository; `scripts/download_data.py` fetches it.

**Nature:** synthetic data that imitates a milling machine. Failures are generated from threshold rules, so the data
is not a record of real machine operation.

## Contents
- 10,000 rows. Columns: product ID and type (L, M, H), air temperature, process temperature, rotational speed, torque,
  tool wear, a binary machine-failure label and five failure-mode flags (TWF, HDF, PWF, OSF, RNF).
- 339 failures (3.39%). No missing values and no duplicate rows (see `results/eda_summary.json`).
- Failure rate by product type: L 3.92%, M 2.77%, H 2.09%.

## Failure modes (as described by the dataset documentation)
- TWF: tool wear failure. HDF: heat dissipation failure (small air-process temperature gap with low rotational speed).
- PWF: power failure (mechanical power outside a band). OSF: overstrain failure (tool wear x torque above a type-specific limit).
- RNF: random failure.

## Known quirks
- 9 failures have no failure-mode flag, and 18 rows have a mode flag but no recorded failure.
- The mode flags are components of the label, so they are never used as features here.

## Intended use here
Benchmarking models, decision thresholds and a simple maintenance policy under a leakage-safe protocol. Not suitable for
estimating performance on real equipment.
