# Report: predictive maintenance on AI4I 2020

## Question
How well can failures of a milling machine be predicted from five sensor readings and the product type, how do
classical and neural models compare under one protocol, and how should predictions be turned into maintenance
decisions when errors have different costs?

## Data
AI4I 2020: 10,000 records, 339 failures (3.39%), no missing values or duplicates. Failure rate by product type is
3.92% (L), 2.77% (M) and 2.09% (H). Linear correlation with failure is weak (torque 0.19, tool wear 0.11), so the
relationships are non-linear. The failure-mode flags (TWF, HDF, PWF, OSF, RNF) and identifiers are excluded from the
features because they leak the label. Nine failures have no mode flag and 18 records have a flag without a failure.
The data are synthetic and generated from threshold rules.

## Protocol
Stratified 70/15/15 split. Models are fitted on train, thresholds and early stopping use validation only, and the
test split is evaluated once per run. The primary metric is PR-AUC because of the class imbalance. The protocol is
repeated over five split seeds, and models are compared on identical splits.

## Models
Logistic regression, random forest and histogram gradient boosting with class weighting, each with and without three
engineered features (mechanical power, process-air temperature gap, wear x torque). A PyTorch MLP (64-32-1, dropout,
class-weighted loss, early stopping) without tuning.

## Results
See the tables in the README. Over five seeds, tree models with engineered features reach PR-AUC of about 0.87-0.88.
The engineered features improve random forest by 0.148 and gradient boosting by 0.090 PR-AUC on average, better in 5
of 5 seeds. The MLP reaches 0.73-0.75 and is behind gradient boosting in all seeds. Random forest and gradient
boosting cannot be ranked from these results. A single split (seed 42) gave PR-AUC about 0.94, which was a
favourable split; the multi-seed mean is the number to cite.

## Decisions
With assumed costs for a missed failure and an unnecessary action, a cost-optimal threshold chosen on validation data
lowers expected cost compared with the F1-optimal threshold at cost ratios 10 and 20, and is similar at ratio 5.
A tool-replacement MDP over wear bins, with hazards estimated from the data, is solved exactly by value iteration and
with tabular Q-learning. The exact optimum replaces the tool at about 180-200 minutes of wear, a simple fixed rule is
nearly as good, and Q-learning is 2-12% more expensive than the exact optimum.

## What this does and does not show
The experiments show that the generating rules of this synthetic dataset can be recovered with physically motivated
features and tree models, and that cost-aware decisions matter when error costs differ. They do not show performance on
real machines. The test sets are small, the MLP is untuned, costs are assumed, and the hazard model rests on treating
independent snapshots as per-cycle failure probabilities.

## Reproducing
See the Setup section of the README. All results are produced by code in `src/ipm` and stored in `results/`.
