import pytest

from src.utils.metrics import (
    f1_at_threshold,
    pr_auc,
    recall_at_precision,
    roc_auc,
)

Y_TRUE = [0, 0, 1, 1]
Y_SCORE = [0.1, 0.4, 0.35, 0.8]


def test_pr_auc_known_value():
    # Hand-computed: precision 1.0 at recall 0.5, then 2/3 at recall 1.0
    # AP = 0.5 * 1.0 + 0.5 * 2/3 = 0.8333...
    assert pr_auc(Y_TRUE, Y_SCORE) == pytest.approx(5 / 6)


def test_recall_at_precision():
    # Only the top score (0.8) keeps precision at 1.0 -> recall 0.5.
    assert recall_at_precision(Y_TRUE, Y_SCORE, min_precision=1.0) == 0.5
    # Threshold 0.35 gives precision 2/3 and full recall.
    assert recall_at_precision(Y_TRUE, Y_SCORE, min_precision=0.6) == 1.0


def test_roc_auc_known_value():
    # Positives 0.35, 0.8 vs negatives 0.1, 0.4: 3 of 4 pairs ranked correctly.
    assert roc_auc(Y_TRUE, Y_SCORE) == pytest.approx(0.75)


def test_f1_at_threshold_known_value():
    # Threshold 0.35 -> predictions [0, 1, 1, 1]: tp=2, fp=1, fn=0 -> F1 = 0.8.
    assert f1_at_threshold(Y_TRUE, Y_SCORE, threshold=0.35) == pytest.approx(0.8)
