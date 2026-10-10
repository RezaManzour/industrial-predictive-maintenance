import pytest

from src.utils.metrics import pr_auc, recall_at_precision

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
