import pytest

from src.utils.metrics import pr_auc, recall_at_precision

Y_TRUE = [0, 0, 1, 1]
Y_SCORE = [0.1, 0.4, 0.35, 0.8]


def test_pr_auc_known_value():
    assert pr_auc(Y_TRUE, Y_SCORE) == pytest.approx(5 / 6)


def test_recall_at_precision():
    assert recall_at_precision(Y_TRUE, Y_SCORE, min_precision=1.0) == 0.5
    assert recall_at_precision(Y_TRUE, Y_SCORE, min_precision=0.6) == 1.0
