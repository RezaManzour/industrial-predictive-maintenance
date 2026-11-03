from ipm.metrics import imbalance_metrics


def test_perfect_scores():
    m = imbalance_metrics([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.8])
    assert m["pr_auc"] == 1.0 and m["recall"] == 1.0 and m["precision"] == 1.0


def test_no_positive_predictions_is_safe():
    m = imbalance_metrics([0, 1], [0.1, 0.2])
    assert m["precision"] == 0.0 and m["recall"] == 0.0
