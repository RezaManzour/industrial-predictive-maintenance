import numpy as np
import pytest

from src.models.classical import best_f1_threshold, scale_pos_weight


def test_scale_pos_weight_uses_train_labels_only():
    y_train = np.array([0] * 97 + [1] * 3)
    assert scale_pos_weight(y_train) == pytest.approx(97 / 3)


def test_best_f1_threshold_known_value():
    # F1 is 2/3 at 0.8, 0.5 at 0.4 and 0.8 at 0.35, so 0.35 is the best cut.
    threshold, f1 = best_f1_threshold([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert threshold == pytest.approx(0.35)
    assert f1 == pytest.approx(0.8)
