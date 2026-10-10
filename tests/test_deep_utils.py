import numpy as np
import pandas as pd
import pytest

from src.models.deep_utils import EarlyStopping, Standardizer, encode_type


def test_encode_type_maps_and_rejects_unknown():
    assert list(encode_type(pd.Series(["L", "M", "H"]))) == [0, 1, 2]
    with pytest.raises(ValueError, match="Unknown Type"):
        encode_type(pd.Series(["X"]))


def test_standardizer_uses_train_statistics_only():
    train = np.array([[0.0], [2.0]])  # mean 1, std 1
    test = np.array([[3.0]])
    s = Standardizer().fit(train)
    np.testing.assert_allclose(s.transform(test), [[2.0]])


def test_early_stopping_stops_after_patience_and_keeps_best_epoch():
    stopper = EarlyStopping(patience=2)
    scores = [0.5, 0.6, 0.59, 0.58]
    stops = [stopper.step(s, e) for e, s in enumerate(scores)]
    assert stops == [False, False, False, True]
    assert stopper.best_epoch == 1
