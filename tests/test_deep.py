import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ipm.deep import fit_predict, make_mlp, run_seed, summarize
from tests.test_data import make_df


def test_mlp_output_shape():
    out = make_mlp(8)(torch.zeros(5, 8))
    assert out.shape == (5, 1)


def test_fit_predict_learns_a_separable_signal_and_is_reproducible():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4)).astype("float32")
    y = (X[:, 0] > 1.0).astype(int)
    a = fit_predict(X[:400], y[:400], X[400:500], y[400:500], X[500:], seed=1, max_epochs=40, patience=8)
    b = fit_predict(X[:400], y[:400], X[400:500], y[400:500], X[500:], seed=1, max_epochs=40, patience=8)
    from sklearn.metrics import average_precision_score

    assert average_precision_score(y[500:], a[1]) > 0.8
    assert np.allclose(a[1], b[1])


def test_run_seed_and_summarize_on_small_data():
    df = make_df(n=1500, rate=0.05)
    runs = {"mlp": [run_seed(df, s, False) for s in (0, 1)], "mlp+eng": [run_seed(df, s, True) for s in (0, 1)]}
    out = summarize(runs)
    assert set(out["models"]) == {"mlp", "mlp+eng"}
    assert 0 <= out["models"]["mlp"]["pr_auc"]["mean"] <= 1
