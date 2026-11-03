from ipm.multiseed import MODELS, run_seed, summarize
from tests.test_data import make_df


def test_multiseed_summary_structure():
    df = make_df(n=1500, rate=0.05)
    per_seed = [run_seed(df, s) for s in (0, 1, 2)]
    out = summarize(per_seed)
    assert out["n_seeds"] == 3
    assert set(out["paired_eng_minus_base"]) == set(MODELS)
    for v in out["paired_eng_minus_base"].values():
        assert 0 <= v["seeds_where_eng_better"] <= 3
    assert 0 <= out["paired_rf_minus_hgb_eng"]["seeds_where_rf_better"] <= 3


def test_different_seeds_give_different_splits():
    from ipm.data import train_val_test_split

    df = make_df()
    a = train_val_test_split(df, seed=0)[2]["UDI"].tolist()
    b = train_val_test_split(df, seed=1)[2]["UDI"].tolist()
    assert a != b
