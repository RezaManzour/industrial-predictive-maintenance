from ipm.eda import summarize
from tests.test_data import make_df


def test_summarize_basic_fields():
    df = make_df()
    s = summarize(df)
    assert s["rows"] == len(df)
    assert 0 <= s["failure_rate"] <= 1
    assert set(s["failure_rate_by_type"]) == {"L", "M", "H"}
