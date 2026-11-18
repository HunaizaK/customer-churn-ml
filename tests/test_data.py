import pandas as pd
import pytest

from churn.data import FEATURE_COLUMNS, DataValidationError, clean, validate


def test_clean_fixes_blank_total_charges(raw_df):
    df = clean(raw_df)
    assert df["TotalCharges"].dtype.kind == "f"
    assert (df.loc[df["tenure"] == 0, "TotalCharges"] == 0).all()
    assert set(df["Churn"].unique()) <= {0, 1}


def test_clean_does_not_mutate_input(raw_df):
    before = raw_df.copy()
    clean(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_validate_passes_on_clean_data(raw_df):
    validate(clean(raw_df))


def test_validate_rejects_missing_column(raw_df):
    with pytest.raises(DataValidationError, match="Missing columns"):
        validate(clean(raw_df).drop(columns=["Contract"]))


def test_validate_rejects_negative_charges(raw_df):
    df = clean(raw_df)
    df.loc[0, "MonthlyCharges"] = -5
    with pytest.raises(DataValidationError, match="Negative"):
        validate(df)


def test_feature_columns_exclude_id_and_target():
    assert "customerID" not in FEATURE_COLUMNS
    assert "Churn" not in FEATURE_COLUMNS
