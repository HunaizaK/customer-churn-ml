import numpy as np

from churn.data import clean
from churn.features import ENGINEERED_NUMERIC, SERVICE_COLUMNS, add_features, build_preprocessor


def test_engineered_features_present_and_finite(raw_df):
    df = add_features(clean(raw_df))
    for col in ENGINEERED_NUMERIC + ["tenure_band"]:
        assert col in df
    assert np.isfinite(df[ENGINEERED_NUMERIC].to_numpy()).all()


def test_num_services_counts_yes_and_internet(raw_df):
    row = clean(raw_df).head(1).copy()
    row[SERVICE_COLUMNS] = "No"
    row["InternetService"] = "DSL"
    row["OnlineSecurity"] = "Yes"
    assert add_features(row)["num_services"].iloc[0] == 2  # internet + security


def test_new_customer_avg_spend_equals_monthly(raw_df):
    df = add_features(clean(raw_df))
    new = df[df["tenure"] == 0]
    assert (new["avg_monthly_spend"] == new["MonthlyCharges"]).all()


def test_preprocessor_handles_unseen_category(raw_df):
    df = add_features(clean(raw_df))
    pre = build_preprocessor().fit(df)
    unseen = df.head(1).copy()
    unseen["PaymentMethod"] = "Crypto"
    out = pre.transform(unseen)
    assert out.shape[1] == pre.transform(df.head(1)).shape[1]
