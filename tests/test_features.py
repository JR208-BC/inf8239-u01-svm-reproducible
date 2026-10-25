import numpy as np
import pandas as pd
import pytest

from inf8239_u01.features import (CATEGORICAL_COLUMNS, DROP_COLUMNS, TARGET,
                                  build_preprocessor, split_features_target)


def toy_frame(n=40):
    rng = np.random.default_rng(0)
    cols = {c: rng.integers(0, 3, n) for c in CATEGORICAL_COLUMNS}
    cols["Admission grade"] = rng.normal(130, 10, n)
    cols["Curricular units 1st sem (grade)"] = rng.normal(12, 2, n)
    cols["Debtor"] = rng.integers(0, 2, n)
    cols["Curricular units 2nd sem (grade)"] = rng.normal(12, 2, n)
    cols[TARGET] = rng.choice(["Dropout", "Graduate", "Enrolled"], n)
    return pd.DataFrame(cols)


def test_split_removes_target_and_leakage_columns():
    X, y = split_features_target(toy_frame())
    assert TARGET not in X.columns
    assert not set(DROP_COLUMNS) & set(X.columns)
    assert len(X) == len(y)


def test_split_requires_target():
    with pytest.raises(KeyError):
        split_features_target(toy_frame().drop(columns=[TARGET]))


def test_split_rejects_missing_target():
    frame = toy_frame()
    frame.loc[0, TARGET] = None
    with pytest.raises(ValueError):
        split_features_target(frame)


def test_preprocessor_learns_scaling_only_from_training_data():
    X, _ = split_features_target(toy_frame())
    train, test = X.iloc[:30], X.iloc[30:]
    prep = build_preprocessor(X).fit(train)
    num_cols = next(cols for name, _, cols in prep.transformers_ if name == "num")
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    assert np.allclose(scaler.mean_[num_cols.index("Admission grade")],
                       train["Admission grade"].mean())
    assert not np.isclose(train["Admission grade"].mean(), test["Admission grade"].mean())
