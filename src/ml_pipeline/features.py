"""Feature building — the notebook's munging cells, as a module.

NOTE: the mean fill is computed over the full frame, which leaks test statistics into
training. This is carried forward from the original notebook on purpose — Module 1 is
about packaging, not modeling — and is fixed in Module 3, where the fill is learned on
the training split only. It is left here as the honest starting point you improve.
"""

import pandas as pd

NUMERIC = ["loan_amnt", "int_rate", "annual_inc", "dti", "revol_util"]
CATEGORICAL = ["home_ownership", "purpose"]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Munge the raw frame into (X, y, feature_names)."""
    df = df.copy()
    df[NUMERIC] = df[NUMERIC].fillna(df[NUMERIC].mean())
    df = pd.get_dummies(df, columns=CATEGORICAL)
    dummy_cols = [
        c for c in df.columns if c.startswith(tuple(f"{p}_" for p in CATEGORICAL))
    ]
    feature_cols = NUMERIC + dummy_cols
    return df[feature_cols], df["bad_loan"], feature_cols
