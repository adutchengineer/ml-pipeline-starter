"""Feature building — turn a raw frame into a model matrix.

Dataset-agnostic: you pass the frame and say which columns are numeric and which are
categorical. The transform fills missing numerics, one-hot encodes the categoricals,
and returns the feature matrix plus the column names it produced.

NOTE: the mean fill is computed over the whole frame, which leaks test statistics into
training. Carried forward from the original notebook on purpose (Module 1 is about
packaging, not modeling); Module 3 fixes it by fitting the fill on the training split
only. It is left here as the honest starting point you improve.
"""

import pandas as pd


def build_features(
    df: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Build the model matrix from `df`.

    Returns (X, feature_names). The label column is the caller's concern — pass the
    frame without it, or drop it before calling — so this stays purely about features.
    """
    df = df.copy()
    df[numeric] = df[numeric].fillna(df[numeric].mean())
    df = pd.get_dummies(df, columns=categorical)
    dummy_cols = [
        c for c in df.columns if c.startswith(tuple(f"{p}_" for p in categorical))
    ]
    feature_cols = numeric + dummy_cols
    return df[feature_cols], feature_cols
