"""Feature building — a deterministic transform that is identical at train and serve.

`build_features` (added in 1.0.0) one-hot encodes with `get_dummies`, which produces a
*different* set of columns depending on which categories happen to appear in the batch —
so a single serving row yields a different matrix than the training frame did. That is
train/serve skew, and it is the failure this module removes.

`FeatureTransform` fixes it. `fit` learns the categorical levels and the numeric fill
values from the *training* frame; `transform` applies exactly those, so any later input —
the full frame, or one incoming request row — produces the same columns in the same
order. The same fitted object is used at train time and at serve time, which is what
makes the two identical.

Dataset-agnostic: the column lists are passed in, so the transform works on any tabular
set, not just the loan data.
"""

import pandas as pd


def build_features(
    df: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Naive 1.0.0 feature build — kept for compatibility; prefer `FeatureTransform`.

    One-hot encodes with `get_dummies`, so the column set depends on the batch — not
    safe for serving. `FeatureTransform` is the train==serve-safe replacement.
    """
    df = df.copy()
    df[numeric] = df[numeric].fillna(df[numeric].mean())
    df = pd.get_dummies(df, columns=categorical)
    dummy_cols = [
        c for c in df.columns if c.startswith(tuple(f"{p}_" for p in categorical))
    ]
    feature_cols = numeric + dummy_cols
    return df[feature_cols], feature_cols


class FeatureTransform:
    """A fit-once, apply-anywhere feature transform: identical at train and serve.

    Usage::

        ft = FeatureTransform(numeric=[...], categorical=[...]).fit(train_df)
        X_train = ft.transform(train_df)
        X_row = ft.transform(one_request_row)   # same columns, same order
    """

    def __init__(self, numeric: list[str], categorical: list[str]) -> None:
        self.numeric = numeric
        self.categorical = categorical
        self._fill: dict[str, float] = {}
        self._levels: dict[str, list[str]] = {}
        self._columns: list[str] = []

    def fit(self, df: pd.DataFrame) -> "FeatureTransform":
        """Learn fill values and category levels from the TRAINING frame only."""
        self._fill = {c: float(df[c].mean()) for c in self.numeric}
        self._levels = {
            c: sorted(df[c].dropna().unique().tolist()) for c in self.categorical
        }
        # Freeze the output column order so transform always reproduces it.
        self._columns = list(self.numeric) + [
            f"{c}_{level}" for c in self.categorical for level in self._levels[c]
        ]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transform — same columns, same order, for any input."""
        if not self._columns:
            raise RuntimeError("FeatureTransform.transform called before fit")
        out = pd.DataFrame(index=df.index)
        for c in self.numeric:
            out[c] = df[c].fillna(self._fill[c])
        for c in self.categorical:
            for level in self._levels[c]:
                out[f"{c}_{level}"] = (df[c] == level).astype(int)
        return out[self._columns]
