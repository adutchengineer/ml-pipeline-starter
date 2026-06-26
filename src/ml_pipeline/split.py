"""Leakage-free splitting — fit the transform on the training rows only.

Through 1.x, the transform learned its fill values and category levels from the *whole*
frame and only then split — so statistics from the test rows (their means, their
categories) leaked into training, and the reported score was optimistic. This module
fixes the order: split the RAW frame first, fit the transform on the training split
only, then apply the fitted transform to both splits.

This changes the numbers a model scores, which is exactly why the release that adds it
is a MAJOR version bump (2.0.0): the same inputs no longer produce the same AUC.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_pipeline.features import FeatureTransform

SEED = 42


def leakage_free_split(
    df: pd.DataFrame,
    y: pd.Series,
    transform: FeatureTransform,
    test_size: float = 0.2,
    random_state: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split first, fit the transform on TRAIN only, apply to both. Returns
    (X_train, X_test, y_train, y_test) with no test statistics in the training matrix."""
    raw_train, raw_test, y_train, y_test = train_test_split(
        df, y, test_size=test_size, random_state=random_state, stratify=y
    )
    transform.fit(raw_train)  # learns fills + levels from training rows only
    x_train = transform.transform(raw_train)
    x_test = transform.transform(raw_test)
    return x_train, x_test, y_train, y_test
