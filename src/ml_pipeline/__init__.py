"""ml_pipeline — dataset-agnostic feature-building and model scoring.

A small library extracted from the Module 1 example of the DutchEngineer "Ship an
End-to-End ML Product" track. It is the *logic* a data scientist's notebook contained —
build a feature matrix, train a model, score it — lifted out of the notebook and into
functions you can import and apply to any tabular dataset.

It carries no data of its own: you bring a DataFrame and the column lists, and the
functions apply to it regardless of the dataset.

    from ml_pipeline import build_features, train_and_score

    X, feature_cols = build_features(df, numeric=[...], categorical=[...])
    auc = train_and_score(X, y)
"""

from ml_pipeline.features import FeatureTransform, build_features
from ml_pipeline.model import train_and_score

__version__ = "1.1.0"
__all__ = ["FeatureTransform", "build_features", "train_and_score"]
