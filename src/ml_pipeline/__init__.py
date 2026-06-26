"""ml_pipeline — dataset-agnostic feature-building and model scoring.

A small library extracted from the Module 1 example of the DutchEngineer "Ship an
End-to-End ML Product" track. It is the *logic* a data scientist's notebook contained —
build a feature matrix, train a model, score it — lifted out of the notebook and into
functions you can import and apply to any tabular dataset.

It carries no data of its own: you bring a DataFrame and the column lists, and the
functions apply to it regardless of the dataset.

    from ml_pipeline import FeatureTransform, leakage_free_split, train_artifact

    ft = FeatureTransform(numeric=[...], categorical=[...])
    X_train, X_test, y_train, y_test = leakage_free_split(df, y, ft)
    model = train_artifact(X_train, y_train, ft, metric_name="roc_auc", metric_value=auc)
    model.save("model.joblib")
"""

from ml_pipeline.artifact import ModelCard, TrainedModel, train_artifact
from ml_pipeline.features import FeatureTransform, build_features
from ml_pipeline.model import train_and_score
from ml_pipeline.split import leakage_free_split

__version__ = "2.1.0"
__all__ = [
    "FeatureTransform",
    "ModelCard",
    "TrainedModel",
    "build_features",
    "leakage_free_split",
    "train_and_score",
    "train_artifact",
]
