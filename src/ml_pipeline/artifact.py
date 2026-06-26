"""The trained model as a single, reloadable artifact.

Through 2.0.0 the package builds features and scores a model, but the trained thing only
ever lives in memory for the length of one run. A served model has to *persist*: train
once, save one file, and reload it later — in a different process, on a different machine —
and get byte-identical predictions. That is what `TrainedModel` adds.

It bundles the fitted `FeatureTransform`, a calibrated classifier, and a small model card
(metric, calibration method, seed, library versions) into one object. `save`/`load`
round-trip it with joblib; because the transform is bundled, the reloaded artifact applies
the exact same feature steps the model was trained on — no train/serve skew across the save.
"""

import sys
from dataclasses import dataclass, field

import joblib
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression

from ml_pipeline.features import FeatureTransform


@dataclass
class ModelCard:
    """What the artifact records about itself, for reproducibility."""

    metric_name: str
    metric_value: float
    calibration: str
    seed: int
    versions: dict[str, str] = field(default_factory=dict)


@dataclass
class TrainedModel:
    """A fitted transform + calibrated classifier + card, saved and loaded as one file."""

    transform: FeatureTransform
    classifier: CalibratedClassifierCV
    card: ModelCard

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """Score raw rows: apply the bundled transform, then the calibrated model."""
        x = self.transform.transform(df)
        return pd.Series(self.classifier.predict_proba(x)[:, 1], index=df.index)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "TrainedModel":
        # Loads an artifact this code base produced (joblib of our own object) — not
        # untrusted input, so the load is safe.
        return joblib.load(path)


def train_artifact(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    transform: FeatureTransform,
    metric_name: str,
    metric_value: float,
    seed: int = 42,
) -> TrainedModel:
    """Fit a calibrated logistic regression and bundle it with the fitted transform."""
    base = LogisticRegression(max_iter=1000, random_state=seed).fit(x_train, y_train)
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    calibrated.fit(x_train, y_train)
    card = ModelCard(
        metric_name=metric_name,
        metric_value=metric_value,
        calibration="isotonic",
        seed=seed,
        versions={"python": sys.version.split()[0], "sklearn": sklearn.__version__},
    )
    return TrainedModel(transform=transform, classifier=calibrated, card=card)
