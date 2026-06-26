"""The entry point — what `python -m ml_pipeline` runs.

This is the only place the work fires. Importing any module above runs nothing; the
load-features-train-score sequence happens here, behind the guard.
"""

from ml_pipeline.data import load
from ml_pipeline.features import build_features
from ml_pipeline.model import train_and_score


def main() -> None:
    df = load()
    x, y, feature_cols = build_features(df)
    auc = train_and_score(x, y)
    print(f"rows: {len(df)}")
    print(f"features: {len(feature_cols)}")
    print(f"test AUC: {auc:.3f}")


if __name__ == "__main__":
    main()
