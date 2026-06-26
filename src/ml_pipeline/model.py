"""Training and scoring — the split-and-fit cells, as a module.

The dependency edges run one way: `__main__` -> `model` -> `features` -> `data`, so a
test can import `model` to train without dragging in the entry point.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42


def train_and_score(x: pd.DataFrame, y: pd.Series) -> float:
    """Split, fit a logistic regression, and return the held-out ROC-AUC."""
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED
    )
    model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, proba))
