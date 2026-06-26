"""Training and scoring on a pre-split, leakage-free feature matrix.

BREAKING CHANGE in 2.0.0: `train_and_score` no longer splits internally. It takes
data that has *already* been split by `leakage_free_split` (so the transform was fit on
the training rows only), fits the model on the training matrix, and scores on the test
matrix. The old "pass the whole X,y and let it split" form is gone — because that form
fit the transform before splitting, which is the leak this version removes.

`model` and `split`/`features` stay independent of each other; the caller composes them.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def train_and_score(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> float:
    """Fit a logistic regression on the training matrix; return the held-out ROC-AUC."""
    model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, proba))
