# ml-pipeline-starter

A small, **real** ML package — the example built in Module 1 of the DutchEngineer
[*Ship an End-to-End ML Product*](https://learn.dutchengineer.org) track.

It is the *logic* a data scientist's notebook contained — build a feature matrix, train
a model, score it — lifted out of the notebook and into functions you can import and
apply to **any** tabular dataset. It carries no data of its own; you bring a DataFrame
and the column lists, and the functions apply to it regardless of the dataset.

The point is not the model (the AUC is modest and the feature step has a deliberate
flaw — see below); the point is the **shape**: reusable functions instead of a notebook
you can only run top to bottom.

## Layout

```
ml-pipeline-starter/
  pyproject.toml            # declares the package + its dependencies
  src/
    ml_pipeline/
      __init__.py           # exports the feature, split, model, and artifact pieces
      features.py           # build the model matrix from any frame + column lists
      split.py              # leakage-free split: fit the transform on train rows only
      model.py              # fit a logistic regression on pre-split data, return test AUC
      artifact.py           # bundle transform + calibrated model + card; save/reload as one file
```

Nothing runs at import time, and the two functions are independent — import either one.

## Install

```bash
pip install dutchengineer-ml-pipeline
```

The import name is `ml_pipeline`; the distribution name is namespaced (the bare
`ml-pipeline` is taken on PyPI). To work on it, clone this repo and `pip install -e .`.

## Use it

Bring your own DataFrame, name the columns, and run a leakage-free split, fit, and score:

```python
from ml_pipeline import FeatureTransform, leakage_free_split, train_and_score

# df is any raw tabular frame; y is the binary label column
ft = FeatureTransform(
    numeric=["loan_amnt", "int_rate", "annual_inc"],
    categorical=["home_ownership", "purpose"],
)
X_train, X_test, y_train, y_test = leakage_free_split(df, y, ft)
auc = train_and_score(X_train, X_test, y_train, y_test)
print(f"test AUC {auc:.3f}")
```

The same calls work on Lending Club, the Adult/Census data, or any other tabular
classification set — the package applies *to* the data; it does not contain it.

### The pieces

- **`FeatureTransform` (1.1.0)** — `build_features` one-hot encodes with `get_dummies`, so
  the columns depend on the batch and a serving row differs from the training frame.
  `FeatureTransform` removes that skew: `fit` learns the levels and fills on the training
  frame, `transform` reproduces the same columns in the same order for any input (an unseen
  category maps to all-zeros).
- **`leakage_free_split` (2.0.0)** — splits the *raw* frame first, fits the transform on the
  **training rows only**, then applies it to both splits, so no test statistic leaks into
  training. This is why `train_and_score` now takes pre-split data instead of splitting
  internally — see Versioning.
- **`TrainedModel` / `train_artifact` (2.1.0)** — bundles the fitted `FeatureTransform`, a
  calibrated classifier, and a `ModelCard` (metric, calibration method, seed, library
  versions) into one object that `save`s and `load`s as a single file. Because the transform
  is bundled *inside* the artifact, the reloaded model applies the exact same feature steps it
  was trained on — `predict_proba` takes **raw** rows and returns identical scores before and
  after a save/reload, in any process.

```python
from ml_pipeline import FeatureTransform

ft = FeatureTransform(numeric=[...], categorical=[...]).fit(train_df)
X_train = ft.transform(train_df)
X_row   = ft.transform(one_request_row)   # identical columns and order, even for an
                                          # unseen category (it maps to all-zeros)
```

Train a serializable artifact and reload it without train/serve skew:

```python
from ml_pipeline import FeatureTransform, leakage_free_split, train_artifact, TrainedModel

ft = FeatureTransform(numeric=[...], categorical=[...])
X_train, X_test, y_train, y_test = leakage_free_split(df, y, ft)

model = train_artifact(X_train, y_train, ft, metric_name="roc_auc", metric_value=auc)
model.save("model.joblib")

# later, different process — same transform travels inside the file:
reloaded = TrainedModel.load("model.joblib")
scores = reloaded.predict_proba(df)   # takes RAW rows; identical to before the save
print(reloaded.card)                  # metric, calibration, seed, library versions
```

## The flaw that drove the 2.0.0 bump

Through `1.x`, the transform learned its fill values and category levels from the
**whole** frame — including the rows that become the test set — so a test statistic
leaked into training and the reported AUC was optimistic. That flaw was carried forward
unchanged from the original notebook on purpose: Module 1 was about *packaging*, not
*modeling*, so the conversion preserved the notebook's behavior, leak and all, in a
place you could actually find it.

`leakage_free_split` (2.0.0) closes it: split the raw frame first, fit the transform on
the training rows only, then apply to both. Because the model now sees an honest training
distribution, **the score changes** — which is precisely why this is a MAJOR version bump
rather than a quiet patch. It is also a breaking *API* change: `train_and_score` now takes
the four pre-split arrays (`X_train, X_test, y_train, y_test`) instead of `(X, y)`, because
the split has to happen — on raw data — before the transform is fit. The old `1.x`
`train_and_score(X, y)` is gone; that is what a MAJOR bump signals to anyone who imported it.

## Versioning

This package is versioned with [semantic versioning](https://semver.org): the version
is `MAJOR.MINOR.PATCH`, and each part means something specific.

- **MAJOR** — a breaking change to the public API (a renamed or removed function, a
  changed return shape). Bumping it tells anyone who imports the package that their code
  may need to change.
- **MINOR** — a new capability added in a backward-compatible way. Old imports still work.
- **PATCH** — a bug fix that changes no interface.

The first release is **`1.0.0`** — the first version that is actually usable: it
installs, imports, and runs end to end. It is `1.0.0`, not `0.x`, because the package
presents a stable surface (`build_features`, `train_and_score`) that the rest of the
course builds against.

Each module then adds a layer to this same package, and the version moves the way
semantic versioning says it should:

- **`1.0.0`** — `build_features` + `train_and_score` (Module 1: packaging the notebook).
- **`1.1.0`** — adds `FeatureTransform`, a fittable transform that freezes columns and
  order. Backward-compatible (old imports still work) → **MINOR**.
- **`2.0.0`** — `leakage_free_split` closes the train/test leak, which changes the scores
  *and* the API (`train_and_score` now takes pre-split arrays). Breaking → **MAJOR**.
- **`2.1.0`** — adds `TrainedModel` / `train_artifact`: a calibrated, serializable model
  that reloads and predicts identically, with a model card. Purely additive → **MINOR**.

`2.1.0` is the last version of the package itself: later modules (deployment, monitoring,
the API service) build code and configuration *around* this package rather than growing
it further, so they add no new versions. Tagged releases (`git tag v1.0.0` … `v2.1.0`)
mark each state so you can check out the package exactly as it stood at any point.
