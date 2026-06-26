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
      __init__.py           # exports build_features, train_and_score
      features.py           # build the model matrix from any frame + column lists
      model.py              # split, fit a logistic regression, return the test AUC
```

Nothing runs at import time, and the two functions are independent — import either one.

## Install

```bash
pip install dutchengineer-ml-pipeline
```

The import name is `ml_pipeline`; the distribution name is namespaced (the bare
`ml-pipeline` is taken on PyPI). To work on it, clone this repo and `pip install -e .`.

## Use it

Bring your own DataFrame and say which columns are numeric and which are categorical:

```python
from ml_pipeline import build_features, train_and_score

# df is any tabular frame; y is the binary label column
X, feature_cols = build_features(
    df,
    numeric=["loan_amnt", "int_rate", "annual_inc"],
    categorical=["home_ownership", "purpose"],
)
auc = train_and_score(X, y)
print(f"{len(feature_cols)} features, test AUC {auc:.3f}")
```

The same two calls work on Lending Club, the Adult/Census data, or any other tabular
classification set — the package applies *to* the data; it does not contain it.

## A known flaw, on purpose

`features.py` fills missing values with the mean of the **whole** frame, including the
rows that become the test set — so a test statistic leaks into training. It is left in
place because Module 1 is about *packaging*, not *modeling*; Module 3 fixes it by
fitting the fill on the training split only. Converting the notebook faithfully means
carrying its behavior forward unchanged, flaws included, so you can fix them one at a
time in a place you can actually find them.

## Versioning

This package is versioned with [semantic versioning](https://semver.org): the version
is `MAJOR.MINOR.PATCH`, and each part means something specific.

- **MAJOR** — a breaking change to the public API (a renamed or removed function, a
  changed return shape). Bumping it tells anyone who imports the package that their code
  may need to change.
- **MINOR** — a new capability added in a backward-compatible way. Old imports still work.
- **PATCH** — a bug fix that changes no interface.

This release is **`1.0.0`** — the first version that is actually usable: it installs,
imports, and runs end to end. It is `1.0.0`, not `0.x`, because the package presents a
stable surface (`load`, `build_features`, `train_and_score`, `python -m ml_pipeline`)
that the rest of the course builds against.

As the course continues, each module adds a backward-compatible layer to this same
package — a real feature transform, a typed contract, a locked environment — so the
version moves by **MINOR** bumps: `1.1.0`, `1.2.0`, and so on. The day a change breaks
the public surface (for example, the Module 3 fix that changes how features are built
and therefore the scores), that is a **MAJOR** bump to `2.0.0` — and the version number
is how a downstream user knows to expect it. Tagged releases (`git tag v1.0.0`) mark
each of these states so you can check out the package exactly as it stood at any point.
