# ml-pipeline-starter

A small, **real** ML package — the example built in Module 1 of the DutchEngineer
[*Ship an End-to-End ML Product*](https://learn.dutchengineer.org) track.

It is what a data scientist's notebook becomes once it is converted into a package you
can install, import, and run: load the loan data, build features, train a logistic
regression, and print a test ROC-AUC. The point is not the model (the AUC is modest and
the feature step has a deliberate flaw — see below); the point is the **shape**.

## Layout

```
ml-pipeline-starter/
  pyproject.toml            # declares the package + its dependencies
  src/
    ml_pipeline/
      __init__.py
      data.py               # load the data (one job)
      features.py           # build the model matrix
      model.py              # split, fit, score
      __main__.py           # the entry point: python -m ml_pipeline
      datasets/
        lending_club_sample.csv
```

The import edges run one way — `__main__` → `model` → `features` → `data` — so you can
import any piece without dragging in the rest. Nothing runs at import time; the work
lives behind the `__main__` guard.

## Run it

```bash
pip install -e .          # editable install, so imports resolve from anywhere
python -m ml_pipeline     # load → features → train → score
```

You should see something like:

```
rows: 5000
features: 24
test AUC: 0.691
```

## Use it from anywhere

Because it is a package, you can import its parts — in a test, a script, or a notebook —
without rerunning everything:

```python
from ml_pipeline.data import load
from ml_pipeline.features import build_features

df = load()
X, y, feature_cols = build_features(df)   # just the features, no training fires
```

## A known flaw, on purpose

`features.py` fills missing values with the mean of the **whole** frame, including the
rows that become the test set — so a test statistic leaks into training. It is left in
place because Module 1 is about *packaging*, not *modeling*; Module 3 fixes it by
fitting the fill on the training split only. Converting the notebook faithfully means
carrying its behavior forward unchanged, flaws included, so you can fix them one at a
time in a place you can actually find them.
