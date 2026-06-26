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
