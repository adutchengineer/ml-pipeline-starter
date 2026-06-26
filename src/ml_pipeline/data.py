"""Data loading — one job: hand back the raw loan frame.

The CSV ships inside the package and is located with `importlib.resources`, so the
load works from any directory after an editable install — no `sys.path` juggling and
no hardcoded paths. Importing this module loads nothing; the read happens only when
something calls `load()`.
"""

from importlib import resources

import pandas as pd


def load() -> pd.DataFrame:
    """Return the vendored Lending Club loan sample as a DataFrame."""
    with resources.files("ml_pipeline.datasets").joinpath(
        "lending_club_sample.csv"
    ).open() as f:
        return pd.read_csv(f)
