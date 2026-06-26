"""ml_pipeline — a small, real ML package: load, build features, train, score.

This is the example built in Module 1 of the DutchEngineer "Ship an End-to-End ML
Product" track: a data scientist's notebook converted into a package with clear
responsibilities. `data` loads, `features` builds the model matrix, `model` trains and
scores, and `__main__` is the one entry point that runs them in order.

Importing any module runs no work; the run lives behind the `__main__` guard, so the
package is both importable for its parts and runnable as a program (`python -m ml_pipeline`).
"""

__version__ = "0.1.0"
