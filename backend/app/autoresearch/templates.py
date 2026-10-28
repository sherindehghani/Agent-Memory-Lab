BASELINE_TRAIN = '''from sklearn.ensemble import RandomForestClassifier


def build_model(seed):
    return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
'''

PROGRAM_TEMPLATE = '''# autoresearch program

You are an autonomous research agent improving a breast-cancer classifier.

## Goal
Maximise `val_f1`: the mean binary F1 over stratified 5-fold cross-validation, computed by the fixed harness.

## What you may change
Only `train.py`. It must define `build_model(seed)` and return an unfitted scikit-learn compatible estimator
(anything with `fit(X, y)` and `predict(X)`). Model choice, preprocessing, feature engineering, pipelines,
ensembles and hyperparameters are all fair game.

## What you may not change
- The dataset, the split, the cross-validation folds and the metric (they live in the harness).
- Allowed imports: numpy, scipy, sklearn (except sklearn.datasets), math, functools, itertools, collections,
  typing, dataclasses, numbers, statistics. No file, network, OS or dataset access.

## Rules
- Each experiment has a fixed time limit; runs that exceed it are killed and discarded.
- A change is kept only if `val_f1` improves; otherwise the branch is reset to the last kept commit.
- Simplicity wins: equal `val_f1` with less code is a keep. Small gains that add a lot of complexity are not
  worth it.
- If a run crashes, fix obvious mistakes; if the idea is fundamentally broken, move on.
- Read the results history and memories first; do not repeat ideas that were already discarded.
- Never stop to ask whether to continue; keep generating ideas until the session budget ends.
'''

HARNESS_CONTRACT = '''You are an autonomous ML research agent working in a git repository. You can only change train.py.
train.py must define build_model(seed) returning an unfitted scikit-learn compatible estimator. A fixed harness
(which you cannot change) imports build_model, trains it on each of 5 stratified folds of a 30-feature binary
tabular dataset (455 samples) and reports val_f1 (mean F1, higher is better).
Allowed imports: numpy, scipy, sklearn (not sklearn.datasets), math, functools, itertools, collections, typing,
dataclasses, numbers, statistics. Do not use open, eval, exec, getattr, dunder attributes or any I/O.
Return the COMPLETE new train.py, one focused idea per experiment, a short description (under 80 characters,
like a commit subject) and a one-sentence hypothesis.'''
