"""Fixed dataset + split. Agents never touch this module's data directly.

Split: 60% train / 20% validation / 20% held-out test, stratified, fixed seed.
The held-out test split is used only for the final evaluation of a run's
final baseline — never to guide experiment selection.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

DATASET_NAME = "breast_cancer_wisconsin"


@dataclass(frozen=True)
class DatasetSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    version: str


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.setflags(write=False)  # guard against accidental mutation of evaluation data
    return a


@lru_cache(maxsize=4)
def load_splits(seed: int) -> DatasetSplits:
    X, y = load_breast_cancer(return_X_y=True)
    X_trval, X_test, y_trval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval, y_trval, test_size=0.25, stratify=y_trval, random_state=seed
    )
    digest = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()[:10]
    version = f"{DATASET_NAME}-{digest}-split60/20/20-seed{seed}-sklearn{sklearn.__version__}"
    return DatasetSplits(
        *(_readonly(a) for a in (X_train, y_train, X_val, y_val, X_test, y_test)), version=version
    )
