"""Metric calculation and validation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SCORE_METRICS = ("accuracy", "f1", "precision", "recall")
RESOURCE_METRICS = ("duration_seconds", "peak_memory_mb", "process_rss_mb", "model_size_kb")
REQUIRED_METRICS = SCORE_METRICS + ("duration_seconds", "peak_memory_mb")

METRIC_UNITS = {
    "accuracy": "",
    "f1": "",
    "precision": "",
    "recall": "",
    "f1_std": "",
    "duration_seconds": "s",
    "peak_memory_mb": "MB",
    "process_rss_mb": "MB",
    "model_size_kb": "KB",
}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def validate_metrics(metrics: Any) -> list[str]:
    """Return a list of problems; empty means the metrics are usable for a decision."""
    if not isinstance(metrics, dict):
        return ["metrics must be an object"]
    errors: list[str] = []
    for name in REQUIRED_METRICS:
        if name not in metrics:
            errors.append(f"missing metric {name!r}")
    for name, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{name} is not numeric")
            continue
        if not math.isfinite(value):
            errors.append(f"{name} is not finite")
        elif name in SCORE_METRICS and not 0.0 <= value <= 1.0:
            errors.append(f"{name}={value} outside [0, 1]")
        elif name in RESOURCE_METRICS and value < 0:
            errors.append(f"{name}={value} is negative")
    return errors
