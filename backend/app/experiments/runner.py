"""Train and evaluate one validated configuration, measuring time and memory.

Measurement notes (also in the README): ``duration_seconds`` and
``process_rss_mb`` vary between machines and runs. ``peak_memory_mb`` is the
``tracemalloc`` peak of Python/NumPy allocations during fit + predict and is far
more stable, so it is the memory metric used for comparisons. ``model_size_kb``
is the pickled model size and is deterministic for a given configuration.
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import queue
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.experiments.datasets import load_splits
from app.experiments.metrics import classification_metrics
from app.experiments.search_space import validate_configuration

PROTOCOL_VERSION = "eval-v2-cv5"  # stratified 5-fold CV on train+validation, mean binary F1
CV_FOLDS = 5
MB = 1024 * 1024


def build_model(model_type: str, configuration: dict[str, Any], seed: int):  # noqa: ANN201
    cfg = validate_configuration(model_type, configuration)
    if model_type == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", random_state=seed, **cfg))
    if model_type == "random_forest":
        # n_jobs=1 keeps results and memory measurements deterministic.
        return RandomForestClassifier(random_state=seed, n_jobs=1, **cfg)
    raise ValueError(f"unsupported model_type {model_type!r}")


def train_and_evaluate(
    model_type: str, configuration: dict[str, Any], seed: int, split: Literal["validation", "test"] = "validation"
) -> dict[str, float]:
    """``validation``: stratified k-fold CV on train+validation (scores are fold means, plus ``f1_std``).
    ``test``: fit on train+validation, score once on the held-out test split."""
    data = load_splits(seed)
    X_dev = np.concatenate([data.X_train, data.X_val])
    y_dev = np.concatenate([data.y_train, data.y_val])
    if split == "validation":
        folds = list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed).split(X_dev, y_dev))
    else:
        folds = [(None, None)]

    started_tracing = not tracemalloc.is_tracing()
    if started_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    baseline_traced, _ = tracemalloc.get_traced_memory()
    t0 = time.perf_counter()
    fold_scores: list[dict[str, float]] = []
    model = None
    for train_idx, eval_idx in folds:
        model = build_model(model_type, configuration, seed)
        if train_idx is None:
            model.fit(X_dev, y_dev)
            fold_scores.append(classification_metrics(data.y_test, model.predict(data.X_test)))
        else:
            model.fit(X_dev[train_idx], y_dev[train_idx])
            fold_scores.append(classification_metrics(y_dev[eval_idx], model.predict(X_dev[eval_idx])))
    duration = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    if started_tracing:
        tracemalloc.stop()

    metrics = {k: float(np.mean([f[k] for f in fold_scores])) for k in fold_scores[0]}
    if split == "validation":
        metrics["f1_std"] = float(np.std([f["f1"] for f in fold_scores]))
    metrics.update(
        duration_seconds=round(duration, 4),
        peak_memory_mb=round(max(0, peak - baseline_traced) / MB, 4),  # peak of a single fit (folds run sequentially)
        process_rss_mb=round(psutil.Process().memory_info().rss / MB, 2),
        model_size_kb=round(len(pickle.dumps(model)) / 1024, 2),
    )
    return metrics


@dataclass
class ExecutionOutcome:
    status: Literal["completed", "failed", "timeout", "cancelled"]
    metrics: dict[str, float] | None = None
    error: str | None = None
    wall_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def _child(result_queue: Any, model_type: str, configuration: dict[str, Any], seed: int, split: str) -> None:
    try:
        result_queue.put(("ok", train_and_evaluate(model_type, configuration, seed, split)))  # type: ignore[arg-type]
    except BaseException as exc:  # report every failure to the parent
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def execute(
    model_type: str,
    configuration: dict[str, Any],
    seed: int,
    *,
    split: Literal["validation", "test"] = "validation",
    timeout: float = 120.0,
    mode: Literal["subprocess", "inprocess"] = "subprocess",
    should_stop: Callable[[], bool] = lambda: False,
) -> ExecutionOutcome:
    """Run one experiment. ``subprocess`` mode enforces the timeout and stop requests
    by terminating the worker; ``inprocess`` mode can only flag an overrun afterwards."""
    t0 = time.monotonic()
    if mode == "inprocess":
        try:
            metrics = train_and_evaluate(model_type, configuration, seed, split)
        except Exception as exc:
            return ExecutionOutcome("failed", error=f"{type(exc).__name__}: {exc}", wall_seconds=time.monotonic() - t0)
        elapsed = time.monotonic() - t0
        if elapsed > timeout:
            return ExecutionOutcome("timeout", error=f"exceeded timeout of {timeout:.1f}s", wall_seconds=elapsed)
        return ExecutionOutcome("completed", metrics=metrics, wall_seconds=elapsed)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_child, args=(result_queue, model_type, configuration, seed, split), daemon=True)
    proc.start()
    deadline = t0 + timeout
    try:
        while True:
            try:
                kind, payload = result_queue.get(timeout=0.2)
                break
            except queue.Empty:
                if should_stop():
                    proc.terminate()
                    return ExecutionOutcome("cancelled", error="stop requested", wall_seconds=time.monotonic() - t0)
                if time.monotonic() > deadline:
                    proc.terminate()
                    return ExecutionOutcome(
                        "timeout", error=f"exceeded timeout of {timeout:.1f}s", wall_seconds=time.monotonic() - t0
                    )
                if not proc.is_alive() and result_queue.empty():
                    return ExecutionOutcome(
                        "failed",
                        error=f"worker exited with code {proc.exitcode} without a result",
                        wall_seconds=time.monotonic() - t0,
                    )
    finally:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
        result_queue.close()
    elapsed = time.monotonic() - t0
    if kind == "ok":
        return ExecutionOutcome("completed", metrics=payload, wall_seconds=elapsed)
    return ExecutionOutcome("failed", error=payload, wall_seconds=elapsed)
