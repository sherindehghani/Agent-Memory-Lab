from __future__ import annotations

import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FOLDS = 5
MB = 1024 * 1024
RESULT_PREFIX = "RESULT_JSON:"
BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _load_build_model(path: str) -> Callable[[int], Any]:
    spec = importlib.util.spec_from_file_location("train", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build_model = getattr(module, "build_model", None)
    if not callable(build_model):
        raise TypeError("train.py must define build_model(seed)")
    return build_model


def evaluate(path: str, seed: int, split: str = "validation") -> dict[str, float]:
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    from app.experiments.datasets import load_splits
    from app.experiments.metrics import classification_metrics

    build_model = _load_build_model(path)
    data = load_splits(seed)
    X_dev = np.concatenate([data.X_train, data.X_val])
    y_dev = np.concatenate([data.y_train, data.y_val])
    tracemalloc.start()
    t0 = time.perf_counter()
    scores = []
    model = None
    if split == "validation":
        for train_idx, eval_idx in StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(X_dev, y_dev):
            model = build_model(seed)
            model.fit(X_dev[train_idx].copy(), y_dev[train_idx].copy())
            scores.append(classification_metrics(y_dev[eval_idx], np.asarray(model.predict(X_dev[eval_idx].copy()))))
    else:
        model = build_model(seed)
        model.fit(X_dev.copy(), y_dev.copy())
        scores.append(classification_metrics(data.y_test, np.asarray(model.predict(data.X_test.copy()))))
    duration = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    metrics = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
    if split == "validation":
        metrics["f1_std"] = float(np.std([s["f1"] for s in scores]))
    metrics["duration_seconds"] = round(duration, 4)
    metrics["peak_memory_mb"] = round(peak / MB, 4)
    try:
        metrics["model_size_kb"] = round(len(pickle.dumps(model)) / 1024, 2)
    except Exception:
        pass
    return metrics


def main(argv: list[str]) -> int:
    path, seed = argv[1], int(argv[2])
    split = argv[3] if len(argv) > 3 else "validation"
    metrics = evaluate(path, seed, split)
    for key, value in metrics.items():
        sys.stdout.write(f"{key}: {value:.6f}\n")
    sys.stdout.write(RESULT_PREFIX + json.dumps(metrics) + "\n")
    sys.stdout.flush()
    return 0


@dataclass
class HarnessOutcome:
    status: str
    metrics: dict[str, float] | None
    error: str | None
    log_tail: str
    wall_seconds: float


def run_harness(source: str, seed: int, *, split: str = "validation", timeout: float = 120.0,
                should_stop: Callable[[], bool] = lambda: False, log_path: Path | None = None) -> HarnessOutcome:
    t0 = time.monotonic()
    tmp = Path(tempfile.mkdtemp(prefix="autoresearch-"))
    try:
        train_path = tmp / "train.py"
        train_path.write_text(source)
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(BACKEND_ROOT), "HOME": str(tmp),
               "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        log_file = tmp / "run.log"
        with open(log_file, "w") as log:
            proc = subprocess.Popen([sys.executable, "-m", "app.autoresearch.harness", str(train_path), str(seed),
                                     split], cwd=tmp, env=env, stdout=log, stderr=subprocess.STDOUT)
            status = None
            while proc.poll() is None:
                if should_stop():
                    proc.kill()
                    status = "cancelled"
                    break
                if time.monotonic() - t0 > timeout:
                    proc.kill()
                    status = "timeout"
                    break
                time.sleep(0.1)
            proc.wait()
        text = log_file.read_text(errors="replace")
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(text)
        tail = "\n".join(text.splitlines()[-50:])
        wall = time.monotonic() - t0
        if status == "cancelled":
            return HarnessOutcome("cancelled", None, "stop requested", tail, wall)
        if status == "timeout":
            return HarnessOutcome("timeout", None, f"exceeded the {timeout:.0f}s time limit", tail, wall)
        results = [line for line in text.splitlines() if line.startswith(RESULT_PREFIX)]
        if proc.returncode != 0 or not results:
            last = tail.strip().splitlines()[-1] if tail.strip() else f"exit code {proc.returncode}"
            return HarnessOutcome("failed", None, last[:500], tail, wall)
        return HarnessOutcome("completed", json.loads(results[-1][len(RESULT_PREFIX):]), None, tail, wall)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
