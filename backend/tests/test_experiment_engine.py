"""Unit tests: configuration validation, metrics, acceptance logic, runner, versioning."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.experiments import runner
from app.experiments.acceptance import check_hypothesis, decide
from app.experiments.datasets import load_splits
from app.experiments.metrics import classification_metrics, validate_metrics
from app.experiments.search_space import (
    ConfigValidationError,
    config_hash,
    default_configuration,
    neighbours,
    validate_configuration,
)
from app.experiments.versioning import GitError, GitWorkspace

GOOD = {"accuracy": 0.95, "f1": 0.95, "precision": 0.95, "recall": 0.95, "duration_seconds": 0.5,
        "peak_memory_mb": 1.0, "model_size_kb": 100.0}


# -- configuration validation ------------------------------------------------
def test_valid_configuration_fills_defaults() -> None:
    cfg = validate_configuration("random_forest", {"n_estimators": 50})
    assert cfg == {"n_estimators": 50, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"}


@pytest.mark.parametrize("model_type,cfg", [
    ("random_forest", {"n_estimators": 5000}),          # out of bounds
    ("random_forest", {"n_estimators": "many"}),        # wrong type
    ("random_forest", {"n_estimators": True}),          # bool is not a hyperparameter
    ("random_forest", {"criterion": "gini"}),           # parameter not in the allowed space
    ("logistic_regression", {"C": -1.0}),
    ("logistic_regression", {"C": float("nan")}),
    ("svm", {}),                                        # unsupported model
])
def test_invalid_configuration_rejected(model_type: str, cfg: dict) -> None:
    with pytest.raises(ConfigValidationError):
        validate_configuration(model_type, cfg)


def test_neighbours_are_bounded_unique_and_valid() -> None:
    base = default_configuration("random_forest")
    cands = neighbours("random_forest", base)
    hashes = [c.config_hash for c in cands]
    assert len(hashes) == len(set(hashes)) and config_hash("random_forest", base) not in hashes
    for c in cands:
        assert validate_configuration(c.model_type, c.configuration) == c.configuration
        assert c.moves
    edge = validate_configuration("random_forest", {"n_estimators": 10, "min_samples_leaf": 1})
    assert all(c.configuration.get("n_estimators", 10) >= 10 for c in neighbours("random_forest", edge))


def test_move_names() -> None:
    from app.experiments.search_space import moves_from_diff

    assert moves_from_diff({"max_depth": {"from": None, "to": 8}}) == ["max_depth:decrease"]
    assert moves_from_diff({"max_depth": {"from": 32, "to": None}}) == ["max_depth:increase"]
    assert moves_from_diff({"class_weight": {"from": None, "to": "balanced"}}) == ["class_weight:set_balanced"]
    assert moves_from_diff({"C": {"from": 1.0, "to": 0.316}}) == ["C:decrease"]


def test_dataset_split_is_fixed_and_read_only() -> None:
    a, b = load_splits(42), load_splits(42)
    assert a.version == b.version and np.array_equal(a.y_val, b.y_val)
    assert len(a.y_train) + len(a.y_val) + len(a.y_test) == 569
    with pytest.raises(ValueError):
        a.X_val[0, 0] = 1.0


# -- metrics ------------------------------------------------------------------
def test_metric_calculation() -> None:
    m = classification_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]))
    assert m["accuracy"] == 0.75 and m["precision"] == 1.0 and m["recall"] == 0.5
    assert math.isclose(m["f1"], 2 / 3)


def test_invalid_metrics_detected() -> None:
    assert validate_metrics(GOOD) == []
    assert validate_metrics({**GOOD, "f1": float("nan")})
    assert validate_metrics({**GOOD, "f1": 1.7})
    assert validate_metrics({k: v for k, v in GOOD.items() if k != "f1"})
    assert validate_metrics({**GOOD, "peak_memory_mb": -1})
    assert validate_metrics("not a dict")


# -- acceptance ---------------------------------------------------------------
def test_initial_baseline_accepted() -> None:
    assert decide("completed", GOOD, None, 0.001).decision == "accepted"


def test_improvement_accepted_and_regression_rejected() -> None:
    better = decide("completed", {**GOOD, "f1": 0.97}, GOOD, 0.001)
    assert better.decision == "accepted" and math.isclose(better.primary_delta, 0.02)
    worse = decide("completed", {**GOOD, "f1": 0.94}, GOOD, 0.001)
    assert worse.decision == "rejected" and worse.primary_delta < 0


def test_sub_threshold_gain_rejected() -> None:
    assert decide("completed", {**GOOD, "f1": 0.9505}, GOOD, 0.001).decision == "rejected"


def test_equal_primary_scores_rejected() -> None:
    d = decide("completed", dict(GOOD), GOOD, 0.0)
    assert d.decision == "rejected" and "tied" in d.reason


def test_rejected_experiment_preserves_memory_improvement() -> None:
    cand = {**GOOD, "f1": 0.94, "peak_memory_mb": 0.6}
    d = decide("completed", cand, GOOD, 0.001)
    assert d.decision == "rejected"
    assert [i["metric"] for i in d.secondary_improvements] == ["peak_memory_mb"]
    assert math.isclose(d.secondary_deltas["peak_memory_mb"]["delta"], -0.4)
    assert "Secondary improvement preserved" in d.reason


def test_failed_or_invalid_never_accepted() -> None:
    assert decide("failed", None, GOOD, 0.0, "boom").decision == "invalid"
    assert decide("completed", {**GOOD, "f1": float("nan")}, GOOD, 0.0).decision == "invalid"
    assert decide("completed", {**GOOD, "f1": 2.0}, None, 0.0).decision == "invalid"


def test_hypothesis_check() -> None:
    effects = [{"metric": "f1", "direction": "no_worse"}, {"metric": "peak_memory_mb", "direction": "decrease"}]
    assert check_hypothesis(effects, {**GOOD, "peak_memory_mb": 0.5}, GOOD)[0] == "supported"
    assert check_hypothesis(effects, {**GOOD, "f1": 0.9, "peak_memory_mb": 0.5}, GOOD)[0] == "partially_supported"
    assert check_hypothesis(effects, {**GOOD, "f1": 0.9, "peak_memory_mb": 2}, GOOD)[0] == "not_supported"
    assert check_hypothesis(effects, None, GOOD)[0] == "not_evaluated"


# -- runner ---------------------------------------------------------------------
def test_real_experiment_runs_and_is_reproducible() -> None:
    cfg = default_configuration("logistic_regression")
    a = runner.execute("logistic_regression", cfg, 42, mode="inprocess")
    b = runner.execute("logistic_regression", cfg, 42, mode="inprocess")
    assert a.status == "completed" and validate_metrics(a.metrics) == []
    assert a.metrics["f1"] == b.metrics["f1"]  # scores are deterministic; time/memory may vary


def test_training_crash_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a, **_k):  # noqa: ANN002, ANN003
        raise RuntimeError("simulated training crash")

    monkeypatch.setattr(runner, "build_model", boom)
    out = runner.execute("random_forest", default_configuration("random_forest"), 42, mode="inprocess")
    assert out.status == "failed" and "simulated training crash" in out.error and out.metrics is None


def test_subprocess_timeout_terminates_worker() -> None:
    out = runner.execute("random_forest", default_configuration("random_forest"), 42, mode="subprocess",
                         timeout=0.05)
    assert out.status == "timeout" and out.metrics is None


def test_subprocess_execution_completes() -> None:
    out = runner.execute("logistic_regression", default_configuration("logistic_regression"), 42,
                         mode="subprocess", timeout=60)
    assert out.status == "completed" and 0 < out.metrics["f1"] <= 1


# -- git versioning -----------------------------------------------------------------
def test_git_commits_form_experiment_dag(tmp_path) -> None:  # noqa: ANN001
    ws = GitWorkspace(tmp_path / "ws")
    root = ws.init()
    assert ws.init() == root  # idempotent
    a = ws.commit_experiment("exp_aaaaaaaaaaaa", {"configuration": {"C": 1.0}}, None, "a")
    b = ws.commit_experiment("exp_bbbbbbbbbbbb", {"configuration": {"C": 3.16}}, a, "b")
    assert ws.history(b)[1]["commit"] == a and ws.history(a)[1]["commit"] == root
    assert '"C": 3.16' in ws.diff(a, b)
    assert ws.show_config(b) == {"configuration": {"C": 3.16}}
    assert (tmp_path / "ws" / "artifacts" / "exp_bbbbbbbbbbbb" / "config.json").exists()


def test_git_handles_missing_commits_and_unsafe_input(tmp_path) -> None:  # noqa: ANN001
    ws = GitWorkspace(tmp_path / "ws")
    ws.init()
    assert not ws.commit_exists("deadbeef" * 5)
    assert not ws.commit_exists("not-a-sha; rm -rf /")
    assert ws.history("deadbeef" * 5) == []
    with pytest.raises(GitError):
        ws.commit_experiment("../../etc", {}, None, "x")
    with pytest.raises(GitError):
        ws.safe_path("..", "outside")
