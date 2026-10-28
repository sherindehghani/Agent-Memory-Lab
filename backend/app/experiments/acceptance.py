"""Deterministic acceptance rules and hypothesis checking.

Rule (procedural memory ``acceptance_rule``):
  * the candidate must have completed and produced valid metrics;
  * with no accepted baseline yet, a valid candidate becomes the initial baseline;
  * otherwise it is accepted iff ``f1_candidate - f1_baseline > threshold``.
Ties and sub-threshold gains are rejected. Lower resource usage is recorded as a
*secondary improvement* so rejected experiments stay useful and searchable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.experiments.metrics import validate_metrics

PRIMARY_METRIC = "f1"
EPS = 1e-12

# metric -> (better direction, minimum relative change counted as a secondary improvement)
SECONDARY_METRICS: dict[str, tuple[str, float]] = {
    "peak_memory_mb": ("lower", 0.05),
    "model_size_kb": ("lower", 0.05),
    "duration_seconds": ("lower", 0.25),  # timing is noisy, so a larger margin is required
    "accuracy": ("higher", 0.0),
    "precision": ("higher", 0.0),
    "recall": ("higher", 0.0),
}
RESOURCE_IMPROVEMENT_METRICS = ("peak_memory_mb", "model_size_kb", "duration_seconds")


@dataclass
class Decision:
    decision: str  # accepted | rejected | invalid
    reason: str
    primary_delta: float | None
    secondary_deltas: dict[str, dict[str, float]] = field(default_factory=dict)
    secondary_improvements: list[dict[str, Any]] = field(default_factory=list)
    threshold: float = 0.0


def metric_deltas(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name in SECONDARY_METRICS:
        if name in candidate and name in baseline:
            delta = candidate[name] - baseline[name]
            rel = delta / baseline[name] if baseline[name] else 0.0
            out[name] = {"candidate": candidate[name], "baseline": baseline[name], "delta": round(delta, 6),
                         "relative": round(rel, 6)}
    return out


def decide(
    status: str,
    candidate: dict[str, float] | None,
    baseline: dict[str, float] | None,
    threshold: float,
    error: str | None = None,
) -> Decision:
    if status != "completed":
        return Decision("invalid", f"Execution {status}: {error or 'no result'}; baseline unchanged.", None,
                        threshold=threshold)
    problems = validate_metrics(candidate)
    if problems:
        return Decision("invalid", f"Invalid metrics ({'; '.join(problems)}); baseline unchanged.", None,
                        threshold=threshold)
    assert candidate is not None
    if baseline is None:
        return Decision("accepted", "No accepted baseline exists; this valid result becomes the initial baseline.",
                        None, threshold=threshold)

    delta = candidate[PRIMARY_METRIC] - baseline[PRIMARY_METRIC]
    deltas = metric_deltas(candidate, baseline)
    improvements = [
        {"metric": name, **d}
        for name, d in deltas.items()
        if name in RESOURCE_IMPROVEMENT_METRICS and d["relative"] <= -SECONDARY_METRICS[name][1]
    ]
    if delta > threshold + EPS:
        reason = f"Cross-validated F1 improved by {delta:+.4f} (> threshold {threshold})."
        return Decision("accepted", reason, delta, deltas, improvements, threshold)
    if abs(delta) <= EPS:
        reason = "Cross-validated F1 tied with the baseline (delta 0); ties never replace the baseline."
    elif delta > 0:
        reason = f"Cross-validated F1 improved by {delta:+.4f}, which does not exceed the threshold {threshold}."
    else:
        reason = f"Cross-validated F1 decreased by {delta:+.4f}."
    if improvements:
        parts = [f"{i['metric']} {i['delta']:+.3f} ({i['relative']:+.0%})" for i in improvements]
        reason += " Secondary improvement preserved: " + ", ".join(parts) + "."
    return Decision("rejected", reason, delta, deltas, improvements, threshold)


def check_hypothesis(
    expected_effects: list[dict[str, Any]], candidate: dict[str, float] | None, baseline: dict[str, float] | None
) -> tuple[str, list[dict[str, Any]]]:
    """Compare each expected effect with the measured change vs. the baseline.

    Directions: ``increase``, ``decrease`` (strict) and ``no_worse`` (for F1: >=).
    """
    if not expected_effects:
        return "not_evaluated", []
    if candidate is None or baseline is None:
        return "not_evaluated", [{"note": "no valid measurement or no baseline to compare against"}]
    details: list[dict[str, Any]] = []
    for effect in expected_effects:
        metric, direction = effect["metric"], effect["direction"]
        if metric not in candidate or metric not in baseline:
            details.append({**effect, "met": False, "note": "metric missing"})
            continue
        delta = candidate[metric] - baseline[metric]
        lower_is_better = SECONDARY_METRICS.get(metric, ("higher", 0))[0] == "lower"
        if direction == "increase":
            met = delta > EPS
        elif direction == "decrease":
            met = delta < -EPS
        else:  # no_worse
            met = delta <= EPS if lower_is_better else delta >= -EPS
        details.append({**effect, "met": met, "delta": round(delta, 6)})
    met_count = sum(1 for d in details if d.get("met"))
    if met_count == len(details):
        return "supported", details
    if met_count == 0:
        return "not_supported", details
    return "partially_supported", details
