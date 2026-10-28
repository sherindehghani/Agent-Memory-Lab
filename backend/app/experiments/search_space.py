"""Allowed model types, hyperparameter bounds, and bounded neighbour moves.

Agents never edit source code: they propose a ``(model_type, configuration)``
pair which must validate against these schemas before anything runs.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ModelType = Literal["logistic_regression", "random_forest"]
MODEL_TYPES: tuple[str, ...] = ("logistic_regression", "random_forest")


class _StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _reject_bools(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, bool):
                    raise ValueError(f"{key}: boolean values are not valid hyperparameters")
        return data


class LogisticRegressionConfig(_StrictConfig):
    C: float = Field(1.0, ge=1e-4, le=1e3, allow_inf_nan=False)
    max_iter: int = Field(1000, ge=100, le=5000)
    class_weight: Literal["balanced"] | None = None


class RandomForestConfig(_StrictConfig):
    n_estimators: int = Field(200, ge=10, le=500)
    max_depth: int | None = Field(None, ge=1, le=50)
    min_samples_leaf: int = Field(1, ge=1, le=50)
    max_features: Literal["sqrt", "log2"] = "sqrt"


CONFIG_SCHEMAS: dict[str, type[_StrictConfig]] = {
    "logistic_regression": LogisticRegressionConfig,
    "random_forest": RandomForestConfig,
}

# Moves expected to shrink the trained model (used for hypotheses only — never as evidence).
DOWNSIZING_MOVES = {"n_estimators:decrease", "max_depth:decrease", "min_samples_leaf:increase"}
UPSIZING_MOVES = {"n_estimators:increase", "max_depth:increase", "min_samples_leaf:decrease"}


class ConfigValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def validate_configuration(model_type: str, configuration: Any) -> dict[str, Any]:
    """Return a normalised configuration (defaults filled) or raise ConfigValidationError."""
    if model_type not in CONFIG_SCHEMAS:
        raise ConfigValidationError([f"unsupported model_type {model_type!r}; allowed: {list(MODEL_TYPES)}"])
    if not isinstance(configuration, dict):
        raise ConfigValidationError(["configuration must be an object"])
    try:
        return CONFIG_SCHEMAS[model_type].model_validate(configuration).model_dump()
    except ValidationError as exc:
        raise ConfigValidationError(
            [f"{'.'.join(str(p) for p in e['loc']) or 'configuration'}: {e['msg']}" for e in exc.errors()]
        ) from None


def default_configuration(model_type: str) -> dict[str, Any]:
    return CONFIG_SCHEMAS[model_type]().model_dump()


def config_hash(model_type: str, configuration: dict[str, Any]) -> str:
    canonical = json.dumps({"model_type": model_type, "configuration": configuration}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def search_space_description() -> dict[str, Any]:
    """JSON description of the allowed space (stored as procedural memory, sent to the LLM)."""
    return {name: schema.model_json_schema()["properties"] for name, schema in CONFIG_SCHEMAS.items()}


def config_diff(old_type: str | None, old: dict[str, Any] | None, new_type: str, new: dict[str, Any]) -> dict[str, Any]:
    if old is None or old_type is None:
        return {}
    if old_type != new_type:
        return {"model_type": {"from": old_type, "to": new_type}}
    return {k: {"from": old.get(k), "to": v} for k, v in new.items() if old.get(k) != v}


def _direction(old: Any, new: Any) -> str:
    if old is None and isinstance(new, (int, float)):
        return "decrease"  # max_depth: unlimited -> bounded
    if new is None and isinstance(old, (int, float)):
        return "increase"  # max_depth: bounded -> unlimited
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        return "increase" if new > old else "decrease"
    return f"set_{new}"


def moves_from_diff(diff: dict[str, Any]) -> list[str]:
    """Named moves such as ``n_estimators:decrease`` or ``model_type:set_random_forest``."""
    return sorted(f"{param}:{_direction(change['from'], change['to'])}" for param, change in diff.items())


@dataclass
class Candidate:
    model_type: str
    configuration: dict[str, Any]
    diff: dict[str, Any] = field(default_factory=dict)

    @property
    def moves(self) -> list[str]:
        return moves_from_diff(self.diff)

    @property
    def config_hash(self) -> str:
        return config_hash(self.model_type, self.configuration)


def _round_sig(x: float, sig: int = 3) -> float:
    return float(f"{x:.{sig}g}")


def neighbours(model_type: str, configuration: dict[str, Any]) -> list[Candidate]:
    """Bounded one-step moves from a configuration, in a fixed deterministic order."""
    proposals: list[tuple[str, dict[str, Any]]] = []
    c = dict(configuration)
    if model_type == "random_forest":
        n, d, leaf = c["n_estimators"], c["max_depth"], c["min_samples_leaf"]
        proposals += [
            ("random_forest", {**c, "n_estimators": max(10, n // 2)}),
            ("random_forest", {**c, "n_estimators": min(500, n * 2)}),
            ("random_forest", {**c, "max_depth": 8 if d is None else max(1, d // 2)}),
            ("random_forest", {**c, "max_depth": None if d is None or d * 2 > 50 else d * 2}),
            ("random_forest", {**c, "min_samples_leaf": min(50, leaf * 2)}),
            ("random_forest", {**c, "min_samples_leaf": max(1, leaf // 2)}),
            ("random_forest", {**c, "max_features": "log2" if c["max_features"] == "sqrt" else "sqrt"}),
            ("logistic_regression", default_configuration("logistic_regression")),
        ]
    elif model_type == "logistic_regression":
        step = math.sqrt(10)
        proposals += [
            ("logistic_regression", {**c, "C": _round_sig(min(1e3, c["C"] * step))}),
            ("logistic_regression", {**c, "C": _round_sig(max(1e-4, c["C"] / step))}),
            ("logistic_regression", {**c, "class_weight": None if c["class_weight"] else "balanced"}),
            ("random_forest", default_configuration("random_forest")),
        ]
    else:
        raise ConfigValidationError([f"unsupported model_type {model_type!r}"])

    seen: set[str] = {config_hash(model_type, configuration)}
    out: list[Candidate] = []
    for new_type, new_cfg in proposals:
        try:
            new_cfg = validate_configuration(new_type, new_cfg)
        except ConfigValidationError:
            continue
        cand = Candidate(new_type, new_cfg, config_diff(model_type, configuration, new_type, new_cfg))
        if cand.config_hash in seen:
            continue
        seen.add(cand.config_hash)
        out.append(cand)
    return out


def describe_moves(moves: list[str]) -> str:
    return ", ".join(moves) if moves else "no change"
