"""API request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.experiments.search_space import ConfigValidationError, default_configuration, validate_configuration

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class RunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str = Field("Improve validation F1 score", min_length=3, max_length=500)
    memory_enabled: bool = True
    planner_mode: Literal["deterministic", "llm"] = "deterministic"
    max_experiments: int | None = Field(None, ge=1, le=50)
    max_duration_seconds: int | None = Field(None, ge=10, le=3600)
    max_retries: int | None = Field(None, ge=0, le=3)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    initial_model_type: Literal["logistic_regression", "random_forest"] = "random_forest"
    initial_configuration: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_initial(self) -> RunCreate:
        try:
            self.initial_configuration = validate_configuration(
                self.initial_model_type, self.initial_configuration or default_configuration(self.initial_model_type)
            )
        except ConfigValidationError as exc:
            raise ValueError(f"initial_configuration invalid: {exc}") from None
        return self


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    namespace: str
    objective: str
    state: str
    memory_enabled: bool
    planner_mode: str
    seed: int
    max_experiments: int
    max_duration_seconds: int
    max_retries: int
    initial_model_type: str
    initial_configuration: dict[str, Any]
    stop_requested: bool
    parent_run_id: str | None
    start_baseline_experiment_id: str | None
    end_baseline_experiment_id: str | None
    stop_reason: str | None
    error: str | None
    summary: dict[str, Any] | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class ExperimentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    run_id: str
    namespace: str
    sequence: int
    parent_experiment_id: str | None
    baseline_experiment_id: str | None
    proposed_by: str
    executed_by: str | None
    objective: str
    hypothesis: str
    expected_effects: list[dict[str, Any]]
    reasoning_summary: str
    plan_source: str
    referenced_memory_ids: list[str]
    model_type: str
    configuration: dict[str, Any]
    config_hash: str
    config_diff: dict[str, Any]
    moves: list[str]
    status: str
    attempts: int
    error: str | None
    decision: str | None
    decision_reason: str | None
    primary_metric: str
    primary_delta: float | None
    secondary_deltas: dict[str, Any]
    acceptance_threshold: float
    hypothesis_outcome: str | None
    hypothesis_details: list[dict[str, Any]]
    git_commit: str | None
    seed: int
    dataset_version: str
    protocol_version: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    metrics: dict[str, float] = {}


class MemoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    namespace: str
    category: str
    kind: str
    title: str
    content: str
    source_run_id: str | None
    source_experiment_id: str | None
    finding_id: str | None
    created_by: str
    evidence: dict[str, Any]
    verification_status: str
    tags: list[str]
    created_at: datetime
    updated_at: datetime


class MemorySearchRequest(BaseModel):
    """Either a natural-language ``query`` (rule-based parsing) or structured filters."""

    model_config = ConfigDict(extra="forbid")

    query: str | None = Field(None, max_length=500)
    filters: dict[str, Any] | None = None
    namespace: str = "main"
    limit: int = Field(10, ge=1, le=100)


class AgentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    role: str
    description: str
    tools: list[str]


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2], min_length=1, max_length=10)
    budget: int = Field(6, ge=2, le=20, description="experiments per trial, including the initial baseline")
    initial_model_type: Literal["logistic_regression", "random_forest"] = "random_forest"
    initial_configuration: dict[str, Any] | None = None
    objective: str = "Improve validation F1 score"

    @model_validator(mode="after")
    def _validate_initial(self) -> CompareRequest:
        try:
            self.initial_configuration = validate_configuration(
                self.initial_model_type, self.initial_configuration or default_configuration(self.initial_model_type)
            )
        except ConfigValidationError as exc:
            raise ValueError(f"initial_configuration invalid: {exc}") from None
        return self
