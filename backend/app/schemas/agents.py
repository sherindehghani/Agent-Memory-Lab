from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models import utcnow

AGENT_IDS = ("planner_01", "executor_01", "evaluator_01", "researcher_01", "autoresearcher_01")
AgentId = Literal["planner_01", "executor_01", "evaluator_01", "researcher_01", "autoresearcher_01"]
MessageType = Literal[
    "experiment_proposal", "execution_result", "evaluation_decision", "research_suggestion", "memory_stored",
]
MetricName = Literal["f1", "accuracy", "precision", "recall", "peak_memory_mb", "model_size_kb", "duration_seconds"]


class AgentMessageSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sender: AgentId
    recipient: AgentId | Literal["broadcast"]
    message_type: MessageType
    run_id: str = Field(pattern=r"^run_[0-9a-f]{12}$")
    experiment_id: str | None = Field(default=None, pattern=r"^exp_[0-9a-f]{12}$")
    payload: dict[str, Any]
    timestamp: datetime = Field(default_factory=utcnow)


class ExpectedEffect(BaseModel):
    metric: MetricName
    direction: Literal["increase", "decrease", "no_worse"]


class BaselineSnapshot(BaseModel):
    experiment_id: str
    model_type: str
    configuration: dict[str, Any]
    metrics: dict[str, float]
    git_commit: str | None = None


class ResearchSuggestion(BaseModel):
    memory_id: str
    source_experiment_id: str
    source_memory_ids: list[str]
    hypothesis: str
    model_type: str
    configuration: dict[str, Any]
    config_hash: str
    expected_effects: list[ExpectedEffect]
    evidence: dict[str, Any]


class PlannerInput(BaseModel):
    objective: str
    baseline: BaselineSnapshot | None
    initial_model_type: str
    initial_configuration: dict[str, Any]
    memory_enabled: bool
    mode: Literal["deterministic", "llm"] = "deterministic"
    iteration: int
    research_suggestion: ResearchSuggestion | None = None


class PlannerOutput(BaseModel):
    hypothesis: str
    model_type: str
    proposed_configuration: dict[str, Any]
    parent_experiment_id: str | None
    expected_effects: list[ExpectedEffect] = []
    referenced_memory_ids: list[str] = []
    reasoning_summary: str
    plan_source: Literal["initial_baseline", "policy", "research_suggestion", "llm"]
    exhausted: bool = False
    llm_usage: dict[str, Any] | None = None


class ExecutorInput(BaseModel):
    experiment_id: str
    model_type: str
    configuration: dict[str, Any]
    seed: int
    timeout_seconds: float
    max_retries: int
    parent_commit: str | None
    artifact: dict[str, Any]
    commit_message: str


class ExecutorOutput(BaseModel):
    status: Literal["completed", "failed", "timeout", "cancelled"]
    metrics: dict[str, float] | None
    error: str | None
    attempts: int
    git_commit: str | None
    git_error: str | None = None
    wall_seconds: float


class EvaluatorInput(BaseModel):
    experiment_id: str
    status: str
    metrics: dict[str, Any] | None
    error: str | None
    baseline: BaselineSnapshot | None
    expected_effects: list[ExpectedEffect]
    threshold: float


class EvaluatorOutput(BaseModel):
    decision: Literal["accepted", "rejected", "invalid"]
    reason: str
    primary_delta: float | None
    secondary_deltas: dict[str, dict[str, float]]
    secondary_improvements: list[dict[str, Any]]
    threshold: float
    hypothesis_outcome: Literal["supported", "partially_supported", "not_supported", "not_evaluated"]
    hypothesis_details: list[dict[str, Any]]


class ResearcherInput(BaseModel):
    objective: str
    baseline: BaselineSnapshot


class ResearcherOutput(BaseModel):
    suggestion: ResearchSuggestion | None
    examined_memory_ids: list[str]
    summary: str
