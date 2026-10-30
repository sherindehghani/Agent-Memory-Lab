from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

from app.core.database import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class UTCDateTime(TypeDecorator):

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect):
        if value is not None and value.tzinfo is not None:
            value = value.astimezone(UTC).replace(tzinfo=None)
        return value

    def process_result_value(self, value: datetime | None, dialect):
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value


class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(64))
    description: Mapped[str] = mapped_column(Text)
    tools: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("run"))
    namespace: Mapped[str] = mapped_column(String(128), index=True, default="main")
    objective: Mapped[str] = mapped_column(Text)
    state: Mapped[str] = mapped_column(String(32), index=True, default="CREATED")
    memory_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    planner_mode: Mapped[str] = mapped_column(String(32), default="deterministic")
    seed: Mapped[int] = mapped_column(Integer)
    max_experiments: Mapped[int] = mapped_column(Integer)
    max_duration_seconds: Mapped[int] = mapped_column(Integer)
    max_retries: Mapped[int] = mapped_column(Integer)
    initial_model_type: Mapped[str] = mapped_column(String(64))
    initial_configuration: Mapped[dict[str, Any]] = mapped_column(JSON)
    stop_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    parent_run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"), nullable=True)
    start_baseline_experiment_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    end_baseline_experiment_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    stop_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)


class Experiment(Base):
    __tablename__ = "experiments"
    __table_args__ = (
        Index("ix_experiments_ns_decision", "namespace", "decision"),
        Index("ix_experiments_ns_model", "namespace", "model_type"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("exp"))
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    namespace: Mapped[str] = mapped_column(String(128), index=True, default="main")
    sequence: Mapped[int] = mapped_column(Integer)
    parent_experiment_id: Mapped[str | None] = mapped_column(ForeignKey("experiments.id"), nullable=True, index=True)
    baseline_experiment_id: Mapped[str | None] = mapped_column(ForeignKey("experiments.id"), nullable=True)
    proposed_by: Mapped[str] = mapped_column(ForeignKey("agents.id"))
    executed_by: Mapped[str | None] = mapped_column(ForeignKey("agents.id"), nullable=True)
    objective: Mapped[str] = mapped_column(Text)
    hypothesis: Mapped[str] = mapped_column(Text)
    expected_effects: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    reasoning_summary: Mapped[str] = mapped_column(Text, default="")
    plan_source: Mapped[str] = mapped_column(String(32), default="policy")
    referenced_memory_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    model_type: Mapped[str] = mapped_column(String(64))
    configuration: Mapped[dict[str, Any]] = mapped_column(JSON)
    config_hash: Mapped[str] = mapped_column(String(32), index=True)
    config_diff: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    moves: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), index=True, default="planned")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    decision_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    primary_metric: Mapped[str] = mapped_column(String(32), default="f1")
    primary_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    secondary_deltas: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    acceptance_threshold: Mapped[float] = mapped_column(Float)
    hypothesis_outcome: Mapped[str | None] = mapped_column(String(32), nullable=True)
    hypothesis_details: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    git_commit: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    seed: Mapped[int] = mapped_column(Integer)
    dataset_version: Mapped[str] = mapped_column(String(128))
    protocol_version: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)


class ExperimentMetric(Base):
    __tablename__ = "experiment_metrics"
    __table_args__ = (UniqueConstraint("experiment_id", "split", "name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    split: Mapped[str] = mapped_column(String(16))
    name: Mapped[str] = mapped_column(String(64))
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(16), default="")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class BaselinePointer(Base):

    __tablename__ = "baseline_pointers"

    namespace: Mapped[str] = mapped_column(String(128), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"))
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    reason: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class AgentAction(Base):
    __tablename__ = "agent_actions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("act"))
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"), index=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"), nullable=True, index=True)
    experiment_id: Mapped[str | None] = mapped_column(ForeignKey("experiments.id"), nullable=True)
    action_type: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32))
    input: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    output: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieved_memory_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    retrieval_count: Mapped[int] = mapped_column(Integer, default=0)
    retrieval_ms: Mapped[float] = mapped_column(Float, default=0.0)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)


class AgentMessage(Base):
    __tablename__ = "agent_messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("msg"))
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    experiment_id: Mapped[str | None] = mapped_column(ForeignKey("experiments.id"), nullable=True)
    sender: Mapped[str] = mapped_column(ForeignKey("agents.id"), index=True)
    recipient: Mapped[str] = mapped_column(String(64), index=True)
    message_type: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class Finding(Base):
    __tablename__ = "findings"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("fnd"))
    namespace: Mapped[str] = mapped_column(String(128), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    finding_type: Mapped[str] = mapped_column(String(64), index=True)
    statement: Mapped[str] = mapped_column(Text)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON)
    verification_status: Mapped[str] = mapped_column(String(32))
    created_by: Mapped[str] = mapped_column(ForeignKey("agents.id"))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class MemoryRecord(Base):
    __tablename__ = "memory_records"
    __table_args__ = (Index("ix_memory_ns_category", "namespace", "category"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("mem"))
    namespace: Mapped[str] = mapped_column(String(128), index=True)
    category: Mapped[str] = mapped_column(String(16))
    kind: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(255))
    content: Mapped[str] = mapped_column(Text)
    source_run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"), nullable=True)
    source_experiment_id: Mapped[str | None] = mapped_column(ForeignKey("experiments.id"), nullable=True, index=True)
    finding_id: Mapped[str | None] = mapped_column(ForeignKey("findings.id"), nullable=True)
    created_by: Mapped[str] = mapped_column(String(64))
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    verification_status: Mapped[str] = mapped_column(String(16), index=True)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    search_text: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class GraphRelationship(Base):
    __tablename__ = "graph_relationships"
    __table_args__ = (
        UniqueConstraint("source_id", "relation", "target_id"),
        Index("ix_graph_source", "source_id"),
        Index("ix_graph_target", "target_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    namespace: Mapped[str] = mapped_column(String(128), index=True)
    source_id: Mapped[str] = mapped_column(String(96))
    relation: Mapped[str] = mapped_column(String(32))
    target_id: Mapped[str] = mapped_column(String(96))
    evidence_ref: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class RunEvent(Base):
    __tablename__ = "run_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    experiment_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    agent_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    from_state: Mapped[str | None] = mapped_column(String(32), nullable=True)
    to_state: Mapped[str | None] = mapped_column(String(32), nullable=True)
    message: Mapped[str] = mapped_column(Text, default="")
    data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("eval"))
    status: Mapped[str] = mapped_column(String(32), default="running")
    params: Mapped[dict[str, Any]] = mapped_column(JSON)
    results: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
