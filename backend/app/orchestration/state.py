"""Run state machine. Every transition is validated and persisted as a RunEvent."""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

from sqlalchemy.orm import Session

from app.core.logging import log_event
from app.models import Run, RunEvent, utcnow

logger = logging.getLogger(__name__)


class RunState(StrEnum):
    CREATED = "CREATED"
    PLANNING = "PLANNING"
    READY = "READY"
    EXECUTING = "EXECUTING"
    EVALUATING = "EVALUATING"
    STORING_MEMORY = "STORING_MEMORY"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


TERMINAL = {RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED, RunState.INTERRUPTED}
ACTIVE = set(RunState) - TERMINAL
_ABORT = {RunState.FAILED, RunState.CANCELLED, RunState.INTERRUPTED}

TRANSITIONS: dict[RunState, set[RunState]] = {
    RunState.CREATED: {RunState.PLANNING} | _ABORT,
    RunState.PLANNING: {RunState.READY, RunState.COMPLETED} | _ABORT,  # COMPLETED: budget/search exhausted
    RunState.READY: {RunState.EXECUTING} | _ABORT,
    RunState.EXECUTING: {RunState.EVALUATING} | _ABORT,
    RunState.EVALUATING: {RunState.STORING_MEMORY} | _ABORT,
    RunState.STORING_MEMORY: {RunState.PLANNING, RunState.COMPLETED} | _ABORT,
    RunState.COMPLETED: set(),
    RunState.FAILED: set(),
    RunState.CANCELLED: set(),
    RunState.INTERRUPTED: set(),
}


class InvalidTransition(RuntimeError):
    pass


def record_event(session: Session, run_id: str, event_type: str, message: str = "", *,
                 experiment_id: str | None = None, agent_id: str | None = None,
                 from_state: str | None = None, to_state: str | None = None, data: dict[str, Any] | None = None,
                 level: int = logging.INFO) -> RunEvent:
    event = RunEvent(run_id=run_id, experiment_id=experiment_id, agent_id=agent_id, event_type=event_type,
                     from_state=from_state, to_state=to_state, message=message, data=data)
    session.add(event)
    session.flush()
    log_event(logger, event_type, message, run_id=run_id, experiment_id=experiment_id, agent_id=agent_id,
              level=level, **({"from": from_state, "to": to_state} if to_state else {}), **(data or {}))
    return event


def transition(session: Session, run: Run, to: RunState, message: str = "", *,
               experiment_id: str | None = None) -> None:
    current = RunState(run.state)
    if to not in TRANSITIONS[current]:
        raise InvalidTransition(f"run {run.id}: {current} -> {to} is not allowed")
    run.state = to.value
    if current == RunState.CREATED and run.started_at is None:
        run.started_at = utcnow()
    if to in TERMINAL:
        run.finished_at = utcnow()
    record_event(session, run.id, "state_transition", message or f"{current} -> {to}", experiment_id=experiment_id,
                 from_state=current.value, to_state=to.value)
