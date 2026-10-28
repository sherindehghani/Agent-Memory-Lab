"""Shared API helpers: consistent errors, pagination, serialisation."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from sqlalchemy import Select, func, select, true
from sqlalchemy.orm import Session

from app.models import Experiment, ExperimentMetric
from app.orchestration.manager import RunManager
from app.schemas.api import ExperimentOut

MAX_PAGE = 200


def api_error(status: int, code: str, message: str, details: Any = None) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message, "details": details})


def not_found(kind: str, key: str) -> HTTPException:
    return api_error(404, "not_found", f"{kind} {key!r} not found")


def get_manager(request: Request) -> RunManager:
    return request.app.state.manager


def paginate(session: Session, stmt: Select, limit: int, offset: int) -> tuple[list[Any], int]:
    total = session.scalar(select(func.count()).select_from(stmt.order_by(None).subquery())) or 0
    items = session.scalars(stmt.limit(limit).offset(offset)).all()
    return list(items), total


def metrics_by_experiment(session: Session, ids: list[str], split: str = "validation") -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {i: {} for i in ids}
    if ids:
        for row in session.scalars(select(ExperimentMetric).where(ExperimentMetric.experiment_id.in_(ids),
                                                                  ExperimentMetric.split == split)):
            out[row.experiment_id][row.name] = row.value
    return out


def experiments_out(session: Session, exps: list[Experiment]) -> list[ExperimentOut]:
    metrics = metrics_by_experiment(session, [e.id for e in exps])
    return [ExperimentOut.model_validate(e).model_copy(update={"metrics": metrics[e.id]}) for e in exps]


def namespace_filter(column: Any, namespace: str) -> Any:
    """``*`` selects every namespace; anything else is an exact match."""
    return true() if namespace == "*" else column == namespace
