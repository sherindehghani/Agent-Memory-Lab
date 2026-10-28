from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.common import (
    MAX_PAGE,
    experiments_out,
    get_manager,
    metrics_by_experiment,
    namespace_filter,
    not_found,
    paginate,
)
from app.core.database import get_db
from app.memory.store import GLOBAL_NAMESPACE, MemoryStore
from app.models import Experiment, ExperimentMetric, MemoryRecord, Run, RunEvent
from app.orchestration.manager import RunManager, lab_is_busy
from app.schemas.api import ExperimentOut, Page, RunOut

router = APIRouter(prefix="/api", tags=["experiments"])


@router.get("/experiments", response_model=Page[ExperimentOut])
def list_experiments(namespace: str = "main", run_id: str | None = None, status: str | None = None,
                     decision: str | None = None, model_type: str | None = None,
                     order: str = Query("desc", pattern="^(asc|desc)$"),
                     limit: int = Query(50, ge=1, le=MAX_PAGE), offset: int = Query(0, ge=0),
                     db: Session = Depends(get_db)) -> dict[str, Any]:
    stmt = select(Experiment).where(namespace_filter(Experiment.namespace, namespace))
    for col, val in ((Experiment.run_id, run_id), (Experiment.status, status), (Experiment.decision, decision),
                     (Experiment.model_type, model_type)):
        if val:
            stmt = stmt.where(col == val)
    stmt = stmt.order_by(Experiment.created_at.asc() if order == "asc" else Experiment.created_at.desc())
    items, total = paginate(db, stmt, limit, offset)
    return {"items": experiments_out(db, items), "total": total, "limit": limit, "offset": offset}


@router.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str, db: Session = Depends(get_db),
                   manager: RunManager = Depends(get_manager)) -> dict[str, Any]:
    exp = db.get(Experiment, experiment_id)
    if exp is None:
        raise not_found("experiment", experiment_id)
    out = experiments_out(db, [exp])[0].model_dump(mode="json")
    baseline = db.get(Experiment, exp.baseline_experiment_id) if exp.baseline_experiment_id else None
    ws = manager.workspace
    influenced_by = db.scalars(select(MemoryRecord).where(MemoryRecord.id.in_(exp.referenced_memory_ids))).all()
    return {
        "experiment": out,
        "test_metrics": MemoryStore(db).metrics_of(exp.id, "test"),
        "baseline": experiments_out(db, [baseline])[0].model_dump(mode="json") if baseline else None,
        "git": {"commit": exp.git_commit, "exists": ws.commit_exists(exp.git_commit),
                "config": ws.show_config(exp.git_commit) if exp.git_commit else None},
        "influenced_by": [{"memory_id": m.id, "kind": m.kind, "title": m.title, "content": m.content,
                           "verification_status": m.verification_status} for m in influenced_by],
        "missing_memory_references": sorted(set(exp.referenced_memory_ids) - {m.id for m in influenced_by}),
    }


@router.get("/experiments/{experiment_id}/history")
def experiment_history(experiment_id: str, db: Session = Depends(get_db),
                       manager: RunManager = Depends(get_manager)) -> dict[str, Any]:
    exp = db.get(Experiment, experiment_id)
    if exp is None:
        raise not_found("experiment", experiment_id)
    lineage, cur, seen = [], exp, set()
    while cur is not None and cur.id not in seen:
        seen.add(cur.id)
        lineage.append(cur)
        cur = db.get(Experiment, cur.parent_experiment_id) if cur.parent_experiment_id else None
    children = db.scalars(select(Experiment).where(Experiment.parent_experiment_id == exp.id)).all()
    parent = db.get(Experiment, exp.parent_experiment_id) if exp.parent_experiment_id else None
    ws = manager.workspace
    events = db.scalars(select(RunEvent).where(RunEvent.experiment_id == exp.id).order_by(RunEvent.id)).all()
    return {
        "lineage": [e.model_dump(mode="json") for e in experiments_out(db, lineage)],
        "children": [e.model_dump(mode="json") for e in experiments_out(db, list(children))],
        "git_log": ws.history(exp.git_commit) if exp.git_commit else [],
        "git_diff_from_parent": ws.diff(parent.git_commit, exp.git_commit)
        if parent and parent.git_commit and exp.git_commit else "",
        "git_commit_missing": bool(exp.git_commit) and not ws.commit_exists(exp.git_commit),
        "events": [{"event_type": e.event_type, "message": e.message, "to_state": e.to_state,
                    "created_at": e.created_at.isoformat()} for e in events],
    }


@router.get("/experiments/{experiment_id}/metrics")
def experiment_metrics(experiment_id: str, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    if db.get(Experiment, experiment_id) is None:
        raise not_found("experiment", experiment_id)
    rows = db.scalars(select(ExperimentMetric).where(ExperimentMetric.experiment_id == experiment_id)
                      .order_by(ExperimentMetric.split, ExperimentMetric.name)).all()
    return [{"split": r.split, "name": r.name, "value": r.value, "unit": r.unit} for r in rows]


@router.get("/overview", tags=["overview"])
def overview(namespace: str = "main", db: Session = Depends(get_db)) -> dict[str, Any]:
    def count(model: Any, *where: Any) -> int:
        return db.scalar(select(func.count()).select_from(model).where(*where)) or 0

    exps = db.scalars(select(Experiment).where(Experiment.namespace == namespace)
                      .order_by(Experiment.created_at)).all()
    metrics = metrics_by_experiment(db, [e.id for e in exps])
    series = [{"index": i + 1, "experiment_id": e.id, "run_id": e.run_id, "f1": metrics[e.id].get("f1"),
               "peak_memory_mb": metrics[e.id].get("peak_memory_mb"), "decision": e.decision,
               "model_type": e.model_type} for i, e in enumerate(exps)]
    baseline = MemoryStore(db, namespace).get_baseline()
    latest = db.scalars(select(Run).where(Run.namespace == namespace).order_by(Run.created_at.desc()).limit(1)).first()
    f1s = [s["f1"] for s in series if s["f1"] is not None]
    return {
        "namespace": namespace,
        "total_runs": count(Run, Run.namespace == namespace),
        "total_experiments": len(exps),
        "accepted": sum(e.decision == "accepted" for e in exps),
        "rejected": sum(e.decision == "rejected" for e in exps),
        "invalid": sum(e.decision == "invalid" for e in exps),
        "best_validation_f1": max(f1s) if f1s else None,
        "current_baseline": experiments_out(db, [baseline])[0].model_dump(mode="json") if baseline else None,
        "latest_run": RunOut.model_validate(latest).model_dump(mode="json") if latest else None,
        "total_memory_records": count(MemoryRecord, MemoryRecord.namespace.in_([namespace, GLOBAL_NAMESPACE])),
        "busy": lab_is_busy(),
        "f1_series": series,
    }
