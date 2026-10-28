from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.common import experiments_out, not_found
from app.core.database import get_db
from app.memory import graph as kg
from app.models import Experiment, Finding, Run

router = APIRouter(prefix="/api/graph", tags=["graph"])

LEGEND = {
    "nodes": list(kg.NODE_TYPES),
    "relations": list(kg.RELATIONS),
}


@router.get("")
def get_graph(namespace: str = "main", run_id: str | None = None, limit: int = Query(150, ge=10, le=1000),
              full: bool = False, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Graph for React Flow. Defaults to the latest run's neighbourhood to limit clutter."""
    graph, dangling = kg.build_graph(db, namespace)
    if not full:
        if run_id is None:  # latest run that actually produced experiments
            run_id = db.scalar(select(Experiment.run_id).where(Experiment.namespace == namespace)
                               .order_by(Experiment.created_at.desc()).limit(1))
        if run_id:
            centre = [f"run:{run_id}"] + kg.experiments_in_run(graph, run_id)
            graph = kg.neighbourhood(graph, centre, radius=1, limit=limit)
    return {**kg.to_flow(graph, dangling), "run_id": run_id, "legend": LEGEND,
            "stats": {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
                      "dangling": len(dangling)}}


@router.get("/experiments/{experiment_id}")
def experiment_graph(experiment_id: str, radius: int = Query(1, ge=1, le=3), db: Session = Depends(get_db)) -> dict:
    exp = db.get(Experiment, experiment_id)
    if exp is None:
        raise not_found("experiment", experiment_id)
    graph, dangling = kg.build_graph(db, exp.namespace)
    sub = kg.neighbourhood(graph, [f"experiment:{experiment_id}"], radius=radius)
    return {**kg.to_flow(sub, dangling), "derived_experiments": kg.derived_from(graph, experiment_id),
            "influenced_by": kg.influences(graph, experiment_id)}


@router.get("/findings/{finding_id}/evidence")
def finding_evidence(finding_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Multi-hop evidence path: Finding -SUPPORTED_BY-> Experiment -USES-> Configuration (+ commit, model)."""
    finding = db.get(Finding, finding_id)
    if finding is None:
        raise not_found("finding", finding_id)
    graph, _ = kg.build_graph(db, finding.namespace)
    path = kg.evidence_path(graph, finding_id)
    exp = db.get(Experiment, finding.experiment_id)
    return {**path, "statement": finding.statement, "finding_type": finding.finding_type,
            "verification_status": finding.verification_status, "stored_evidence": finding.evidence,
            "supporting_experiment": experiments_out(db, [exp])[0].model_dump(mode="json") if exp else None,
            "caveat": "Findings describe measured associations on one fixed validation split, not causation."}


@router.get("/runs/{run_id}/experiments")
def run_experiments(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    run = db.get(Run, run_id)
    if run is None:
        raise not_found("run", run_id)
    graph, _ = kg.build_graph(db, run.namespace)
    return {"run_id": run_id, "experiments": kg.experiments_in_run(graph, run_id)}


@router.get("/experiments/{experiment_id}/derived")
def derived(experiment_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    exp = db.get(Experiment, experiment_id)
    if exp is None:
        raise not_found("experiment", experiment_id)
    graph, _ = kg.build_graph(db, exp.namespace)
    return {"experiment_id": experiment_id, "derived": kg.derived_from(graph, experiment_id)}


@router.get("/queries/rejected-findings")
def rejected_findings(namespace: str = "main", db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    graph, _ = kg.build_graph(db, namespace)
    return kg.findings_from_rejected(graph)
