from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import ValidationError
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.api.common import MAX_PAGE, api_error, experiments_out, not_found, paginate
from app.core.database import get_db
from app.memory.retrieval import MemoryQuery, MemoryRetriever, parse_natural_query
from app.memory.store import GLOBAL_NAMESPACE, MemoryStore
from app.models import Experiment, Finding, MemoryRecord
from app.schemas.api import MemoryOut, MemorySearchRequest, Page

router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.get("", response_model=Page[MemoryOut])
def list_memory(namespace: str = "main", category: str | None = None, kind: str | None = None,
                verification_status: str | None = None, q: str | None = None, include_global: bool = True,
                limit: int = Query(50, ge=1, le=MAX_PAGE), offset: int = Query(0, ge=0),
                db: Session = Depends(get_db)) -> dict[str, Any]:
    namespaces = [namespace] + ([GLOBAL_NAMESPACE] if include_global else [])
    stmt = select(MemoryRecord).where(MemoryRecord.namespace.in_(namespaces))
    for col, val in ((MemoryRecord.category, category), (MemoryRecord.kind, kind),
                     (MemoryRecord.verification_status, verification_status)):
        if val:
            stmt = stmt.where(col == val)
    if q:
        stmt = stmt.where(or_(*[MemoryRecord.search_text.contains(w.lower()) for w in q.split()]))
    items, total = paginate(db, stmt.order_by(MemoryRecord.created_at.desc()), limit, offset)
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.post("/search")
def search_memory(body: MemorySearchRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    if not body.query and not body.filters:
        raise api_error(422, "invalid_query", "Provide 'query' or 'filters'.")
    try:
        if body.filters:
            query = MemoryQuery.model_validate({"text": body.query or "", "limit": body.limit, **body.filters})
        else:
            query = parse_natural_query(body.query or "", limit=body.limit)
    except ValidationError as exc:
        raise api_error(422, "invalid_filters", "Invalid search filters.", exc.errors(include_url=False)) from None
    retriever = MemoryRetriever(db, body.namespace)
    t0 = time.perf_counter()
    try:
        results = retriever.search(query)
    except ValueError as exc:
        raise api_error(422, "invalid_filters", str(exc)) from None
    return {
        "parsed_query": query.model_dump(exclude_none=True),
        "ranking_formula": "0.25*text_match + 0.15*model_match + 0.25*config_similarity + "
                           "0.25*metric_relevance + 0.10*recency",
        "results": [r.to_dict() for r in results],
        "skipped": retriever.skipped,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
    }


@router.get("/{memory_id}", response_model=MemoryOut)
def get_memory(memory_id: str, db: Session = Depends(get_db)) -> MemoryRecord:
    record = db.get(MemoryRecord, memory_id)
    if record is None:
        raise not_found("memory", memory_id)
    return record


@router.get("/{memory_id}/evidence")
def memory_evidence(memory_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    record = db.get(MemoryRecord, memory_id)
    if record is None:
        raise not_found("memory", memory_id)
    out: dict[str, Any] = {"memory": MemoryOut.model_validate(record).model_dump(mode="json"),
                           "evidence_status": "ok", "experiment": None, "baseline": None, "finding": None,
                           "problems": []}
    if not isinstance(record.evidence, dict):
        out["evidence_status"] = "malformed"
        out["problems"].append("evidence is not an object")
        return out
    exp_id = record.source_experiment_id or record.evidence.get("experiment_id")
    if exp_id:
        exp = db.get(Experiment, exp_id)
        if exp is None:
            out["evidence_status"] = "missing_experiment"
            out["problems"].append(f"referenced experiment {exp_id} does not exist")
        else:
            out["experiment"] = experiments_out(db, [exp])[0].model_dump(mode="json")
            if exp.baseline_experiment_id and (base := db.get(Experiment, exp.baseline_experiment_id)):
                out["baseline"] = experiments_out(db, [base])[0].model_dump(mode="json")
            out["test_metrics_note"] = "Held-out test metrics are not part of memory evidence."
            store = MemoryStore(db)
            stored = store.metrics_of(exp.id)
            claimed = record.evidence.get("metrics")
            if isinstance(claimed, dict) and any(abs(stored.get(k, float("nan")) - v) > 1e-9
                                                 for k, v in claimed.items() if isinstance(v, (int, float))):
                out["evidence_status"] = "mismatch"
                out["problems"].append("metrics in the memory differ from the stored measurements")
    elif record.category != "procedural":
        out["evidence_status"] = "no_experiment_reference"
    if record.finding_id and (finding := db.get(Finding, record.finding_id)):
        out["finding"] = {"id": finding.id, "finding_type": finding.finding_type, "statement": finding.statement,
                          "verification_status": finding.verification_status, "evidence": finding.evidence}
    return out
