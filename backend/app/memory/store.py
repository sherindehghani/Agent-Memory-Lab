from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.logging import log_event
from app.experiments.metrics import METRIC_UNITS
from app.models import (
    BaselinePointer,
    Experiment,
    ExperimentMetric,
    Finding,
    MemoryRecord,
    new_id,
    utcnow,
)

logger = logging.getLogger(__name__)

CATEGORIES = ("episodic", "semantic", "procedural")
VERIFICATION_STATES = ("verified", "unverified", "contradicted")
GLOBAL_NAMESPACE = "global"

_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class MemoryError_(ValueError):
    pass


class MemoryStore:
    def __init__(self, session: Session, namespace: str = "main"):
        self.session = session
        self.namespace = namespace

    def add_memory(
        self,
        *,
        category: str,
        kind: str,
        title: str,
        content: str,
        created_by: str,
        verification_status: str,
        evidence: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        source_run_id: str | None = None,
        source_experiment_id: str | None = None,
        finding_id: str | None = None,
        namespace: str | None = None,
    ) -> MemoryRecord:
        if category not in CATEGORIES:
            raise MemoryError_(f"invalid memory category {category!r}")
        if verification_status not in VERIFICATION_STATES:
            raise MemoryError_(f"invalid verification status {verification_status!r}")
        tags = sorted(set(tags or []))
        record = MemoryRecord(
            id=new_id("mem"),
            namespace=namespace or self.namespace,
            category=category,
            kind=kind,
            title=title[:255],
            content=content,
            created_by=created_by,
            verification_status=verification_status,
            evidence=evidence or {},
            tags=tags,
            search_text=" ".join(tokenize(f"{title} {content} {kind} {' '.join(tags)}")),
            source_run_id=source_run_id,
            source_experiment_id=source_experiment_id,
            finding_id=finding_id,
        )
        self.session.add(record)
        self.session.flush()
        log_event(logger, "memory_created", f"{category}/{kind} memory created", run_id=source_run_id,
                  experiment_id=source_experiment_id, agent_id=created_by, memory_id=record.id)
        return record

    def set_verification(self, memory_id: str, status: str, note: str | None = None) -> MemoryRecord:
        if status not in VERIFICATION_STATES:
            raise MemoryError_(f"invalid verification status {status!r}")
        record = self.session.get(MemoryRecord, memory_id)
        if record is None:
            raise MemoryError_(f"memory {memory_id} not found")
        record.verification_status = status
        if note:
            record.evidence = {**(record.evidence or {}), "verification_note": note}
        record.updated_at = utcnow()
        self.session.flush()
        return record

    def upsert_procedural(self, kind: str, title: str, content: str, evidence: dict[str, Any],
                          tags: list[str]) -> MemoryRecord:
        existing = self.session.scalar(
            select(MemoryRecord).where(MemoryRecord.namespace == GLOBAL_NAMESPACE, MemoryRecord.kind == kind)
        )
        if existing is None:
            return self.add_memory(category="procedural", kind=kind, title=title, content=content,
                                   created_by="system", verification_status="verified", evidence=evidence,
                                   tags=tags, namespace=GLOBAL_NAMESPACE)
        if existing.content != content or existing.evidence != evidence:
            existing.content, existing.evidence, existing.title = content, evidence, title
            existing.search_text = " ".join(tokenize(f"{title} {content} {kind} {' '.join(tags)}"))
            existing.updated_at = utcnow()
        return existing

    def add_finding(self, *, run_id: str, experiment_id: str, finding_type: str, statement: str,
                    evidence: dict[str, Any], created_by: str, verification_status: str = "verified") -> Finding:
        if verification_status not in VERIFICATION_STATES:
            raise MemoryError_(f"invalid verification status {verification_status!r}")
        finding = Finding(
            id=new_id("fnd"), namespace=self.namespace, run_id=run_id, experiment_id=experiment_id,
            finding_type=finding_type, statement=statement, evidence=evidence,
            verification_status=verification_status, created_by=created_by,
        )
        self.session.add(finding)
        self.session.flush()
        return finding

    def save_metrics(self, experiment_id: str, metrics: dict[str, float], split: str = "validation") -> None:
        for name, value in metrics.items():
            self.session.add(ExperimentMetric(experiment_id=experiment_id, split=split, name=name,
                                              value=float(value), unit=METRIC_UNITS.get(name, "")))
        self.session.flush()

    def metrics_of(self, experiment_id: str, split: str = "validation") -> dict[str, float]:
        rows = self.session.scalars(
            select(ExperimentMetric).where(ExperimentMetric.experiment_id == experiment_id,
                                           ExperimentMetric.split == split)
        )
        return {r.name: r.value for r in rows}

    def get_baseline(self) -> Experiment | None:
        pointer = self.session.get(BaselinePointer, self.namespace)
        return self.session.get(Experiment, pointer.experiment_id) if pointer else None

    def set_baseline(self, experiment: Experiment, reason: str) -> None:
        if experiment.status != "completed" or experiment.decision != "accepted":
            raise MemoryError_(f"experiment {experiment.id} is not a completed accepted experiment")
        if not self.metrics_of(experiment.id):
            raise MemoryError_(f"experiment {experiment.id} has no stored metrics")
        pointer = self.session.get(BaselinePointer, self.namespace)
        if pointer is None:
            pointer = BaselinePointer(namespace=self.namespace, experiment_id=experiment.id,
                                      run_id=experiment.run_id, reason=reason)
            self.session.add(pointer)
        else:
            pointer.experiment_id, pointer.run_id, pointer.reason = experiment.id, experiment.run_id, reason
            pointer.updated_at = utcnow()
        self.session.flush()

    def count_memories(self) -> int:
        return self.session.scalar(select(func.count()).select_from(MemoryRecord)
                                   .where(MemoryRecord.namespace == self.namespace)) or 0
