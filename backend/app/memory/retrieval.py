"""Evidence-backed memory retrieval with a transparent ranking formula.

Hard filters are applied in SQL (category, kind, decision, status, model type,
hyperparameter presence, verification) and, if requested, a metric goal
(e.g. "peak memory decreased vs. its baseline"). Remaining candidates are scored:

    score = 0.25 * text_match          # share of query keywords found in the memory text
          + 0.15 * model_match         # 1 if the experiment's model type equals the query's
          + 0.25 * config_similarity   # per-parameter similarity (log-scale for numbers)
          + 0.25 * metric_relevance    # size of the improvement in the requested direction
          + 0.10 * recency             # 1 / (1 + age_hours / 24)

Every component lies in [0, 1]; components that do not apply to a query are 0
for all candidates, so they never change the ordering. Each result carries its
component breakdown, human-readable reasons, and evidence taken from stored
measurements only.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.logging import log_event
from app.experiments.runner import PROTOCOL_VERSION
from app.experiments.search_space import CONFIG_SCHEMAS
from app.memory.store import GLOBAL_NAMESPACE, tokenize
from app.models import Experiment, ExperimentMetric, MemoryRecord, utcnow

logger = logging.getLogger(__name__)

RANK_WEIGHTS = {"text_match": 0.25, "model_match": 0.15, "config_similarity": 0.25,
                "metric_relevance": 0.25, "recency": 0.10}
CANDIDATE_POOL = 500
KNOWN_PARAMS = sorted({p for schema in CONFIG_SCHEMAS.values() for p in schema.model_fields})
METRIC_ALIASES = {"memory": "peak_memory_mb", "ram": "peak_memory_mb", "size": "model_size_kb",
                  "duration": "duration_seconds", "time": "duration_seconds", "faster": "duration_seconds",
                  "f1": "f1", "accuracy": "accuracy", "precision": "precision", "recall": "recall"}
STOPWORDS = {"find", "show", "the", "a", "an", "that", "which", "with", "of", "for", "to", "in", "and", "or",
             "use", "used", "using", "experiments", "experiment", "me", "all", "any", "related", "earlier",
             "previous", "given", "by", "on", "were", "was", "is", "are", "had", "have", "than"}


class MetricGoal(BaseModel):
    metric: Literal["f1", "accuracy", "precision", "recall", "peak_memory_mb", "model_size_kb", "duration_seconds"]
    direction: Literal["increase", "decrease"]


class MemoryQuery(BaseModel):
    text: str = ""
    categories: list[Literal["episodic", "semantic", "procedural"]] = []
    kinds: list[str] = []
    decision: Literal["accepted", "rejected", "invalid"] | None = None
    status: Literal["completed", "failed", "cancelled", "interrupted", "planned", "running"] | None = None
    model_type: Literal["logistic_regression", "random_forest"] | None = None
    configuration: dict[str, Any] | None = None
    hyperparameter: str | None = None
    metric_goal: MetricGoal | None = None
    require_metric_goal: bool = True
    verification_status: list[Literal["verified", "unverified", "contradicted"]] = []
    include_global: bool = True
    limit: int = Field(10, ge=1, le=100)


@dataclass
class RetrievalResult:
    memory: MemoryRecord
    score: float
    components: dict[str, float]
    reasons: list[str]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        m = self.memory
        return {
            "memory_id": m.id,
            "category": m.category,
            "kind": m.kind,
            "title": m.title,
            "content": m.content,
            "verification_status": m.verification_status,
            "source_experiment_id": m.source_experiment_id,
            "tags": m.tags,
            "score": round(self.score, 4),
            "score_components": {k: round(v, 4) for k, v in self.components.items()},
            "reason": "; ".join(self.reasons),
            "reasons": self.reasons,
            "evidence": self.evidence,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }


@dataclass
class RetrievalStats:
    count: int = 0
    total_ms: float = 0.0
    memory_ids: list[str] = field(default_factory=list)


def config_similarity(model_a: str, cfg_a: dict[str, Any], model_b: str, cfg_b: dict[str, Any]) -> float:
    if model_a != model_b:
        return 0.0
    keys = set(cfg_a) | set(cfg_b)
    if not keys:
        return 1.0
    sims = []
    for key in keys:
        a, b = cfg_a.get(key), cfg_b.get(key)
        if a == b:
            sims.append(1.0)
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and a > 0 and b > 0:
            sims.append(max(0.0, 1 - abs(math.log(a) - math.log(b)) / math.log(100)))
        else:
            sims.append(0.0)
    return sum(sims) / len(sims)


def recency(created_at: datetime | None, now: datetime) -> float:
    if created_at is None:
        return 0.0
    age_hours = max(0.0, (now - created_at).total_seconds() / 3600)
    return 1 / (1 + age_hours / 24)


def _metric_delta(exp: Experiment, metric: str) -> tuple[float | None, float | None]:
    """(absolute delta, relative delta) vs. the experiment's comparison baseline."""
    if metric == "f1":
        return exp.primary_delta, exp.primary_delta
    d = (exp.secondary_deltas or {}).get(metric)
    if not isinstance(d, dict):
        return None, None
    return d.get("delta"), d.get("relative")


def parse_natural_query(text: str, limit: int = 10) -> MemoryQuery:
    """Rule-based keyword parser for natural-language memory queries (no LLM)."""
    t = text.lower()
    q: dict[str, Any] = {"text": text, "limit": limit}
    if "rejected" in t:
        q["decision"] = "rejected"
    elif "accepted" in t:
        q["decision"] = "accepted"
    if any(w in t for w in ("failed", "failure", "crash")):
        q["status"] = "failed"
        q.pop("decision", None)
    if "random forest" in t or "random_forest" in t:
        q["model_type"] = "random_forest"
    elif "logistic" in t:
        q["model_type"] = "logistic_regression"
    words = set(tokenize(t))
    for param in KNOWN_PARAMS:
        if param.lower() in words:
            q["hyperparameter"] = param
            break
    decrease = any(w in t for w in ("reduc", "lower", "less", "decreas", "smaller", "faster", "saved", "cheaper"))
    increase = any(w in t for w in ("better", "improv", "higher", "increas", "best"))
    for alias, metric in METRIC_ALIASES.items():
        if alias in t:
            if metric in ("f1", "accuracy", "precision", "recall") and increase:
                q["metric_goal"] = {"metric": metric, "direction": "increase"}
            elif metric not in ("f1", "accuracy", "precision", "recall") and (decrease or alias == "faster"):
                q["metric_goal"] = {"metric": metric, "direction": "decrease"}
            if "metric_goal" in q:
                break
    for cat in ("episodic", "semantic", "procedural"):
        if cat in t:
            q["categories"] = [cat]
    if "similar" in t:
        q["kinds"] = ["experiment_outcome"]
    if any(k in q for k in ("decision", "status", "model_type", "metric_goal", "hyperparameter")) \
            and "categories" not in q:
        q["kinds"] = ["experiment_outcome"]  # experiment questions -> one result per experiment
    return MemoryQuery.model_validate(q)


class MemoryRetriever:
    """Read-only memory interface used by agents and the API."""

    def __init__(self, session: Session, namespace: str = "main"):
        self.session = session
        self.namespace = namespace
        self.stats = RetrievalStats()
        self.skipped: list[dict[str, str]] = []

    def _record(self, t0: float, ids: list[str], label: str) -> None:
        elapsed = (time.perf_counter() - t0) * 1000
        self.stats.count += 1
        self.stats.total_ms += elapsed
        self.stats.memory_ids.extend(i for i in ids if i not in self.stats.memory_ids)
        log_event(logger, "memory_retrieval", f"retrieval '{label}' returned {len(ids)} records",
                  namespace=self.namespace, latency_ms=round(elapsed, 3))

    # -- general search ---------------------------------------------------
    def search(self, query: MemoryQuery, *, now: datetime | None = None) -> list[RetrievalResult]:
        t0 = time.perf_counter()
        now = now or utcnow()
        namespaces = [self.namespace] + ([GLOBAL_NAMESPACE] if query.include_global else [])
        stmt = (select(MemoryRecord, Experiment)
                .outerjoin(Experiment, MemoryRecord.source_experiment_id == Experiment.id)
                .where(MemoryRecord.namespace.in_(namespaces)))
        if query.categories:
            stmt = stmt.where(MemoryRecord.category.in_(query.categories))
        if query.kinds:
            stmt = stmt.where(MemoryRecord.kind.in_(query.kinds))
        if query.verification_status:
            stmt = stmt.where(MemoryRecord.verification_status.in_(query.verification_status))
        if query.decision:
            stmt = stmt.where(Experiment.decision == query.decision)
        if query.status:
            stmt = stmt.where(Experiment.status == query.status)
        if query.model_type:
            stmt = stmt.where(Experiment.model_type == query.model_type)
        if query.hyperparameter:
            if query.hyperparameter not in KNOWN_PARAMS:
                raise ValueError(f"unknown hyperparameter {query.hyperparameter!r}; known: {KNOWN_PARAMS}")
            # json_type is NULL only when the key is absent (an explicit null gives 'null').
            stmt = stmt.where(func.json_type(Experiment.configuration, f"$.{query.hyperparameter}").is_not(None))
        tokens = [w for w in tokenize(query.text) if w not in STOPWORDS]
        # With no structured criteria, a keyword match is required (otherwise text only ranks).
        text_only = bool(tokens) and not any([query.categories, query.kinds, query.decision, query.status,
                                              query.model_type, query.configuration, query.hyperparameter,
                                              query.metric_goal, query.verification_status])
        stmt = stmt.order_by(MemoryRecord.created_at.desc()).limit(CANDIDATE_POOL)

        rows = self.session.execute(stmt).all()
        results: list[RetrievalResult] = []
        self.skipped = []
        exp_ids = [e.id for _, e in rows if e is not None]
        f1_by_exp = self._metric_values(exp_ids, "f1")
        for memory, exp in rows:
            if not isinstance(memory.evidence, dict) or not isinstance(memory.tags, list):
                self.skipped.append({"memory_id": memory.id, "reason": "malformed evidence or tags"})
                continue
            if memory.source_experiment_id and exp is None:
                self.skipped.append({"memory_id": memory.id, "reason": "references a missing experiment"})
                continue
            comps = dict.fromkeys(RANK_WEIGHTS, 0.0)
            reasons: list[str] = []

            if query.metric_goal:
                abs_d, rel_d = _metric_delta(exp, query.metric_goal.metric) if exp else (None, None)
                sign = -1 if query.metric_goal.direction == "decrease" else 1
                improved = abs_d is not None and sign * abs_d > 0
                if query.require_metric_goal and not improved:
                    continue
                if improved:
                    scale = 0.05 if query.metric_goal.metric in ("f1", "accuracy", "precision", "recall") else 0.5
                    magnitude = abs(abs_d if query.metric_goal.metric == "f1" else (rel_d or 0.0))
                    comps["metric_relevance"] = min(1.0, magnitude / scale)
                    reasons.append(f"{query.metric_goal.metric} {query.metric_goal.direction}d by "
                                   f"{abs(abs_d):.4g} vs. its baseline")

            if tokens:
                words = set(memory.search_text.split())
                matched = [w for w in tokens if any(x.startswith(w[:5]) for x in words)]
                comps["text_match"] = len(matched) / len(tokens)
                if matched:
                    reasons.append(f"matches keywords: {', '.join(matched)}")
                elif text_only:
                    continue
            if exp is not None and query.model_type and exp.model_type == query.model_type:
                comps["model_match"] = 1.0
                reasons.append(f"model type is {exp.model_type}")
            if exp is not None and query.configuration is not None:
                ref_model = query.model_type or exp.model_type
                sim = config_similarity(ref_model, query.configuration, exp.model_type, exp.configuration)
                comps["config_similarity"] = sim
                if sim > 0:
                    reasons.append(f"configuration similarity {sim:.2f}")
            comps["recency"] = recency(memory.created_at, now)
            if exp is not None and query.decision:
                reasons.insert(0, f"{exp.decision} experiment")
            if exp is not None and query.hyperparameter:
                reasons.append(f"uses {query.hyperparameter}={exp.configuration.get(query.hyperparameter)!r}")
            if not reasons:
                reasons.append(f"{memory.category} memory matching the filters")

            score = sum(RANK_WEIGHTS[k] * v for k, v in comps.items())
            results.append(RetrievalResult(memory, score, comps, reasons, self._evidence(memory, exp, f1_by_exp)))

        results.sort(key=lambda r: (r.score, r.memory.created_at), reverse=True)
        results = results[: query.limit]
        if self.skipped:
            log_event(logger, "memory_retrieval_skipped", f"skipped {len(self.skipped)} unusable memory records",
                      level=logging.WARNING, skipped=self.skipped)
        self._record(t0, [r.memory.id for r in results], query.text or "structured")
        return results

    def _metric_values(self, experiment_ids: list[str], name: str) -> dict[str, float]:
        if not experiment_ids:
            return {}
        rows = self.session.execute(
            select(ExperimentMetric.experiment_id, ExperimentMetric.value).where(
                ExperimentMetric.experiment_id.in_(experiment_ids), ExperimentMetric.split == "validation",
                ExperimentMetric.name == name)
        ).all()
        return dict(rows)

    @staticmethod
    def _evidence(memory: MemoryRecord, exp: Experiment | None, f1_by_exp: dict[str, float]) -> dict[str, Any]:
        if exp is None:
            return {"type": "record", **{k: v for k, v in memory.evidence.items() if k != "verification_note"}}
        deltas = exp.secondary_deltas or {}

        def d(name: str) -> float | None:
            v = deltas.get(name)
            return v.get("delta") if isinstance(v, dict) else None

        return {
            "type": "experiment_measurement",
            "experiment_id": exp.id,
            "baseline_experiment_id": exp.baseline_experiment_id,
            "decision": exp.decision,
            "status": exp.status,
            "model_type": exp.model_type,
            "configuration": exp.configuration,
            "f1": f1_by_exp.get(exp.id),
            "f1_delta": exp.primary_delta,
            "memory_delta_mb": d("peak_memory_mb"),
            "model_size_delta_kb": d("model_size_kb"),
            "duration_delta_seconds": d("duration_seconds"),
        }

    # -- helpers used by the planner / researcher ----------------------------
    def tried_configurations(self) -> dict[str, list[str]]:
        """config_hash -> memory ids of experiment outcomes already recorded."""
        t0 = time.perf_counter()
        rows = self.session.execute(
            select(Experiment.config_hash, MemoryRecord.id)
            .join(Experiment, MemoryRecord.source_experiment_id == Experiment.id)
            .where(MemoryRecord.namespace == self.namespace, MemoryRecord.kind == "experiment_outcome",
                   Experiment.protocol_version == PROTOCOL_VERSION)
        ).all()
        out: dict[str, list[str]] = {}
        for h, mem_id in rows:
            out.setdefault(h, []).append(mem_id)
        self._record(t0, [], "tried_configurations")
        return out

    def tested_experiment(self, config_hash: str) -> dict[str, Any] | None:
        """Most recent recorded outcome for a configuration (used to extend the search from it)."""
        row = self.session.execute(
            select(MemoryRecord.id, Experiment)
            .join(Experiment, MemoryRecord.source_experiment_id == Experiment.id)
            .where(MemoryRecord.namespace == self.namespace, MemoryRecord.kind == "experiment_outcome",
                   Experiment.protocol_version == PROTOCOL_VERSION,
                   Experiment.config_hash == config_hash, Experiment.status == "completed")
            .order_by(MemoryRecord.created_at.desc()).limit(1)
        ).first()
        if row is None:
            return None
        mem_id, exp = row
        return {"memory_id": mem_id, "experiment_id": exp.id, "model_type": exp.model_type,
                "configuration": exp.configuration, "f1_delta": exp.primary_delta}

    def move_evidence(self) -> dict[str, list[dict[str, Any]]]:
        """move name -> measured F1 effects of past experiments that made that move."""
        t0 = time.perf_counter()
        rows = self.session.execute(
            select(MemoryRecord.id, Experiment)
            .join(Experiment, MemoryRecord.source_experiment_id == Experiment.id)
            .where(MemoryRecord.namespace == self.namespace, MemoryRecord.kind == "experiment_outcome",
                   Experiment.protocol_version == PROTOCOL_VERSION,
                   Experiment.primary_delta.is_not(None))
        ).all()
        out: dict[str, list[dict[str, Any]]] = {}
        for mem_id, exp in rows:
            for move in exp.moves or []:
                out.setdefault(move, []).append({"memory_id": mem_id, "experiment_id": exp.id,
                                                 "f1_delta": exp.primary_delta, "decision": exp.decision})
        self._record(t0, [], "move_evidence")
        return out

    def followed_up_sources(self) -> set[str]:
        """Experiments that already have a research suggestion derived from them."""
        rows = self.session.scalars(
            select(MemoryRecord.source_experiment_id).where(
                MemoryRecord.namespace == self.namespace, MemoryRecord.kind == "research_suggestion")
        )
        return {r for r in rows if r}

    def get(self, memory_id: str) -> MemoryRecord | None:
        return self.session.get(MemoryRecord, memory_id)

