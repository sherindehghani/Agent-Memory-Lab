from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy import text

from app.core.database import init_engine, new_session
from app.memory import graph as kg
from app.memory.retrieval import (
    RANK_WEIGHTS,
    MemoryQuery,
    MemoryRetriever,
    MetricGoal,
    config_similarity,
    parse_natural_query,
)
from app.memory.store import MemoryError_, MemoryStore
from app.models import Experiment, GraphRelationship, MemoryRecord, Run, new_id, utcnow
from app.orchestration.state import InvalidTransition, RunState, transition
from app.schemas.agents import AgentMessageSchema


def make_run(session, namespace: str = "main") -> Run:
    run = Run(id=new_id("run"), namespace=namespace, objective="Improve validation F1 score", state="CREATED",
              seed=0, max_experiments=5, max_duration_seconds=600, max_retries=1,
              initial_model_type="random_forest", initial_configuration={})
    session.add(run)
    session.flush()
    return run


def make_experiment(session, run: Run, *, decision: str = "rejected", model_type: str = "random_forest",
                    config: dict | None = None, f1_delta: float | None = -0.01, mem_delta: float = -0.1,
                    moves: list[str] | None = None) -> Experiment:
    config = config or {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"}
    exp = Experiment(
        id=new_id("exp"), run_id=run.id, namespace=run.namespace, sequence=1, proposed_by="planner_01",
        objective=run.objective, hypothesis="h", model_type=model_type, configuration=config,
        config_hash=new_id("h")[2:], status="completed", decision=decision, primary_delta=f1_delta,
        secondary_deltas={"peak_memory_mb": {"delta": mem_delta, "relative": mem_delta / 0.35,
                                             "candidate": 0.35 + mem_delta, "baseline": 0.35}},
        moves=moves or ["n_estimators:decrease"], acceptance_threshold=0.001, seed=42, dataset_version="d",
        protocol_version="eval-v1")
    session.add(exp)
    session.flush()
    MemoryStore(session, run.namespace).save_metrics(exp.id, {"f1": 0.95 + (f1_delta or 0)})
    return exp


def outcome_memory(session, exp: Experiment) -> MemoryRecord:
    return MemoryStore(session, exp.namespace).add_memory(
        category="episodic", kind="experiment_outcome", title=f"{exp.id} {exp.decision}",
        content=f"Experiment {exp.id} {exp.decision} {exp.model_type}", created_by="evaluator_01",
        verification_status="verified", evidence={"experiment_id": exp.id}, tags=[exp.model_type],
        source_run_id=exp.run_id, source_experiment_id=exp.id)


def test_memory_storage_validates_category_and_status(session) -> None:
    store = MemoryStore(session)
    with pytest.raises(MemoryError_):
        store.add_memory(category="dreams", kind="x", title="t", content="c", created_by="system",
                         verification_status="verified")
    with pytest.raises(MemoryError_):
        store.add_memory(category="semantic", kind="x", title="t", content="c", created_by="system",
                         verification_status="probably")
    m = store.add_memory(category="semantic", kind="note", title="t", content="Reduced memory", created_by="system",
                         verification_status="unverified", tags=["b", "a", "a"])
    assert m.tags == ["a", "b"] and "reduced" in m.search_text


def test_procedural_memory_seeded(session) -> None:
    kinds = {m.kind for m in session.query(MemoryRecord).filter_by(category="procedural")}
    assert {"acceptance_rule", "evaluation_protocol", "execution_constraints", "retry_limits",
            "allowed_changes"} <= kinds


def test_memory_persists_across_restart(settings) -> None:
    with new_session() as s:
        run = make_run(s)
        mem = outcome_memory(s, make_experiment(s, run))
        s.commit()
        mem_id = mem.id
    init_engine(settings.database_url)
    with new_session() as s:
        assert s.get(MemoryRecord, mem_id) is not None


def test_baseline_pointer_rejects_unaccepted(session) -> None:
    run = make_run(session)
    exp = make_experiment(session, run, decision="rejected")
    with pytest.raises(MemoryError_):
        MemoryStore(session).set_baseline(exp, "nope")


def test_retrieval_finds_rejected_memory_reduction_with_evidence(session) -> None:
    run = make_run(session)
    saver = make_experiment(session, run, decision="rejected", mem_delta=-0.12)
    make_experiment(session, run, decision="rejected", mem_delta=+0.05)
    make_experiment(session, run, decision="accepted", f1_delta=0.02, mem_delta=-0.2)
    for e in session.query(Experiment).all():
        outcome_memory(session, e)
    results = MemoryRetriever(session).search(parse_natural_query("Find rejected experiments that reduced memory use"))
    assert [r.memory.source_experiment_id for r in results] == [saver.id]
    r = results[0]
    assert r.evidence["memory_delta_mb"] == -0.12 and r.evidence["f1_delta"] == -0.01
    assert r.evidence["decision"] == "rejected" and "peak_memory_mb decreased" in "; ".join(r.reasons)


def test_ranking_formula_is_transparent(session) -> None:
    run = make_run(session)
    near = make_experiment(session, run, config={"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1,
                                                 "max_features": "sqrt"})
    far = make_experiment(session, run, config={"n_estimators": 10, "max_depth": 2, "min_samples_leaf": 40,
                                                "max_features": "log2"})
    for e in (near, far):
        outcome_memory(session, e)
    q = MemoryQuery(model_type="random_forest", configuration=near.configuration, kinds=["experiment_outcome"])
    results = MemoryRetriever(session).search(q)
    assert results[0].memory.source_experiment_id == near.id
    for r in results:
        assert r.score == pytest.approx(sum(RANK_WEIGHTS[k] * v for k, v in r.components.items()))
        assert all(0.0 <= v <= 1.0 for v in r.components.values())
    assert results[0].components["config_similarity"] == 1.0 > results[1].components["config_similarity"]


def test_recency_breaks_ties(session) -> None:
    run = make_run(session)
    old, new = make_experiment(session, run), make_experiment(session, run)
    m_old, m_new = outcome_memory(session, old), outcome_memory(session, new)
    m_old.created_at = utcnow() - timedelta(days=3)
    session.flush()
    ids = [r.memory.id for r in MemoryRetriever(session).search(MemoryQuery(kinds=["experiment_outcome"]))]
    assert ids == [m_new.id, m_old.id]


def test_config_similarity() -> None:
    a = {"C": 1.0, "max_iter": 1000, "class_weight": None}
    assert config_similarity("logistic_regression", a, "logistic_regression", a) == 1.0
    assert config_similarity("logistic_regression", a, "random_forest", {}) == 0.0
    assert 0 < config_similarity("logistic_regression", a, "logistic_regression", {**a, "C": 10.0}) < 1


def test_no_relevant_memory_returns_empty(session) -> None:
    retriever = MemoryRetriever(session)
    assert retriever.search(MemoryQuery(decision="rejected", model_type="logistic_regression")) == []
    assert retriever.stats.count == 1


def test_memory_referencing_missing_experiment_is_skipped(session) -> None:
    session.execute(text("PRAGMA foreign_keys=OFF"))
    session.add(MemoryRecord(id=new_id("mem"), namespace="main", category="episodic", kind="experiment_outcome",
                             title="orphan", content="orphan", created_by="system", verification_status="verified",
                             evidence={}, tags=[], search_text="orphan", source_experiment_id="exp_000000000000"))
    session.flush()
    session.execute(text("PRAGMA foreign_keys=ON"))
    retriever = MemoryRetriever(session)
    assert retriever.search(MemoryQuery(text="orphan")) == []
    assert retriever.skipped[0]["reason"] == "references a missing experiment"


def test_hyperparameter_filter_and_bad_parameter(session) -> None:
    run = make_run(session)
    lr = make_experiment(session, run, model_type="logistic_regression",
                         config={"C": 1.0, "max_iter": 1000, "class_weight": None})
    outcome_memory(session, lr)
    outcome_memory(session, make_experiment(session, run))
    res = MemoryRetriever(session).search(MemoryQuery(hyperparameter="C"))
    assert [r.memory.source_experiment_id for r in res] == [lr.id]
    with pytest.raises(ValueError):
        MemoryRetriever(session).search(MemoryQuery(hyperparameter="__import__"))


def test_natural_language_parser() -> None:
    q = parse_natural_query("Find rejected experiments that reduced memory use")
    assert q.decision == "rejected" and q.metric_goal == MetricGoal(metric="peak_memory_mb", direction="decrease")
    assert parse_natural_query("experiments with better F1 scores").metric_goal.metric == "f1"
    assert parse_natural_query("experiments using Random Forest").model_type == "random_forest"
    assert parse_natural_query("experiments that used n_estimators").hyperparameter == "n_estimators"
    assert parse_natural_query("earlier failures related to F1").status == "failed"


def test_graph_relationships_require_existing_evidence(session) -> None:
    run = make_run(session)
    exp = make_experiment(session, run)
    X = f"experiment:{exp.id}"
    kg.add_relationship(session, "main", X, "PART_OF", f"run:{run.id}", X)
    with pytest.raises(kg.GraphEvidenceError):
        kg.add_relationship(session, "main", X, "PART_OF", f"run:{run.id}", "experiment:exp_000000000000")
    with pytest.raises(kg.GraphEvidenceError):
        kg.add_relationship(session, "main", X, "USES", "config:doesnotexist", X)
    with pytest.raises(kg.GraphEvidenceError):
        kg.add_relationship(session, "main", X, "LIKES", f"run:{run.id}", X)
    with pytest.raises(kg.GraphEvidenceError):
        kg.add_relationship(session, "main", "garbage", "PART_OF", f"run:{run.id}", X)


def test_graph_construction_evidence_path_and_dangling(session) -> None:
    run = make_run(session)
    parent = make_experiment(session, run, decision="accepted")
    child = make_experiment(session, run)
    child.parent_experiment_id = parent.id
    store = MemoryStore(session)
    finding = store.add_finding(run_id=run.id, experiment_id=child.id, finding_type="secondary_improvement",
                                statement="reduced memory", evidence={}, created_by="evaluator_01")
    X, P = f"experiment:{child.id}", f"experiment:{parent.id}"
    for src, rel, tgt in [(X, "PART_OF", f"run:{run.id}"), (P, "PART_OF", f"run:{run.id}"),
                          (X, "USES", f"config:{child.config_hash}"), (X, "DERIVED_FROM", P),
                          (X, "PRODUCED", f"finding:{finding.id}"), (f"finding:{finding.id}", "SUPPORTED_BY", X)]:
        kg.add_relationship(session, "main", src, rel, tgt, X)
    session.add(GraphRelationship(namespace="main", source_id="experiment:exp_ffffffffffff", relation="PART_OF",
                                  target_id=f"run:{run.id}", evidence_ref="experiment:exp_ffffffffffff"))
    session.flush()
    graph, dangling = kg.build_graph(session, "main")
    assert len(dangling) == 1 and "experiment:exp_ffffffffffff" not in graph
    assert set(kg.experiments_in_run(graph, run.id)) == {X, P}
    assert kg.derived_from(graph, parent.id) == [X]
    assert kg.findings_from_rejected(graph)[0]["finding"] == f"finding:{finding.id}"
    path = kg.evidence_path(graph, finding.id)
    assert path["found"] and [s["relation"] for s in path["paths"][0]] == ["SUPPORTED_BY", "USES"]
    assert path["paths"][0][1]["target"] == f"config:{child.config_hash}"


def test_agent_message_validation() -> None:
    ok = AgentMessageSchema(sender="planner_01", recipient="executor_01", message_type="experiment_proposal",
                            run_id="run_0123456789ab", experiment_id="exp_0123456789ab", payload={})
    assert ok.timestamp.tzinfo is not None
    for bad in ({"sender": "hacker"}, {"message_type": "shell"}, {"run_id": "run_1"}, {"extra": 1}):
        with pytest.raises(ValidationError):
            AgentMessageSchema(**{"sender": "planner_01", "recipient": "executor_01",
                                  "message_type": "experiment_proposal", "run_id": "run_0123456789ab",
                                  "payload": {}, **bad})


def test_state_transitions(session) -> None:
    run = make_run(session)
    for s in (RunState.PLANNING, RunState.READY, RunState.EXECUTING, RunState.EVALUATING, RunState.STORING_MEMORY,
              RunState.PLANNING, RunState.COMPLETED):
        transition(session, run, s)
    assert run.finished_at is not None
    with pytest.raises(InvalidTransition):
        transition(session, run, RunState.PLANNING)
    run2 = make_run(session)
    with pytest.raises(InvalidTransition):
        transition(session, run2, RunState.EXECUTING)
    events = session.execute(text("SELECT count(*) FROM run_events WHERE run_id=:r AND event_type='state_transition'"),
                             {"r": run.id}).scalar()
    assert events == 7
