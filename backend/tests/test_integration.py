from __future__ import annotations

import time

import pytest
from sqlalchemy import select

from app.agents.base import AgentContext
from app.agents.executor import ExecutorAgent
from app.core.database import new_session
from app.experiments import runner
from app.experiments.versioning import GitWorkspace
from app.memory.retrieval import MemoryRetriever
from app.memory.store import MemoryStore
from app.models import AgentMessage, Experiment, GraphRelationship, MemoryRecord, Run, new_id
from app.orchestration.recovery import bootstrap
from app.schemas.agents import ExecutorInput
from tests.conftest import wait_for_run

LR = {"initial_model_type": "logistic_regression"}


def start(client, **body) -> str:
    r = client.post("/api/runs", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_health_and_openapi(client) -> None:
    assert client.get("/api/health").json()["status"] == "ok"
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/api/runs", "/api/runs/{run_id}/stop", "/api/runs/{run_id}/resume", "/api/memory/search",
              "/api/memory/{memory_id}/evidence", "/api/graph/findings/{finding_id}/evidence",
              "/api/evaluation/compare", "/api/agents/{agent_id}/activity", "/api/experiments/{experiment_id}/history"):
        assert p in paths


def test_create_run_executes_real_experiments_and_persists(client, settings) -> None:
    run_id = start(client, max_experiments=3)
    body = wait_for_run(client, run_id)
    assert body["run"]["state"] == "COMPLETED"
    assert body["run"]["stop_reason"] == "experiment limit reached (3)"
    exps = body["experiments"]
    assert len(exps) == 3
    first = exps[0]
    assert first["plan_source"] == "initial_baseline" and first["decision"] == "accepted"
    for e in exps:
        assert e["status"] == "completed" and 0 < e["metrics"]["f1"] <= 1 and e["metrics"]["peak_memory_mb"] > 0
        assert e["git_commit"] and GitWorkspace(settings.workspace_dir).commit_exists(e["git_commit"])
    states = [ev["to_state"] for ev in body["events"] if ev["event_type"] == "state_transition"]
    assert states[:6] == ["PLANNING", "READY", "EXECUTING", "EVALUATING", "STORING_MEMORY", "PLANNING"]
    assert states[-1] == "COMPLETED"
    metrics = client.get(f"/api/experiments/{first['id']}/metrics").json()
    assert {"validation", "test"} == {m["split"] for m in metrics}
    detail = client.get(f"/api/experiments/{exps[1]['id']}").json()
    assert detail["baseline"]["id"] == first["id"] and detail["git"]["exists"]
    history = client.get(f"/api/experiments/{exps[1]['id']}/history").json()
    assert history["lineage"][1]["id"] == first["id"] and history["git_log"][1]["commit"] == first["git_commit"]
    assert "config.json" in history["git_diff_from_parent"]
    types = {m["message_type"] for m in client.get(f"/api/runs/{run_id}/messages").json()}
    assert {"experiment_proposal", "execution_result", "evaluation_decision", "memory_stored"} <= types


def test_rejected_experiments_retrieved_by_second_agent(client) -> None:
    run_id = start(client, max_experiments=4)
    body = wait_for_run(client, run_id)
    rejected = [e for e in body["experiments"] if e["decision"] == "rejected"]
    assert rejected, "RF neighbours on this split tie or regress, so rejections are expected"
    assert client.get("/api/experiments", params={"decision": "rejected"}).json()["total"] == len(rejected)
    res = client.post("/api/memory/search", json={"filters": {"decision": "rejected",
                                                              "kinds": ["experiment_outcome"]}}).json()
    assert {r["source_experiment_id"] for r in res["results"]} == {e["id"] for e in rejected}
    for r in res["results"]:
        assert r["evidence"]["decision"] == "rejected" and r["reason"]
    nl = client.post("/api/memory/search", json={"query": "Find rejected experiments that reduced memory use"}).json()
    assert all(r["evidence"]["memory_delta_mb"] < 0 for r in nl["results"])
    activity = client.get("/api/agents/planner_01/activity", params={"run_id": run_id}).json()
    assert any(a["retrieval_count"] > 0 for a in activity["actions"]["items"])


def test_memory_evidence_endpoint_and_missing_reference(client) -> None:
    run_id = start(client, max_experiments=2, **LR)
    body = wait_for_run(client, run_id)
    mem = client.get("/api/memory", params={"kind": "experiment_outcome"}).json()["items"][0]
    ev = client.get(f"/api/memory/{mem['id']}/evidence").json()
    assert ev["evidence_status"] == "ok" and ev["experiment"]["metrics"]["f1"] > 0
    with new_session() as s:
        rec = MemoryStore(s).add_memory(category="semantic", kind="note", title="dangling", content="x",
                                        created_by="system", verification_status="unverified",
                                        evidence={"experiment_id": "exp_000000000000"})
        s.commit()
    ev = client.get(f"/api/memory/{rec.id}/evidence").json()
    assert ev["evidence_status"] == "missing_experiment" and ev["problems"]
    assert client.get("/api/memory/mem_doesnotexist").status_code == 404
    assert body["run"]["state"] == "COMPLETED"


def test_graph_relationships_link_real_records(client) -> None:
    run_id = start(client, max_experiments=3)
    body = wait_for_run(client, run_id)
    g = client.get("/api/graph").json()
    assert g["dangling"] == [] and g["stats"]["edges"] > 0
    node_ids = {n["id"] for n in g["nodes"]}
    for e in body["experiments"]:
        assert f"experiment:{e['id']}" in node_ids
    relations = {e["relation"] for e in g["edges"]}
    assert {"PROPOSED", "EXECUTED", "USES", "TRAINS", "PART_OF", "RECORDED_IN", "PRODUCED", "SUPPORTED_BY",
            "DERIVED_FROM", "REFERS_TO", "COMPARED_AGAINST"} <= relations
    with new_session() as s:
        from app.memory.graph import node_exists

        for rel in s.scalars(select(GraphRelationship)):
            assert node_exists(s, rel.evidence_ref), rel.evidence_ref
    finding = next(n for n in g["nodes"] if n["type"] == "finding")
    path = client.get(f"/api/graph/findings/{finding['id'].split(':', 1)[1]}/evidence").json()
    assert path["found"] and path["paths"][0][0]["relation"] == "SUPPORTED_BY"
    assert any(step["relation"] == "USES" for p in path["paths"] for step in p)
    assert client.get(f"/api/graph/runs/{run_id}/experiments").json()["experiments"]


def test_invalid_hyperparameters_rejected(client, settings) -> None:
    r = client.post("/api/runs", json={"initial_configuration": {"n_estimators": 100000}})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    r = client.post("/api/runs", json={"initial_model_type": "svm"})
    assert r.status_code == 422
    with new_session() as s:
        run = Run(id=new_id("run"), namespace="main", objective="o", state="CREATED", seed=0, max_experiments=1,
                  max_duration_seconds=60, max_retries=0, initial_model_type="random_forest",
                  initial_configuration={})
        s.add(run)
        s.flush()
        ctx = AgentContext(session=s, run_id=run.id, namespace="main", settings=settings, store=MemoryStore(s),
                           retriever=MemoryRetriever(s), workspace=GitWorkspace(settings.workspace_dir),
                           rng=__import__("random").Random(0), execution_mode="inprocess")
        res = ExecutorAgent().run(ctx, ExecutorInput(
            experiment_id="exp_0123456789ab", model_type="random_forest", configuration={"max_depth": -3}, seed=42,
            timeout_seconds=10, max_retries=0, parent_commit=None, artifact={}, commit_message="x"))
        assert res.status == "error" and "max_depth" in res.error


def test_experiment_crash_is_recorded_and_baseline_kept(client, monkeypatch) -> None:
    wait_for_run(client, start(client, max_experiments=1, **LR))
    baseline = client.get("/api/overview").json()["current_baseline"]["id"]

    def boom(*_a, **_k):
        raise MemoryError("simulated crash during training")

    monkeypatch.setattr(runner, "build_model", boom)
    body = wait_for_run(client, start(client, max_experiments=2, max_retries=1, **LR))
    assert body["run"]["state"] == "COMPLETED"
    for e in body["experiments"]:
        assert e["status"] == "failed" and e["decision"] == "invalid" and e["attempts"] == 2
        assert "simulated crash" in e["error"]
    assert client.get("/api/overview").json()["current_baseline"]["id"] == baseline
    failures = client.post("/api/memory/search", json={"query": "earlier failures"}).json()["results"]
    assert failures and all(r["evidence"]["status"] == "failed" for r in failures)


def test_invalid_metrics_never_replace_baseline(client, monkeypatch) -> None:
    wait_for_run(client, start(client, max_experiments=1, **LR))
    baseline = client.get("/api/overview").json()["current_baseline"]["id"]
    real = runner.train_and_evaluate
    monkeypatch.setattr(runner, "train_and_evaluate",
                        lambda *a, **k: {**real(*a, **k), "f1": float("nan")})
    body = wait_for_run(client, start(client, max_experiments=1, **LR))
    e = body["experiments"][0]
    assert e["decision"] == "invalid" and "Invalid metrics" in e["decision_reason"]
    assert client.get("/api/overview").json()["current_baseline"]["id"] == baseline


def test_concurrent_run_conflict_and_stop(client, monkeypatch) -> None:
    real = runner.train_and_evaluate

    def slow(*a, **k):
        time.sleep(0.4)
        return real(*a, **k)

    monkeypatch.setattr(runner, "train_and_evaluate", slow)
    run_id = start(client, max_experiments=20, memory_enabled=False, **LR)
    r = client.post("/api/runs", json={})
    assert r.status_code == 409 and r.json()["error"]["code"] == "run_conflict"
    time.sleep(0.6)
    assert client.post(f"/api/runs/{run_id}/stop").status_code == 202
    body = wait_for_run(client, run_id)
    assert body["run"]["state"] == "CANCELLED" and len(body["experiments"]) < 20, body["run"]["stop_reason"]
    assert client.post(f"/api/runs/{run_id}/stop").status_code == 409


def test_restart_after_interrupted_run_resumes_from_baseline(client, settings) -> None:
    body = wait_for_run(client, start(client, max_experiments=2, **LR))
    baseline = client.get("/api/overview").json()["current_baseline"]["id"]
    with new_session() as s:
        run = Run(id=new_id("run"), namespace="main", objective="Improve validation F1", state="EXECUTING", seed=0,
                  max_experiments=3, max_duration_seconds=60, max_retries=0, initial_model_type="logistic_regression",
                  initial_configuration={"C": 1.0, "max_iter": 1000, "class_weight": None})
        s.add(run)
        s.flush()
        s.add(Experiment(id=new_id("exp"), run_id=run.id, namespace="main", sequence=1, proposed_by="planner_01",
                         objective="o", hypothesis="h", model_type="logistic_regression",
                         configuration={"C": 3.16, "max_iter": 1000, "class_weight": None}, config_hash="x",
                         status="running", acceptance_threshold=0.001, seed=42, dataset_version="d",
                         protocol_version="eval-v1"))
        s.commit()
        dead_run = run.id
    assert bootstrap(settings) == [dead_run]
    detail = client.get(f"/api/runs/{dead_run}").json()
    assert detail["run"]["state"] == "INTERRUPTED"
    assert detail["experiments"][0]["status"] == "interrupted"
    assert any(e["event_type"] == "recovery" for e in detail["events"])
    assert client.get("/api/overview").json()["current_baseline"]["id"] == baseline
    r = client.post(f"/api/runs/{dead_run}/resume")
    assert r.status_code == 201
    resumed = wait_for_run(client, r.json()["id"])
    assert resumed["run"]["parent_run_id"] == dead_run
    assert resumed["run"]["start_baseline_experiment_id"] == baseline
    assert resumed["experiments"][0]["plan_source"] != "initial_baseline"
    assert client.post(f"/api/runs/{body['run']['id']}/resume").status_code == 409


def test_end_to_end_memory_reuse_workflow(client) -> None:
    run_id = start(client, max_experiments=6, seed=0)
    body = wait_for_run(client, run_id)
    exps = {e["id"]: e for e in body["experiments"]}
    followups = [e for e in exps.values() if e["plan_source"] == "research_suggestion"]
    secondary = client.get("/api/memory", params={"kind": "secondary_improvement"}).json()["items"]
    if not secondary:
        pytest.skip("no rejected experiment with a secondary improvement occurred in this run")
    assert followups, "a verified secondary improvement must lead to a research follow-up"
    f = followups[0]
    source = exps[f["parent_experiment_id"]]
    assert source["decision"] == "rejected"
    assert any(m["source_experiment_id"] == source["id"] for m in secondary)
    cited = [client.get(f"/api/memory/{m}").json() for m in f["referenced_memory_ids"]]
    assert {c["kind"] for c in cited} == {"research_suggestion", "secondary_improvement"}
    suggestion = next(c for c in cited if c["kind"] == "research_suggestion")
    assert suggestion["created_by"] == "researcher_01" and suggestion["source_experiment_id"] == source["id"]
    assert f["hypothesis_outcome"] in ("supported", "partially_supported", "not_supported")
    expected = "verified" if f["hypothesis_outcome"] == "supported" else "contradicted"
    assert suggestion["verification_status"] == expected
    g = client.get(f"/api/graph/experiments/{f['id']}").json()
    assert {i["memory"] for i in g["influenced_by"]} == {f"memory:{m}" for m in f["referenced_memory_ids"]}
    assert any(e["relation"] == "DERIVED_FROM" and e["target"] == f"experiment:{source['id']}" for e in g["edges"])
    hist = client.get(f"/api/experiments/{f['id']}/history").json()
    assert hist["git_log"][1]["commit"] == source["git_commit"]
    with new_session() as s:
        assert s.scalars(select(AgentMessage).where(AgentMessage.message_type == "research_suggestion")).first()
    summary = body["run"]["summary"]
    assert summary["prior_findings_reused"] >= 2 and summary["research_followups"]


def test_memory_disabled_planner_uses_no_memory(client) -> None:
    body = wait_for_run(client, start(client, max_experiments=4, memory_enabled=False))
    assert all(e["referenced_memory_ids"] == [] for e in body["experiments"])
    assert all(e["plan_source"] in ("initial_baseline", "policy") for e in body["experiments"])
    actions = client.get("/api/agents/researcher_01/activity").json()["actions"]
    assert actions["total"] == 0
    assert body["run"]["summary"]["memory_retrievals"] == 0


def test_evaluation_comparison_is_isolated(client) -> None:
    r = client.post("/api/evaluation/compare", json={"seeds": [0], "budget": 3})
    assert r.status_code == 202
    eval_id = r.json()["id"]
    for _ in range(600):
        row = client.get(f"/api/evaluation/results/{eval_id}").json()
        if row["status"] != "running":
            break
        time.sleep(0.1)
    assert row["status"] == "completed", row
    trials = row["results"]["trials"]
    assert {t["condition"] for t in trials} == {"memory_enabled", "memory_disabled"}
    assert all(t["experiments"] == 3 and t["best_validation_f1"] for t in trials)
    agg = row["results"]["aggregate"]
    assert "difference_memory_minus_no_memory" in agg
    with new_session() as s:
        for t in trials:
            exps = s.scalars(select(Experiment).where(Experiment.run_id == t["run_id"])).all()
            assert {e.namespace for e in exps} == {t["namespace"]}
            for e in exps:
                for mem_id in e.referenced_memory_ids:
                    assert s.get(MemoryRecord, mem_id).namespace == t["namespace"]
    assert client.get("/api/overview").json()["total_experiments"] == 0
    assert client.get("/api/evaluation/results").json()[0]["id"] == eval_id


def test_memory_planner_extends_search_instead_of_repeating(client) -> None:
    body = wait_for_run(client, start(client, max_experiments=7, **LR))
    exps = body["experiments"]
    assert len(exps) == 7 and body["run"]["stop_reason"] == "experiment limit reached (7)"
    with new_session() as s:
        hashes = [s.get(Experiment, e["id"]).config_hash for e in exps]
    assert len(hashes) == len(set(hashes)), "memory-enabled planning never repeats a configuration"
    assert any("extends one more step" in e["reasoning_summary"] or e["plan_source"] == "research_suggestion"
               for e in exps)
    assert body["run"]["summary"]["repeated_configurations"] == 0


def test_llm_mode_without_provider_falls_back_to_deterministic_policy(client) -> None:
    body = wait_for_run(client, start(client, max_experiments=2, planner_mode="llm", **LR))
    second = body["experiments"][1]
    assert second["plan_source"] == "policy"
    assert second["reasoning_summary"].startswith("LLM mode requested but no LLM provider is configured")
    assert body["run"]["summary"]["llm_usage"]["calls"] == 0
