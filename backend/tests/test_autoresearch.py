from __future__ import annotations

import subprocess

import pytest

from app.autoresearch import session as ar_session
from app.autoresearch.guard import check_source, count_code_lines
from app.autoresearch.harness import run_harness
from app.autoresearch.proposers import OfflineProposer, ProposalContext, source_hash
from app.autoresearch.templates import BASELINE_TRAIN
from tests.conftest import wait_for_run

LOGREG = '''from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model(seed):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed))
'''


@pytest.mark.parametrize("source,needle", [
    ("import os\ndef build_model(seed): pass", "'os'"),
    ("import subprocess\ndef build_model(seed): pass", "'subprocess'"),
    ("from sklearn.datasets import load_breast_cancer\ndef build_model(seed): pass", "sklearn.datasets"),
    ("from sklearn import datasets\ndef build_model(seed): pass", "sklearn.datasets"),
    ("import sklearn\ndef build_model(seed):\n    return sklearn.datasets", "'datasets'"),
    ("import numpy as np\ndef build_model(seed):\n    return np.load('x')", "'load'"),
    ("def build_model(seed):\n    return open('x')", "'open'"),
    ("def build_model(seed):\n    return ().__class__.__bases__", "'__class__'"),
    ("def build_model(seed):\n    print('RESULT_JSON:{}')", "'print'"),
    ("from . import x\ndef build_model(seed): pass", "relative"),
    ("def build_model(seed)\n    pass", "syntax error"),
    ("x = 1", "build_model"),
])
def test_guard_blocks_unsafe_or_invalid_code(source: str, needle: str) -> None:
    errors = check_source(source)
    assert errors and any(needle in e for e in errors), errors


def test_guard_allows_normal_sklearn_code() -> None:
    assert check_source(BASELINE_TRAIN) == [] and check_source(LOGREG) == []
    assert count_code_lines(LOGREG) == 5


def test_harness_scores_real_code_and_reports_crashes_and_timeouts() -> None:
    ok = run_harness(LOGREG, 42, timeout=120)
    assert ok.status == "completed" and 0.9 < ok.metrics["f1"] <= 1 and ok.metrics["f1_std"] >= 0
    crash = run_harness("def build_model(seed):\n    raise ValueError('boom in train.py')\n", 42, timeout=60)
    assert crash.status == "failed" and "boom in train.py" in crash.log_tail
    slow = run_harness(BASELINE_TRAIN.replace("200", "500"), 42, timeout=1)
    assert slow.status == "timeout" and slow.metrics is None


def test_harness_ignores_fake_result_lines_printed_by_train_py() -> None:
    sneaky = ('import sys\n' 'def build_model(seed):\n    pass\n')
    assert check_source(sneaky)
    out = run_harness(sneaky.replace("import sys\n", ""), 42, timeout=60)
    assert out.status == "failed"


def test_offline_proposer_skips_tried_code_and_exhausts() -> None:
    proposer = OfflineProposer()
    tried: set[str] = set()
    seen = []
    while (p := proposer.propose(ProposalContext("", BASELINE_TRAIN, None, [], [], tried))) is not None:
        assert check_source(p.source) == [], p.idea
        assert source_hash(p.source) not in tried
        tried.add(source_hash(p.source))
        seen.append(p.idea)
    assert seen and len(seen) == len(set(seen))


def _start(client, **body) -> str:
    r = client.post("/api/autoresearch/sessions", json={"tag": "t", **body})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_session_keeps_improvements_resets_failures_and_logs_results(client, settings) -> None:
    run_id = _start(client, max_experiments=4)
    assert client.post("/api/autoresearch/sessions", json={}).status_code == 409
    body = wait_for_run(client, run_id)
    assert body["run"]["state"] == "COMPLETED", body["run"]
    detail = client.get(f"/api/autoresearch/sessions/{run_id}").json()
    rows = detail["results"]
    assert [r["status"] for r in rows][0] == "keep" and rows[0]["description"] == "baseline"
    assert rows[1]["status"] == "keep" and rows[1]["val_f1"] > rows[0]["val_f1"]
    assert all(r["status"] in ("keep", "discard") for r in rows)
    repo = settings.workspace_dir / "autoresearch"
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    best = [r for r in rows if r["status"] == "keep"][-1]
    assert head.startswith(best["commit"]), "branch must point at the last kept commit"
    tsv = (repo / "results.tsv").read_text().splitlines()
    assert tsv[0] == "commit\tval_f1\tmemory_mb\tstatus\tdescription" and len(tsv) == 5
    assert detail["best"]["source"] and "LogisticRegression" in detail["best"]["source"]
    assert detail["best"]["test_metrics"]["f1"] > 0
    discarded = next(r for r in rows if r["status"] == "discard")
    diff = client.get(f"/api/autoresearch/experiments/{discarded['experiment_id']}/diff").json()
    assert diff["diff"].startswith("diff --git") and diff["source"]
    mem = client.post("/api/memory/search", json={"namespace": "autoresearch",
                                                  "filters": {"kinds": ["experiment_outcome"]}}).json()
    assert len(mem["results"]) == 4


def test_second_session_builds_on_best_code_from_memory(client) -> None:
    first = _start(client, max_experiments=2)
    wait_for_run(client, first)
    best_commit = client.get(f"/api/autoresearch/sessions/{first}").json()["best"]["commit"]
    second = _start(client, max_experiments=3)
    body = wait_for_run(client, second)
    cfg = body["run"]["initial_configuration"]
    assert cfg["start_commit"] == best_commit and cfg["branch"] == "autoresearch/t-2"
    rows = client.get(f"/api/autoresearch/sessions/{second}").json()["results"]
    first_rows = client.get(f"/api/autoresearch/sessions/{first}").json()["results"]
    assert {r["description"] for r in rows[1:]}.isdisjoint({r["description"] for r in first_rows[1:]})


def test_program_md_is_editable(client) -> None:
    original = client.get("/api/autoresearch/program").json()["content"]
    assert "train.py" in original
    client.put("/api/autoresearch/program", json={"content": original + "\nPrefer linear models.\n"})
    assert client.get("/api/autoresearch/program").json()["content"].endswith("Prefer linear models.\n")


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete_json(self, system, user, schema=None, max_tokens=4000, effort="low"):
        self.calls.append(user)
        usage = {"input_tokens": 10, "output_tokens": 5, "model": "fake", "estimated_cost_usd": 0.0}
        if user.startswith("The previous train.py failed"):
            return {"description": "logreg (fixed typo)", "hypothesis": "h", "train_py": LOGREG}, usage
        return {"description": "logreg with a typo", "hypothesis": "A linear model should beat the forest.",
                "train_py": LOGREG.replace("max_iter=5000", "max_iterations=5000")}, usage


def test_llm_proposer_with_crash_fix(client, monkeypatch) -> None:
    fake = FakeLLM()
    monkeypatch.setattr(ar_session, "build_llm_client", lambda settings: fake)
    run_id = _start(client, max_experiments=2, proposer="llm", max_fix_attempts=1)
    body = wait_for_run(client, run_id)
    assert body["run"]["state"] == "COMPLETED"
    rows = client.get(f"/api/autoresearch/sessions/{run_id}").json()["results"]
    assert rows[1]["attempts"] == 2 and rows[1]["status"] == "keep"
    assert any("max_iterations" in c for c in fake.calls)
    assert "## program.md" in fake.calls[0] and "## results history" in fake.calls[0]
    assert body["run"]["summary"]["llm_usage"]["calls"] >= 2
