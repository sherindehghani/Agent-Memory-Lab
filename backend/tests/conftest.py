"""Test fixtures: every test gets a temporary SQLite DB and Git workspace.

Nothing touches the developer's database or repository.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core import config
from app.core.config import Settings
from app.core.database import init_engine, new_session
from app.orchestration.recovery import bootstrap


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    monkeypatch.setenv("LAB_DATABASE_URL", f"sqlite:///{tmp_path / 'lab.db'}")
    monkeypatch.setenv("LAB_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("LAB_EXECUTION_MODE", "inprocess")
    monkeypatch.setenv("LAB_LOG_JSON", "false")
    monkeypatch.setenv("LAB_LOG_LEVEL", "WARNING")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("LAB_LLM_PROVIDER", "none")
    config.get_settings.cache_clear()
    s = config.get_settings()
    bootstrap(s)
    yield s
    config.get_settings.cache_clear()
    init_engine(s.database_url).dispose()


@pytest.fixture
def session(settings: Settings):  # noqa: ANN201
    s = new_session()
    yield s
    s.close()


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    from app.main import app

    with TestClient(app) as c:
        yield c


def wait_for_run(client: TestClient, run_id: str, timeout: float = 120) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/runs/{run_id}").json()
        if body["run"]["state"] in ("COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"):
            # the worker releases the lock right after committing the terminal state
            for _ in range(50):
                if not client.get("/api/health").json()["busy"]:
                    break
                time.sleep(0.05)
            return body
        time.sleep(0.1)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s")
