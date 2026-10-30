from fastapi.testclient import TestClient

from app.core.database import _engine_options
from app.main import app


def test_liveness_and_request_id_header(settings):
    with TestClient(app) as client:
        resp = client.get("/api/health/live", headers={"X-Request-ID": "abc123"})
    assert resp.status_code == 200
    assert resp.headers["X-Request-ID"] == "abc123"
    assert "X-Process-Time-Ms" in resp.headers


def test_readiness_reports_database(settings):
    with TestClient(app) as client:
        resp = client.get("/api/health/ready")
    assert resp.status_code == 200
    assert resp.json()["database"] == "ok"


def test_engine_options_use_pooling_for_server_databases(settings):
    assert "pool_size" not in _engine_options("sqlite:///x.db")
    opts = _engine_options("postgresql+psycopg://u:p@h/db")
    assert opts["pool_pre_ping"] is True and opts["pool_size"] >= 1
