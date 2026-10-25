"""Dashboard API: auth, ingest token, rate limiting, exports (SQLite-backed)."""
import importlib.util
import os
from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"
INGEST = {"X-Ingest-Token": os.environ["DASHBOARD_INGEST_TOKEN"]}


@pytest.fixture(scope="module")
def dash(tmp_path_factory):
    db = tmp_path_factory.mktemp("db") / "dashboard.sqlite"
    os.environ["DATABASE_URL"] = f"sqlite:///{db}"
    spec = importlib.util.spec_from_file_location("dashboard_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def client(dash):
    dash.limiter.reset()
    return dash.app.test_client()


def login(client, password=None, **headers):
    return client.post("/api/auth/login", headers=headers, json={
        "username": os.environ["DASHBOARD_USERNAME"],
        "password": password or os.environ["DASHBOARD_PASSWORD"],
    })


@pytest.fixture
def auth(client):
    token = login(client).get_json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ── Startup secret check ─────────────────────────────────────────────────────

@pytest.mark.parametrize("attr,value", [
    ("JWT_SECRET", "dev-secret-change-in-prod"),
    ("JWT_SECRET", "short"),
    ("ADMIN_PASS", "changeme"),
    ("ADMIN_PASS", "elevenchars"),
    ("INGEST_TOKEN", ""),
])
def test_refuses_insecure_secrets(dash, monkeypatch, attr, value):
    monkeypatch.setattr(dash, attr, value)
    with pytest.raises(RuntimeError, match="Insecure dashboard configuration"):
        dash._check_secrets()


def test_accepts_strong_secrets(dash):
    dash._check_secrets()


# ── Ingest from the action server ────────────────────────────────────────────

@pytest.mark.parametrize("headers", [{}, {"X-Ingest-Token": "wrong"}])
def test_ingest_requires_token(client, headers):
    assert client.post("/api/update_session", headers=headers, json={"session_id": "s"}).status_code == 401


def test_ingest_merges_and_computes_status(client, auth):
    assert client.post("/api/update_session", headers=INGEST,
                       json={"session_id": "merge-1", "criticality": "low"}).status_code == 200
    client.post("/api/update_session", headers=INGEST,
                json={"session_id": "merge-1", "emotion": "frustrated"})
    s = client.get("/api/sessions/merge-1", headers=auth).get_json()
    assert s["criticality"] == "low" and s["emotion"] == "frustrated"
    assert s["status"] == "green"


def test_ingest_requires_session_id(client):
    assert client.post("/api/update_session", headers=INGEST, json={}).status_code == 400


@pytest.mark.parametrize("session,status", [
    ({"resolved": True, "criticality": "high"}, "green"),
    ({"handoff_requested": True}, "red"),
    ({"conversation_escalation": "escalate_now"}, "red"),
    ({"criticality": "high"}, "red"),
    ({"criticality": "low"}, "green"),
    ({"criticality": "medium"}, "black"),
])
def test_compute_status(dash, session, status):
    assert dash._compute_status(session) == status


# ── Login and JWT ────────────────────────────────────────────────────────────

def test_login_rejects_wrong_password(client):
    assert login(client, password="nope").status_code == 401


def test_login_rejects_non_string_password(client):
    r = client.post("/api/auth/login", json={"username": "admin", "password": 123})
    assert r.status_code == 401


def test_api_requires_bearer_token(client, auth):
    assert client.get("/api/sessions").status_code == 401
    assert client.get("/api/sessions", headers={"Authorization": "Bearer junk"}).status_code == 401
    assert client.get("/api/sessions", headers=auth).status_code == 200


def test_token_in_query_string_is_rejected(client, auth):
    token = auth["Authorization"].split()[1]
    assert client.get(f"/api/manager/export.csv?token={token}").status_code == 401


def test_csv_export_with_header(client, auth):
    client.post("/api/update_session", headers=INGEST, json={"session_id": "csv-1", "criticality": "high"})
    r = client.get("/api/manager/export.csv", headers=auth)
    assert r.status_code == 200
    assert r.mimetype == "text/csv"
    assert "csv-1" in r.get_data(as_text=True)


# ── Rate limiting ────────────────────────────────────────────────────────────

def test_login_rate_limit_ignores_spoofed_forwarded_for(client):
    codes = [
        login(client, password="nope", **{"X-Forwarded-For": f"10.0.0.{i}"}).status_code
        for i in range(7)
    ]
    assert codes[:5] == [401] * 5
    assert codes[5:] == [429, 429]


# ── Static files ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", ["/app.py", "/../.env", "/observability/logging_config.py"])
def test_only_html_files_are_served(client, path):
    assert client.get(path).status_code == 404
