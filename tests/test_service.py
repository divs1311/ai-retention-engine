"""End-to-end: pipeline → CLI → REST API on a small simulated dataset."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from retention_engine import config, pipeline
from retention_engine.api.schemas import UserFeatures


@pytest.fixture(scope="module")
def env(settings, tmp_path_factory):
    """Point the global settings (used by CLI/API) at a fresh, fully-built workspace."""
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("svc")
    mp.setenv("RE_DATA_DIR", str(root / "data"))
    mp.setenv("RE_ARTIFACTS_DIR", str(root / "artifacts"))
    mp.setenv("RE_SIMULATION__N_USERS", str(settings.simulation.n_users))
    mp.setenv("RE_SIMULATION__SEED", str(settings.simulation.seed))
    mp.setenv("RE_SIMULATION__END_DATE", str(settings.simulation.end_date))
    mp.delenv("ANTHROPIC_API_KEY", raising=False)
    mp.delenv("OPENAI_API_KEY", raising=False)
    mp.delenv("RE_API_KEY", raising=False)
    config.get_settings.cache_clear()
    s = config.get_settings()
    campaigns = pipeline.run_all(s, provider="template")
    yield s, campaigns
    mp.undo()
    config.get_settings.cache_clear()


@pytest.mark.slow
def test_pipeline_outputs(env):
    s, campaigns = env
    for p in (
        s.users_path,
        s.orders_path,
        s.model_path,
        s.model_card_path,
        s.scores_path,
        s.campaigns_path,
    ):
        assert p.exists(), p
    saved = json.loads(s.campaigns_path.read_text())
    assert len(saved) == len(campaigns) > 0
    assert all(c["guardrails"]["passed"] for c in saved)


@pytest.mark.slow
def test_ensure_artifacts_is_idempotent(env):
    s, _ = env
    mtime = s.model_path.stat().st_mtime
    art = pipeline.ensure_artifacts(s)
    assert s.model_path.stat().st_mtime == mtime  # nothing retrained
    assert len(art.scores) > 0


@pytest.mark.slow
def test_cli(env):
    from retention_engine.cli import app

    runner = CliRunner()
    assert runner.invoke(app, ["--version"]).exit_code == 0
    r = runner.invoke(app, ["kpis"])
    assert r.exit_code == 0 and "m1 retention" in r.output
    r = runner.invoke(app, ["campaigns", "--provider", "template", "--limit", "2"])
    assert r.exit_code == 0, r.output


@pytest.fixture(scope="module")
def client(env):
    from retention_engine.api.main import app

    with TestClient(app) as c:
        yield c


@pytest.mark.slow
def test_api_health_and_model(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["scored_users"] > 0
    card = client.get("/v1/model").json()
    assert "metrics" in card and "curves" not in card


@pytest.mark.slow
def test_api_risk_queue(client):
    rows = client.get("/v1/users/at-risk", params={"tier": "high", "limit": 5}).json()
    assert 0 < len(rows) <= 5
    assert all(r["risk_tier"] == "high" for r in rows)
    one = client.get(f"/v1/users/{rows[0]['user_id']}/risk")
    assert one.status_code == 200
    assert client.get("/v1/users/999999999/risk").status_code == 404
    assert client.get("/v1/users/at-risk", params={"tier": "bogus"}).status_code == 422


@pytest.mark.slow
def test_api_adhoc_scoring(client):
    example = UserFeatures.model_config["json_schema_extra"]["examples"][0]
    r = client.post(
        "/v1/score", json={"users": [example, {**example, "recency_days": 1, "overdue_ratio": 0.1}]}
    )
    assert r.status_code == 200
    res = r.json()["results"]
    assert res[0]["churn_probability"] > res[1]["churn_probability"]
    assert (
        client.post("/v1/score", json={"users": [{**example, "late_rate_90d": 3}]}).status_code
        == 422
    )


@pytest.mark.slow
def test_api_campaigns(client):
    audiences = client.get("/v1/audiences").json()
    assert audiences
    r = client.post(
        "/v1/campaigns/generate",
        json={"audience_id": audiences[0]["audience_id"], "provider": "template"},
    )
    assert r.status_code == 200 and r.json()["guardrails"]["passed"]
    assert client.post("/v1/campaigns/generate", json={"audience_id": "nope"}).status_code == 404


@pytest.mark.slow
def test_api_key_auth(client, monkeypatch):
    monkeypatch.setenv("RE_API_KEY", "secret")
    assert client.get("/v1/kpis").status_code == 401
    assert client.get("/v1/kpis", headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/health").status_code == 200  # health stays public
