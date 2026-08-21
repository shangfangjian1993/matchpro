import pytest

pytestmark = pytest.mark.db

"""API 最小测试:健康/认证/预测。"""


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_login_and_me(client):
    r = client.post(
        "/api/auth/login", json={"username": "admin", "password": "TestAdmin123"}
    )
    assert r.status_code == 200
    r2 = client.get("/api/auth/me")
    assert r2.status_code == 200
    assert r2.json()["user"]["username"] == "admin"


def test_predict_endpoint(client):
    """预测端点(认证 + CSRF)。"""
    client.post(
        "/api/auth/login", json={"username": "admin", "password": "TestAdmin123"}
    )
    csrf = client.cookies.get("csrf_access_token")
    r = client.post(
        "/api/predictions/match",
        json={
            "league_type": "premier_league",
            "home_team": "Arsenal FC",
            "away_team": "Chelsea FC",
        },
        headers={"X-CSRF-TOKEN": csrf},
    )
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert "top_scores" in body["prediction"]
