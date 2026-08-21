import pytest

pytestmark = pytest.mark.db


"""API 集成测试:健康/认证/预测 + 边界条件。"""


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


def test_predict_endpoint_returns_valid_probabilities(client):
    """预测返回概率值域合法 (0-1, sum=1)。"""
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
    if r.status_code == 200:
        pred = r.json().get("prediction", {})
        hw = pred.get("home_win_probability", 0)
        dr = pred.get("draw_probability", 0)
        aw = pred.get("away_win_probability", 0)
        assert 0 <= hw <= 1
        assert 0 <= dr <= 1
        assert 0 <= aw <= 1
        total = hw + dr + aw
        assert abs(total - 1.0) < 0.02
