"""测试环境(最小集):共享 DB(只读优先)+ 应用 fixture。"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.paths import DB_PATH

os.environ.setdefault("DATABASE_URL", f"sqlite:///{DB_PATH}")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-0123456789")
os.environ.setdefault("ADMIN_PASSWORD", "TestAdmin123")


@pytest.fixture(autouse=True)
def _skip_db_tests_if_no_db(request):
    """CI 无 football.db(被 gitignore):依赖 DB 的测试自动跳过。"""
    if "db" in getattr(request.node, "keywords", {}) and not DB_PATH.exists():
        pytest.skip("缺少 data/football.db(DB 依赖测试,CI 跳过)")


@pytest.fixture(scope="session")
def app():
    """FastAPI 应用(TestClient 用)。"""
    from app.api.app import create_app

    return create_app()


@pytest.fixture(scope="session")
def client(app):
    """TestClient(触发 lifespan)。"""
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def db_ctx():
    """DB 会话上下文。"""
    from app.api.db import Base, get_engine, init_db, session_scope

    init_db()
    Base.metadata.create_all(get_engine())
    with session_scope():
        yield


@pytest.fixture(autouse=True)
def _setup_production_artifact(tmp_path, monkeypatch):
    """Create production_artifact.json for all tests that need it."""
    import json
    
    fixture = {
        "league": "premier_league",
        "goal_lambda": {"hgbr": 0.52, "elo": 0.29, "bayes": 0.19},
        "score_distribution": {"poisson": 0.48, "dc": 0.32, "nb": 0.20},
        "outcome": {"shape": 1.0, "gbm": 0.0},
        "tau": -0.071,
        "phi": 2.31,
        "gbm_model_path": "",
        "gbm_model_hash": "",
        "calibration": None,
        "prior": None,
        "lineage": {
            "artifact_version": "ensemble-v3",
            "schema_version": 1,
            "model_version": "test",
            "feature_version": "test",
            "training_cutoff": "2026-08-01",
            "oof_method": "expanding-window",
            "oof_segments": 6,
            "oof_n": 600,
            "shrinkage": 0.15,
            "created_at": "2026-08-21T00:00:00+00:00",
            "training_data_hash": "test",
            "calibration_hash": "",
            "prior_hash": "",
            "gbm_hash": ""
        }
    }
    
    artifact_dir = tmp_path / "ensemble" / "premier_league"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "production_artifact.json"
    artifact_path.write_text(json.dumps(fixture), encoding="utf-8")
    
    import app.core.paths as _paths
    monkeypatch.setattr(_paths, "ARTIFACTS_DIR", tmp_path, raising=False)


@pytest.fixture
def setup_production_artifact(tmp_path):
    """Create a production_artifact.json for testing."""
    import json
    from pathlib import Path
    
    fixture = {
        "league": "premier_league",
        "goal_lambda": {"hgbr": 0.52, "elo": 0.29, "bayes": 0.19},
        "score_distribution": {"poisson": 0.48, "dc": 0.32, "nb": 0.20},
        "outcome": {"shape": 1.0, "gbm": 0.0},
        "tau": -0.071,
        "phi": 2.31,
        "gbm_model_path": "",
        "gbm_model_hash": "",
        "calibration": None,
        "prior": None,
        "lineage": {
            "artifact_version": "ensemble-v3",
            "schema_version": 1,
            "model_version": "test",
            "feature_version": "test",
            "training_cutoff": "2026-08-01",
            "oof_method": "expanding-window",
            "oof_segments": 6,
            "oof_n": 600,
            "shrinkage": 0.15,
            "created_at": "2026-08-21T00:00:00+00:00",
            "training_data_hash": "test",
            "calibration_hash": "",
            "prior_hash": "",
            "gbm_hash": ""
        }
    }
    
    # Create artifact in expected location
    artifact_dir = tmp_path / "ensemble" / "premier_league"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "production_artifact.json"
    artifact_path.write_text(json.dumps(fixture), encoding="utf-8")
    
    return artifact_path
