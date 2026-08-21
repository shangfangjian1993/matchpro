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
