"""V2 API 应用工厂(FastAPI,唯一 API 层)。

设计:统一 /api(无 V1/V2 之分)。
- lifespan:init_db(DATABASE_URL)+ 管理员初始化(ADMIN_PASSWORD)
- 端点:auth / leagues / matches / predictions(+快照复盘)/ models / training /
       user / notifications / experiments / features / health
- 安全:JWT cookie + CSRF(api/security.py)、进程内限流、安全响应头
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.db import init_db, session_scope
from app.api.security import _LIMIT_MAX, _LIMIT_WINDOW, _check_rate, _client_ip

logger = logging.getLogger(__name__)


def _ensure_admin() -> None:
    """首次启动创建管理员(ADMIN_PASSWORD 必须设置)。"""
    admin_password = os.environ.get("ADMIN_PASSWORD", "")
    if not admin_password:
        raise RuntimeError("ADMIN_PASSWORD 环境变量必须设置")
    from app.api.db import User
    from app.api.helpers import _hash_password

    admin = User.query.filter_by(username="admin").first()
    if admin is None:
        db = User.query.session
        db.add(
            User(
                username="admin",
                email="admin@local",
                password_hash=_hash_password(admin_password),
                role="admin",
            )
        )
        db.commit()
        logger.info("管理员 admin 已创建")
    else:
        logger.info("管理员 admin 已存在")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    init_db()
    with session_scope():
        _ensure_admin()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Football Prediction API",
        version="4.0.0",
        description="足球概率预测引擎 API",
        lifespan=_lifespan,
    )

    cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in cors_origins.split(",") if o.strip()],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    # ---------------- V2 路由(新设计唯一入口)----------------
    from app.api.auth import router as auth_router
    from app.api.leagues import router as leagues_router
    from app.api.matches import router as matches_router
    from app.api.meta import router as meta_router
    from app.api.models import router as models_router
    from app.api.predictions import router as pred_router
    from app.api.user import router as user_router

    for r in (
        auth_router,
        leagues_router,
        matches_router,
        pred_router,
        models_router,
        user_router,
        meta_router,
    ):
        app.include_router(r)

    # ---------------- 基础端点 ----------------
    @app.get("/")
    def home():
        return {"service": "football-prediction", "api": "/api", "docs": "/docs"}

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    # ---------------- 中间件:全局限流 + 安全头 ----------------
    @app.middleware("http")
    async def _security_and_limits(request: Request, call_next):
        path = request.url.path
        if path not in ("/health", "/", "/docs", "/openapi.json", "/redoc"):
            try:
                _check_rate(f"rl:{_client_ip(request)}", _LIMIT_MAX, _LIMIT_WINDOW)
            except Exception:
                return JSONResponse(
                    {"error": "请求过于频繁,请稍后再试"}, status_code=429
                )
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        return resp

    return app


# 模块级实例(生产入口:uvicorn api.app:app)
app = create_app()
