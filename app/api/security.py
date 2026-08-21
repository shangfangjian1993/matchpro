"""认证与安全(V2 FastAPI 原生实现,FastAPI 原生实现)。

协议与前端完全兼容:
- 登录成功 → access_token_cookie(httpOnly, SameSite=Lax)+ csrf_access_token(可读)
- 受保护端点:cookie 验签;非安全方法另需 X-CSRF-TOKEN 头 == csrf cookie(双提交)
- 限流:进程内滑动窗口(登录/注册 10 次/分钟/IP+用户名;默认 200 次/分钟)
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, Request, Response

JWT_SECRET = os.environ.get("JWT_SECRET_KEY", "dev-secret-key-change-me")
JWT_EXPIRES_HOURS = int(os.environ.get("JWT_EXPIRES_HOURS", "24"))
ACCESS_COOKIE = "access_token_cookie"
CSRF_COOKIE = "csrf_access_token"
SAFE_METHODS = ("GET", "HEAD", "OPTIONS")


# ---------------- 令牌 ----------------


def create_token(user_id: int) -> str:
 exp = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRES_HOURS)
 return jwt.encode({"sub": str(user_id), "exp": exp}, JWT_SECRET, algorithm="HS256")


def decode_token(token: str) -> int | None:
 try:
 payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
 return int(payload["sub"])
 except Exception:
 return None


def set_auth_cookies(resp: Response, token: str):
 max_age = JWT_EXPIRES_HOURS * 3600
 resp.set_cookie(
 ACCESS_COOKIE, token, max_age=max_age, httponly=True, samesite="lax", path="/"
 )
 resp.set_cookie(
 CSRF_COOKIE, token, max_age=max_age, httponly=False, samesite="lax", path="/"
 )


def unset_auth_cookies(resp: Response):
 resp.delete_cookie(ACCESS_COOKIE, path="/")
 resp.delete_cookie(CSRF_COOKIE, path="/")


# ---------------- 限流(进程内滑动窗口) ----------------

_LIMITS: dict[str, deque] = defaultdict(deque)
_LIMIT_MAX = 200
_LIMIT_WINDOW = 60.0
_AUTH_MAX = 10
_AUTH_WINDOW = 60.0


def _client_ip(request: Request) -> str:
 if os.environ.get("RATELIMIT_X_FORWARDED_FOR", "true").lower() in (
 "1",
 "true",
 "yes",
 ):
 xff = request.headers.get("X-Forwarded-For", "")
 if xff:
 return xff.split(",")[0].strip() or (
 request.client.host if request.client else "unknown"
 )
 return request.client.host if request.client else "unknown"


def _check_rate(key: str, limit: int, window: float):
 now = time.monotonic()
 dq = _LIMITS[key]
 while dq and now - dq[0] > window:
 dq.popleft()
 if len(dq) >= limit:
 raise HTTPException(429, "请求过于频繁,请稍后再试")
 dq.append(now)


def rate_limit(limit: int = _LIMIT_MAX, window: float = _LIMIT_WINDOW):
 """默认限流依赖:200 次/分钟/IP。"""

 def _dep(request: Request):
 _check_rate(f"rl:{_client_ip(request)}", limit, window)

 return _dep


def auth_rate_limit():
 """登录/注册限流:10 次/分钟/IP(进程内;多 worker 部署建议外部网关限流)。"""

 def _dep(request: Request):
 _check_rate(f"auth:{_client_ip(request)}", _AUTH_MAX, _AUTH_WINDOW)

 return _dep


def check_auth_username_limit(username: str):
 """登录端点内按用户名二次限流(防暴力破解;body 已解析后调用)。"""
 if username:
 _check_rate(f"authu:{username}", _AUTH_MAX, _AUTH_WINDOW)


# ---------------- 依赖:当前用户 ----------------


def get_current_user(request: Request):
 """cookie JWT 验签 → User。CSRF 双提交校验(非安全方法)。"""
 token = request.cookies.get(ACCESS_COOKIE)
 if not token:
 raise HTTPException(401, "未登录")
 user_id = decode_token(token)
 if user_id is None:
 raise HTTPException(401, "登录已过期,请重新登录")
 if request.method not in SAFE_METHODS:
 csrf = request.headers.get("X-CSRF-TOKEN", "")
 if not csrf or csrf != request.cookies.get(CSRF_COOKIE, ""):
 raise HTTPException(401, "Missing CSRF token")
 from app.api.db import User, db

 user = db.session.get(User, user_id)
 if user is None:
 raise HTTPException(401, "用户不存在")
 return user


def require_admin(user=Depends(get_current_user)):
 if user.role != "admin":
 raise HTTPException(403, "需要管理员权限")
 return user
