"""API 通用工具:错误响应 / 密码哈希 / 管理员权限校验(V2 FastAPI 版)"""

from fastapi import Depends, HTTPException
from werkzeug.security import check_password_hash, generate_password_hash

from app.api.security import get_current_user


def _hash_password(password: str) -> str:
 """werkzeug pbkdf2 哈希(存量密码格式兼容)。"""
 return generate_password_hash(password)


def _check_password(password_hash: str, password: str) -> bool:
 try:
 return check_password_hash(password_hash, password)
 except Exception:
 return False


def require_admin(user=Depends(get_current_user)):
 """敏感端点守卫:必须登录且 role=admin,否则 403。"""
 if user.role != "admin":
 raise HTTPException(403, "需要管理员权限")
 return user


def validate_password_strength(pwd: str) -> str | None:
 """密码强度校验:至少 8 位且含字母和数字;通过返回 None,否则返回错误文案"""
 if (
 len(pwd) < 8
 or not any(ch.isalpha() for ch in pwd)
 or not any(ch.isdigit() for ch in pwd)
 ):
 return "密码至少 8 位且包含字母和数字"
 return None
