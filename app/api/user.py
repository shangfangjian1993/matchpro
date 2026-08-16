"""V2 用户端点:设置 / 改密 / API 设置 / 通知(新设计)"""
import json

from fastapi import APIRouter, Depends, HTTPException

from app.api.db import Notification, UserSetting, db
from app.api.helpers import _check_password, _hash_password, validate_password_strength
from app.api.schemas import ApiSettingsUpdate, PasswordChange, SettingsUpdate
from app.api.security import get_current_user

router = APIRouter(prefix="/api/user", tags=["user"])


def _get_or_create_settings(user_id: int) -> UserSetting:
    s = db.session.get(UserSetting, user_id)
    if s is None:
        s = UserSetting(user_id=user_id, settings_json="{}")
        db.session.add(s)
        db.session.commit()
    return s


def _load_settings(s) -> dict:
    try:
        return json.loads(s.settings_json or "{}")
    except json.JSONDecodeError:
        return {}


@router.get("/settings")
def get_settings(user=Depends(get_current_user)):
    s = _get_or_create_settings(user.id)
    base = {"username": "", "email": "", "notifications": True, "darkMode": False}
    base.update(_load_settings(s))
    base["username"] = user.username
    base["email"] = user.email
    return base


@router.put("/settings")
def update_settings(body: SettingsUpdate, user=Depends(get_current_user)):
    s = _get_or_create_settings(user.id)
    saved = _load_settings(s)
    saved.update(body.model_dump(exclude_unset=True, exclude_none=True))
    s.settings_json = json.dumps(saved, ensure_ascii=False)
    db.session.commit()
    return {"message": "设置已保存"}


@router.patch("/password")
def change_password(body: PasswordChange, user=Depends(get_current_user)):
    if not _check_password(user.password_hash, body.current_password):
        raise HTTPException(400, "当前密码错误")
    err = validate_password_strength(body.new_password)
    if err:
        raise HTTPException(400, err)
    user.password_hash = _hash_password(body.new_password)
    db.session.commit()
    return {"message": "密码已修改"}


@router.get("/api-settings")
def get_api_settings(user=Depends(get_current_user)):
    s = _get_or_create_settings(user.id)
    api = _load_settings(s).get("api_settings", {})
    key = str(api.get("apiKey", ""))
    masked = (key[:4] + "****" + key[-4:]) if len(key) > 8 else ("****" if key else "")
    return {"enabled": bool(api.get("enabled", False)),
            "rateLimit": int(api.get("rateLimit", 1000)),
            "apiKey": masked, "hasApiKey": bool(key)}


@router.put("/api-settings")
def update_api_settings(body: ApiSettingsUpdate, user=Depends(get_current_user)):
    s = _get_or_create_settings(user.id)
    saved = _load_settings(s)
    current = saved.get("api_settings", {})
    if body.enabled is not None:
        current["enabled"] = body.enabled
    if body.rateLimit is not None:
        current["rateLimit"] = body.rateLimit
    if body.apiKey is not None:
        current["apiKey"] = body.apiKey.strip()
    saved["api_settings"] = current
    s.settings_json = json.dumps(saved, ensure_ascii=False)
    db.session.commit()
    return {"message": "设置已保存", **current}


# ---------------- 通知 ----------------

@router.get("/notifications")
def list_notifications(user=Depends(get_current_user)):
    items = (Notification.query.filter_by(user_id=user.id)
             .order_by(Notification.created_at.desc()).limit(50).all())
    return {"items": [n.to_dict() for n in items], "total": len(items)}


@router.put("/notifications/{notification_id}/read")
def mark_read(notification_id: int, user=Depends(get_current_user)):
    n = db.session.get(Notification, notification_id)
    if n is None or n.user_id != user.id:
        raise HTTPException(404, "通知不存在")
    n.is_read = True
    db.session.commit()
    return {"message": "已标记为已读"}


@router.put("/notifications/read-all")
def mark_all_read(user=Depends(get_current_user)):
    items = Notification.query.filter_by(user_id=user.id, is_read=False).all()
    for n in items:
        n.is_read = True
    db.session.commit()
    return {"message": f"已标记 {len(items)} 条为已读"}


@router.delete("/notifications/{notification_id}")
def delete_notification(notification_id: int, user=Depends(get_current_user)):
    n = db.session.get(Notification, notification_id)
    if n is None or n.user_id != user.id:
        raise HTTPException(404, "通知不存在")
    db.session.delete(n)
    db.session.commit()
    return {"message": "通知已删除"}
