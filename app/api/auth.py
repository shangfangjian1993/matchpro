"""V2 认证端点(新设计)"""

from fastapi import APIRouter, Depends, HTTPException, Response

from app.api.db import User, db
from app.api.helpers import _check_password, _hash_password, validate_password_strength
from app.api.schemas import AuthLogin, AuthRegister
from app.api.security import (
 auth_rate_limit,
 check_auth_username_limit,
 create_token,
 get_current_user,
 set_auth_cookies,
 unset_auth_cookies,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", dependencies=[Depends(auth_rate_limit())], status_code=201)
def register(body: AuthRegister):
 username = body.username.strip()
 email = body.email.strip()
 if "@" not in email or "." not in email.split("@")[-1]:
 raise HTTPException(400, "邮箱格式无效")
 err = validate_password_strength(body.password)
 if err:
 raise HTTPException(400, err)
 if User.query.filter_by(username=username).first():
 raise HTTPException(409, "用户名已存在")
 if User.query.filter_by(email=email).first():
 raise HTTPException(409, "邮箱已被注册")
 user = User(
 username=username, email=email, password_hash=_hash_password(body.password)
 )
 db.session.add(user)
 db.session.commit()
 return {"user": user.to_dict()}


@router.post("/login", dependencies=[Depends(auth_rate_limit())])
def login(response: Response, body: AuthLogin):
 username = body.username.strip()
 check_auth_username_limit(username)
 user = User.query.filter_by(username=username).first()
 if user is None or not _check_password(user.password_hash, body.password):
 raise HTTPException(401, "用户名或密码错误")
 token = create_token(user.id)
 set_auth_cookies(response, token)
 return {"user": user.to_dict()}


@router.post("/logout")
def logout(response: Response, user=Depends(get_current_user)):
 unset_auth_cookies(response)
 return {"message": "已退出登录"}


@router.get("/me")
def me(user=Depends(get_current_user)):
 return {"user": user.to_dict()}
