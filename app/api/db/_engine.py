"""数据库引擎管理(engine / session / Base)。"""
from __future__ import annotations
from app.core.timeutil import utcnow

import os
from functools import partial

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    text,
)
from sqlalchemy.orm import (
    declarative_base,
    scoped_session,
    sessionmaker,
)

# SQLAlchemy 2.0 的 relationship/backref 是描述符对象,实例访问会绑定为方法
relationship = partial(
    __import__("sqlalchemy.orm", fromlist=["relationship"]).relationship
)
backref = partial(__import__("sqlalchemy.orm", fromlist=["backref"]).backref)


class _QueryMixin:
    """Model.query(由 scoped_session 提供,线程安全)。"""
    query = None


Base = declarative_base(cls=_QueryMixin)
_engine = None
_session_factory = None
_session = None


def default_database_url() -> str:
    """数据库 URL 解析。"""
    env = os.environ.get("DATABASE_URL")
    if env:
        return env
    try:
        from app.core.config import load_yaml
        _cfg = load_yaml("system.yaml").get("database") or {}
        if _cfg.get("default_url"):
            return _cfg["default_url"]
    except Exception:
        pass
    return "sqlite:///" + os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        "football.db",
    )


def init_db(database_url: str | None = None):
    """初始化 engine + scoped_session(幂等)。"""
    global _engine, _session_factory, _session
    url = database_url or default_database_url()
    if _engine is not None and str(_engine.url) == url:
        return
    kwargs = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 60}
    _engine = create_engine(url, pool_pre_ping=True, **kwargs)
    _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
    _session = scoped_session(_session_factory)
    _QueryMixin.query = _session.query_property()
    Base.metadata.bind = _engine


def get_engine():
    if _engine is None:
        init_db()
    return _engine


class _DBCompat:
    """SQLAlchemy 兼容代理:模型代码保持 db.Column 风格。"""
    Model = Base
    Column = Column
    Integer = Integer
    String = String
    Float = Float
    DateTime = DateTime
    Boolean = Boolean
    Text = Text
    ForeignKey = ForeignKey
    UniqueConstraint = UniqueConstraint
    Index = Index
    func = func
    text = text
    relationship = relationship
    backref = backref

    @property
    def session(self):
        if _session is None:
            init_db()
        return _session

    def create_all(self):
        Base.metadata.create_all(get_engine())


db = _DBCompat()


def session_scope():
    """脚本/管道用上下文:进入时确保 init_db,退出时关闭会话。"""
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        if _session is None:
            init_db()
        try:
            yield _session
        finally:
            _session.remove()

    return _cm()
