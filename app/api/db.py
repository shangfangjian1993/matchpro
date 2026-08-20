"""数据库模型定义(表结构对齐)。

V2 重构:使用原生 SQLAlchemy 2.x。
db 为兼容代理:模型定义保持 db.Column/db.String 等原写法,
db.session 为 scoped_session(线程本地),脚本用 session_scope() 管理生命周期。
"""

import json
import uuid
from functools import partial

# ---- 原生 SQLAlchemy 2.x ----
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

from app.core.timeutil import utcnow

# 注意:SQLAlchemy 2.0 的 relationship/backref 是描述符对象,实例访问会绑定为方法
# 且第一位置参数变成 secondary —— 必须用 partial 包装,保证 db.relationship("X") 语义
relationship = partial(
    __import__("sqlalchemy.orm", fromlist=["relationship"]).relationship
)
backref = partial(__import__("sqlalchemy.orm", fromlist=["backref"]).backref)


class _QueryMixin:
    """Model.query(由 scoped_session 提供,线程安全)。"""

    query = None  # init_db 后由 query_property 绑定


Base = declarative_base(cls=_QueryMixin)
_engine = None
_session_factory = None
_session = None


def default_database_url() -> str:
    """数据库 URL 解析(PostgreSQL+Redis 改造,配置单一源)。

    优先级:env DATABASE_URL > configs/system.yaml database.default_url
    (生产配置 postgresql://user:pass@host:5432/football) > 默认 sqlite。
    """
    import os

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
    """初始化 engine + scoped_session(幂等)。脚本/管道/API 统一入口。"""
    global _engine, _session_factory, _session
    url = database_url or default_database_url()
    if _engine is not None and str(_engine.url) == url:
        return
    kwargs = {}
    if url.startswith("sqlite"):
        # 12× SQLite 默认超时(5s):避免并发写(采集进程+预测)时 OperationalError
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 60}
    _engine = create_engine(url, pool_pre_ping=True, **kwargs)
    _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
    _session = scoped_session(_session_factory)
    # Model.query(scoped_session)
    _QueryMixin.query = _session.query_property()
    # 模型注册到 metadata(惰性建表能力保留)
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
    """脚本/管道用上下文:进入时确保 init_db,退出时关闭会话。

    用法:
        with session_scope():
            League.query.filter_by(...).first()
    """
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


from app.data.canonical.team_names_zh import to_zh


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default="user")
    created_at = db.Column(db.DateTime, default=utcnow)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class League(db.Model):
    __tablename__ = "leagues"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    country = db.Column(db.String(50))
    season = db.Column(db.String(20))
    league_type = db.Column(db.String(50), nullable=False, default="PREMIER_LEAGUE")
    # 统一赛事表类型:league(联赛)/ cup(杯赛)/ national(国家队赛事)
    comp_type = db.Column(db.String(20), default="league")
    created_at = db.Column(db.DateTime, default=utcnow)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)

    __table_args__ = (
        db.UniqueConstraint(
            "name", "country", "season", name="uq_league_name_country_season"
        ),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "country": self.country,
            "season": self.season,
            "league_type": self.league_type,
        }

    matches = db.relationship("Match", back_populates="league", lazy="dynamic")


class Match(db.Model):
    __tablename__ = "matches"

    id = db.Column(db.Integer, primary_key=True)
    league_id = db.Column(db.Integer, db.ForeignKey("leagues.id"), nullable=False)
    season_id = db.Column(db.Integer, nullable=True)  # §2.1 精简事件表:赛季维度
    league = db.relationship("League", back_populates="matches")
    home_team = db.Column(db.String(100), nullable=False)
    away_team = db.Column(db.String(100), nullable=False)
    home_team_id = db.Column(db.Integer, db.ForeignKey("teams.id"), nullable=True)
    away_team_id = db.Column(db.Integer, db.ForeignKey("teams.id"), nullable=True)
    home_goals = db.Column(db.Integer, default=0)
    away_goals = db.Column(db.Integer, default=0)
    match_date = db.Column(db.DateTime, nullable=False, default=utcnow)
    match_status = db.Column(db.String(20), default="scheduled")
    # ---- 指标扩展列(2026-08 规划,摄入层/特征层后期引用;全 NULL 时模型自动跳过) ----
    # 通用统计(所有联赛可用)
    home_xg = db.Column(db.Float)
    away_xg = db.Column(db.Float)
    home_shots = db.Column(db.Integer)
    away_shots = db.Column(db.Integer)
    home_shots_on_target = db.Column(db.Integer)
    away_shots_on_target = db.Column(db.Integer)
    home_corners = db.Column(db.Integer)
    away_corners = db.Column(db.Integer)
    home_possession = db.Column(db.Float)  # 主队控球率 0-100(客队 = 100 - 主队)
    home_yellow_cards = db.Column(db.Integer)
    away_yellow_cards = db.Column(db.Integer)
    home_red_cards = db.Column(db.Integer)
    away_red_cards = db.Column(db.Integer)
    home_ht_goals = db.Column(db.Integer)  # 半场比分(半全场玩法数据源)
    away_ht_goals = db.Column(db.Integer)
    # 统一指标列(所有联赛同一 schema;各联赛模型按需引用,未引用联赛保持 NULL)
    home_passing_accuracy = db.Column(db.Float)
    away_passing_accuracy = db.Column(db.Float)
    home_xg_chain = db.Column(db.Float)
    away_xg_chain = db.Column(db.Float)
    home_efficiency = db.Column(db.Float)
    away_efficiency = db.Column(db.Float)
    home_transition_speed = db.Column(db.Float)
    away_transition_speed = db.Column(db.Float)
    home_defensive_actions = db.Column(db.Float)
    away_defensive_actions = db.Column(db.Float)
    home_counter_attacks = db.Column(db.Float)
    away_counter_attacks = db.Column(db.Float)
    home_tactical_rating = db.Column(db.Float)
    away_tactical_rating = db.Column(db.Float)
    home_experience = db.Column(db.Float)
    away_experience = db.Column(db.Float)
    # 赛事阶段(联赛为 NULL;赛事模型训练时映射为 stage)
    match_stage = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=utcnow)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)

    # 指标字段(摄入层写入,列表/详情返回,便于核对)
    METRIC_ATTRS = (
        "home_xg",
        "away_xg",
        "home_shots",
        "away_shots",
        "home_shots_on_target",
        "away_shots_on_target",
        "home_corners",
        "away_corners",
        "home_possession",
        "home_yellow_cards",
        "away_yellow_cards",
        "home_red_cards",
        "away_red_cards",
        "home_ht_goals",
        "away_ht_goals",
        "home_passing_accuracy",
        "away_passing_accuracy",
        "home_xg_chain",
        "away_xg_chain",
        "home_efficiency",
        "away_efficiency",
        "home_transition_speed",
        "away_transition_speed",
        "home_defensive_actions",
        "away_defensive_actions",
        "home_counter_attacks",
        "away_counter_attacks",
        "home_tactical_rating",
        "away_tactical_rating",
        "home_experience",
        "away_experience",
        "match_stage",
    )

    def to_dict(self):
        # 中文队名(前端显示用;无映射回退英文)
        home_team_zh = to_zh(self.home_team)
        away_team_zh = to_zh(self.away_team)

        d = {
            "id": self.id,
            "league_id": self.league_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_team_zh": home_team_zh,
            "away_team_zh": away_team_zh,
            "home_goals": self.home_goals,
            "away_goals": self.away_goals,
            "match_date": self.match_date.isoformat() if self.match_date else None,
            "match_status": self.match_status,
        }
        for attr in self.METRIC_ATTRS:
            d[attr] = getattr(self, attr, None)
        return d


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(
        db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    match_id = db.Column(db.Integer, db.ForeignKey("matches.id"), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    league_type = db.Column(db.String(50), nullable=False, default="PREMIER_LEAGUE")
    home_team = db.Column(db.String(100), nullable=False)
    away_team = db.Column(db.String(100), nullable=False)
    home_goals_prediction = db.Column(db.Float, nullable=False)
    away_goals_prediction = db.Column(db.Float, nullable=False)
    home_win_probability = db.Column(db.Float, default=0.0)
    draw_probability = db.Column(db.Float, default=0.0)
    away_win_probability = db.Column(db.Float, default=0.0)
    confidence = db.Column(db.Float, default=0.0)
    prediction_timestamp = db.Column(db.DateTime, default=utcnow)
    created_at = db.Column(db.DateTime, default=utcnow)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)

    def to_dict(self, match_map: dict | None = None):
        # 关联实际比赛结果(若已完赛);match_map 由批量调用方预取,避免 N+1 查询
        actual_home = actual_away = None
        is_correct = None
        if self.match_id:
            m = (
                match_map.get(self.match_id)
                if match_map is not None
                else db.session.get(Match, self.match_id)
            )
            if m and m.match_status == "finished":
                actual_home, actual_away = m.home_goals, m.away_goals
                is_correct = (
                    round(self.home_goals_prediction) == actual_home
                    and round(self.away_goals_prediction) == actual_away
                )
        return {
            "id": self.id,
            "public_id": self.public_id,
            "match_id": self.match_id,
            "league_type": self.league_type,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_team_zh": to_zh(self.home_team),
            "away_team_zh": to_zh(self.away_team),
            "raw_lambda_home": self.home_goals_prediction,
            "raw_lambda_away": self.away_goals_prediction,
            "home_win_probability": self.home_win_probability,
            "draw_probability": self.draw_probability,
            "away_win_probability": self.away_win_probability,
            "confidence": self.confidence,
            "actual_home_goals": actual_home,
            "actual_away_goals": actual_away,
            "is_correct": is_correct,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ModelRecord(db.Model):
    __tablename__ = "models"

    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(
        db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    name = db.Column(db.String(100), nullable=False)
    league_type = db.Column(db.String(50), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)
    version = db.Column(db.String(20), default="1.0.0")
    accuracy = db.Column(db.Float, default=0.0)
    mse = db.Column(db.Float, default=0.0)
    mae = db.Column(db.Float, default=0.0)
    rmse = db.Column(db.Float, default=0.0)
    poisson_loss = db.Column(db.Float, default=0.0)
    exact_accuracy = db.Column(db.Float, default=0.0)
    feature_count = db.Column(db.Integer, default=0)
    model_path = db.Column(db.String(255))
    training_date = db.Column(db.DateTime, default=utcnow)
    created_at = db.Column(db.DateTime, default=utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "public_id": self.public_id,
            "name": self.name,
            "league_type": self.league_type,
            "model_type": self.model_type,
            "model_version": self.version,
            "accuracy": self.accuracy,
            "mse": self.mse,
            "mae": self.mae,
            "rmse": self.rmse,
            "poisson_loss": self.poisson_loss,
            "exact_accuracy": self.exact_accuracy,
            "feature_count": self.feature_count,
            "model_path": self.model_path,
            "training_date": self.training_date.isoformat()
            if self.training_date
            else None,
        }


class UserSetting(db.Model):
    __tablename__ = "user_settings"

    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), primary_key=True)
    settings_json = db.Column(db.Text, default="{}")
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)


class Notification(db.Model):
    __tablename__ = "notifications"

    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(
        db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, default="")
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "public_id": self.public_id,
            "title": self.title,
            "content": self.content,
            "is_read": self.is_read,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class TrainingTask(db.Model):
    """异步训练任务(状态机:pending -> running -> succeeded/failed)"""

    __tablename__ = "training_tasks"

    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(
        db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    league_type = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(20), default="pending")
    message = db.Column(db.Text, default="")
    metrics_json = db.Column(db.Text, default="{}")
    created_at = db.Column(db.DateTime, default=utcnow)
    started_at = db.Column(db.DateTime)
    finished_at = db.Column(db.DateTime)

    def to_dict(self):
        try:
            metrics = json.loads(self.metrics_json or "{}")
        except ValueError:
            metrics = {}
        return {
            "id": self.id,
            "public_id": self.public_id,
            "league_type": self.league_type,
            "status": self.status,
            "message": self.message,
            "metrics": metrics,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


class TeamName(db.Model):
    """队名中英文映射(en_name 主键,zh_name 显示名)"""

    __tablename__ = "team_names"

    en_name = db.Column(db.String(120), primary_key=True)
    zh_name = db.Column(db.String(60), nullable=False)
    source = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, default=utcnow)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)

    def to_dict(self):
        return {"en_name": self.en_name, "zh_name": self.zh_name, "source": self.source}


class Team(db.Model):
    """球队实体表:俱乐部/国家队统一;ELO 评分落点 elo_rating。"""

    __tablename__ = "teams"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False, unique=True)
    name_zh = db.Column(db.String(60), nullable=True)
    team_type = db.Column(db.String(20), nullable=False, default="club")
    country = db.Column(db.String(50), nullable=True)
    founded_year = db.Column(db.Integer, nullable=True)
    stadium = db.Column(db.String(100), nullable=True)
    city = db.Column(db.String(50), nullable=True)
    elo_rating = db.Column(db.Float, nullable=True)
    attack_elo = db.Column(db.Float, nullable=True)
    defense_elo = db.Column(db.Float, nullable=True)
    elo_updated_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=utcnow)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "name_zh": self.name_zh,
            "team_type": self.team_type,
            "country": self.country,
            "elo_rating": self.elo_rating,
        }


class TeamSeason(db.Model):
    """球队 × 赛季:联赛归属/升降级。"""

    __tablename__ = "team_seasons"

    id = db.Column(db.Integer, primary_key=True)
    team_id = db.Column(db.Integer, db.ForeignKey("teams.id"), nullable=False)
    league_id = db.Column(db.Integer, db.ForeignKey("leagues.id"), nullable=False)
    season = db.Column(db.String(20), nullable=False)
    tier = db.Column(db.Integer, nullable=True)
    promoted = db.Column(db.Boolean, nullable=True)
    relegated = db.Column(db.Boolean, nullable=True)
    created_at = db.Column(db.DateTime, default=utcnow)
    __table_args__ = (
        db.UniqueConstraint("team_id", "league_id", "season", name="uq_team_season"),
    )


class PredictionSnapshot(db.Model):
    """预测快照:任何预测可 100% 重放(数据/特征/模型三哈希 + 全部输入)。"""

    __tablename__ = "prediction_snapshots"

    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(
        db.String(36),
        unique=True,
        nullable=False,
        default=lambda: str(__import__("uuid").uuid4()),
    )
    created_at = db.Column(db.DateTime, default=utcnow)
    league_id = db.Column(db.Integer, db.ForeignKey("leagues.id"), nullable=False)
    home_team_id = db.Column(db.Integer, db.ForeignKey("teams.id"), nullable=True)
    away_team_id = db.Column(db.Integer, db.ForeignKey("teams.id"), nullable=True)
    home_team = db.Column(db.String(120), nullable=False)
    away_team = db.Column(db.String(120), nullable=False)
    kickoff = db.Column(db.DateTime, nullable=True)
    model_version = db.Column(db.String(20), nullable=False)
    feature_version = db.Column(db.String(40), nullable=False)
    data_hash = db.Column(db.String(64), nullable=False)
    model_hash = db.Column(db.String(64), nullable=False)
    sha256 = db.Column(db.String(64), nullable=False)
    snapshot_json = db.Column(db.Text, nullable=False)
    probabilities_json = db.Column(db.Text, nullable=False)
    actual_home_goals = db.Column(db.Integer, nullable=True)
    actual_away_goals = db.Column(db.Integer, nullable=True)
    is_correct = db.Column(db.Boolean, nullable=True)
    log_loss = db.Column(db.Float, nullable=True)
    __table_args__ = (
        db.UniqueConstraint(
            "league_id", "home_team", "away_team", "kickoff", name="uq_snapshot_match"
        ),
    )

    def to_dict(self):
        import json

        return {
            "id": self.id,
            "public_id": self.public_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "league_id": self.league_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "kickoff": self.kickoff.isoformat() if self.kickoff else None,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "probabilities": json.loads(self.probabilities_json or "{}"),
            "actual_home_goals": self.actual_home_goals,
            "actual_away_goals": self.actual_away_goals,
            "is_correct": self.is_correct,
            "log_loss": self.log_loss,
            "sha256": self.sha256,
        }


class Experiment(db.Model):
    """训练实验追踪:每次训练的全链路元数据(版本归因)。"""

    __tablename__ = "experiments"

    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(
        db.String(36),
        unique=True,
        nullable=False,
        default=lambda: str(__import__("uuid").uuid4()),
    )
    created_at = db.Column(db.DateTime, default=utcnow)
    league_type = db.Column(db.String(50), nullable=False)
    dataset_version = db.Column(db.String(40), nullable=False)
    feature_version = db.Column(db.String(40), nullable=False)
    model_version = db.Column(db.String(20), nullable=False)
    train_start = db.Column(db.DateTime, nullable=True)
    train_end = db.Column(db.DateTime, nullable=True)
    hyperparameters_json = db.Column(db.Text, nullable=True)
    metrics_json = db.Column(db.Text, nullable=True)
    calibration_json = db.Column(db.Text, nullable=True)
    data_hash = db.Column(db.String(64), nullable=True)
    notes = db.Column(db.String(500), nullable=True)

    def to_dict(self):
        import json

        return {
            "id": self.id,
            "public_id": self.public_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "league_type": self.league_type,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "model_version": self.model_version,
            "metrics": json.loads(self.metrics_json or "{}"),
            "data_hash": self.data_hash,
            "notes": self.notes,
        }


class FeatureStore(db.Model):
    """特征注册表:特征列 → 6 大族 + 版本(支持特征维度归因/回滚)。"""

    __tablename__ = "feature_store"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=utcnow)
    league_type = db.Column(db.String(50), nullable=False)
    feature_name = db.Column(db.String(80), nullable=False)
    family = db.Column(db.String(30), nullable=False)
    version = db.Column(db.String(16), nullable=False)
    formula_hash = db.Column(db.String(64), nullable=False)
    status = db.Column(db.String(10), nullable=False, default="active")
    __table_args__ = (
        db.UniqueConstraint(
            "league_type", "feature_name", "version", name="uq_feature_version"
        ),
    )

    def to_dict(self):
        return {
            "feature_name": self.feature_name,
            "family": self.family,
            "version": self.version,
            "status": self.status,
        }


class TeamMatchStats(db.Model):
    """每队每场指标(V2:matches 胖表拆分,新增指标无需改主表)。"""

    __tablename__ = "team_match_stats"

    id = db.Column(db.Integer, primary_key=True)
    match_id = db.Column(db.Integer, db.ForeignKey("matches.id"), nullable=False)
    team_id = db.Column(db.Integer, db.ForeignKey("teams.id"), nullable=True)
    side = db.Column(db.String(10), nullable=False)  # home/away
    xg = db.Column(db.Float, nullable=True)
    shots = db.Column(db.Integer, nullable=True)
    shots_on_target = db.Column(db.Integer, nullable=True)
    corners = db.Column(db.Integer, nullable=True)
    possession = db.Column(db.Float, nullable=True)
    yellow_cards = db.Column(db.Integer, nullable=True)
    red_cards = db.Column(db.Integer, nullable=True)
    ht_goals = db.Column(db.Integer, nullable=True)
    passing_accuracy = db.Column(db.Float, nullable=True)
    xg_chain = db.Column(db.Float, nullable=True)
    efficiency = db.Column(db.Float, nullable=True)
    transition_speed = db.Column(db.Float, nullable=True)
    defensive_actions = db.Column(db.Float, nullable=True)
    counter_attacks = db.Column(db.Float, nullable=True)
    tactical_rating = db.Column(db.Float, nullable=True)
    experience = db.Column(db.Float, nullable=True)
    # bzzoiro 扩展统计(0013):深度比赛统计(射门外围/门将/大机会等)
    fouls = db.Column(db.Integer, nullable=True)
    offsides = db.Column(db.Integer, nullable=True)
    tackles = db.Column(db.Integer, nullable=True)
    interceptions = db.Column(db.Integer, nullable=True)
    clearances = db.Column(db.Integer, nullable=True)
    blocked_shots = db.Column(db.Integer, nullable=True)
    big_chances = db.Column(db.Integer, nullable=True)
    total_saves = db.Column(db.Integer, nullable=True)
    shots_inside_box = db.Column(db.Integer, nullable=True)
    shots_outside_box = db.Column(db.Integer, nullable=True)
    __table_args__ = (
        db.UniqueConstraint("match_id", "side", name="uq_team_match_side"),
    )


class MatchOdds(db.Model):
    """收盘赔率(bzzoiro /odds 子资源;1x2 + O/U + BTTS)。"""

    __tablename__ = "match_odds"
    id = db.Column(db.Integer, primary_key=True)
    match_id = db.Column(db.Integer, db.ForeignKey("matches.id"), nullable=True)
    league_id = db.Column(db.Integer, nullable=True)
    home_team = db.Column(db.String(100), nullable=True)
    away_team = db.Column(db.String(100), nullable=True)
    event_date = db.Column(db.DateTime, nullable=True)
    home_win = db.Column(db.Float, nullable=True)
    draw = db.Column(db.Float, nullable=True)
    away_win = db.Column(db.Float, nullable=True)
    over_15 = db.Column(db.Float, nullable=True)
    under_15 = db.Column(db.Float, nullable=True)
    over_25 = db.Column(db.Float, nullable=True)
    under_25 = db.Column(db.Float, nullable=True)
    over_35 = db.Column(db.Float, nullable=True)
    under_35 = db.Column(db.Float, nullable=True)
    btts_yes = db.Column(db.Float, nullable=True)
    btts_no = db.Column(db.Float, nullable=True)
    updated_at = db.Column(db.DateTime, nullable=True)
