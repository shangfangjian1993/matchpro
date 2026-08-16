"""写库核心:采集结果 → matches 表(复用 api.db 模型,直接写库,不经过 HTTP)。

用法(需 DATABASE_URL 环境变量):
    from app.data.canonical.store import write_matches
    write_matches(league_type_value, rows, season_label)
"""

import logging
import os

logger = logging.getLogger(__name__)


def _app_ctx():
    """初始化 db(原生 SQLAlchemy;采集管道独立使用,不依赖任何 web 框架)。"""
    from app.api.db import db, init_db

    init_db(os.environ.get("DATABASE_URL", None))
    db.create_all()
    return None, db


LEAGUE_CN_NAMES = {
    "premier_league": "英超", "la_liga": "西甲", "bundesliga": "德甲",
    "serie_a": "意甲", "ligue_1": "法甲", "champions_league": "欧冠",
    "world_cup": "世界杯", "european_championship": "欧洲杯",
}


def _normalize_league_type(league_type_value: str) -> str:
    """统一为小写枚举值(LeagueType.X.value);采集器配置表若传大写枚举名
    (如 PREMIER_LEAGUE)则转换"""
    from app.core.config import LeagueType
    try:
        lt = LeagueType[league_type_value] if league_type_value.isupper() else LeagueType(league_type_value)
        return lt.value
    except (KeyError, ValueError):
        return league_type_value.lower()


def _get_or_create_league(db, League, league_type_value: str, season_label: str):
    """按 league_type 找联赛;找不到则创建(采集管道自动建联赛,无需预置)"""
    league_type_value = _normalize_league_type(league_type_value)
    league = League.query.filter_by(league_type=league_type_value).first()
    if league:
        return league
    name = LEAGUE_CN_NAMES.get(league_type_value, league_type_value)
    league = League(name=name, country="", season=season_label, league_type=league_type_value)
    db.session.add(league)
    db.session.commit()
    logger.info("已创建联赛记录: %s (%s)", name, season_label)
    return league


