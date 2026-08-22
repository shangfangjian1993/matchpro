"""数据规范化:任意数据源原始记录 → NormalizedMatch(规范中间格式)。

清洗职责(入库前完成,入库后不再改名/补数):
  1. 队名归一化(team_names.normalize,一次映射终身生效)
  2. 日期解析与标准化(ISO/双位年/时间戳 → datetime)
  3. 字段映射(各源列名 → 统一字段)
  4. 类型转换与校验(比分非负整数、状态合法)
  5. 分类信息生成(league_type 枚举值、赛季标签、match_status)
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from app.data.canonical.team_names import normalize as normalize_name

VALID_STATUS = {"finished", "scheduled", "postponed", "cancelled"}


@dataclass
class NormalizedMatch:
    """清洗后的统一比赛记录(入库中间格式)"""

    league_type: str  # 小写枚举值,如 premier_league
    date: datetime
    home_team: str  # 规范名(清洗后)
    away_team: str
    match_status: str = "finished"
    home_team_id: int | None = None  # 球队实体 id(ingest 阶段回填)
    away_team_id: int | None = None
    home_goals: int | None = None
    away_goals: int | None = None
    season_label: str = ""  # 分类用:"2016-2017"(推导或源提供)
    # 指标字段(可选)
    home_xg: float | None = None
    away_xg: float | None = None
    home_shots: int | None = None
    away_shots: int | None = None
    home_shots_on_target: int | None = None
    away_shots_on_target: int | None = None
    home_corners: int | None = None
    away_corners: int | None = None
    home_possession: float | None = None
    home_yellow_cards: int | None = None
    away_yellow_cards: int | None = None
    home_red_cards: int | None = None
    away_red_cards: int | None = None
    home_ht_goals: int | None = None
    away_ht_goals: int | None = None
    home_passing_accuracy: float | None = None
    away_passing_accuracy: float | None = None
    match_stage: str | None = None

    def to_row(self) -> dict:
        return asdict(self)


def derive_season_label(date: datetime) -> str:
    """按 8 月为界推导赛季标签:2016-08~2017-07 -> '2016-2017'"""
    y = date.year
    if date.month >= 8:
        return f"{y}-{y + 1}"
    return f"{y - 1}-{y}"


def _parse_date(value) -> datetime | None:
    """日期解析:ISO 字符串 / DD/MM/YYYY / DD/MM/YY / 时间戳"""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    s = str(value).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d/%m/%y",
    ):
        try:
            return datetime.strptime(s[:19], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _to_int(v) -> int | None:
    if v is None or (isinstance(v, str) and v.strip() in ("", "-")):
        return None
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return None


def _to_float(v) -> float | None:
    if v is None or (isinstance(v, str) and v.strip() in ("", "-")):
        return None
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def _cleanse_common(
    league_type: str, date, home, away, status, unmatched: list | None = None
) -> NormalizedMatch | None:
    """公共校验:队名归一化 + 日期 + 状态;不合法返回 None"""
    if date is None:
        return None
    home = normalize_name(home or "", unmatched)
    away = normalize_name(away or "", unmatched)
    if not home or not away or home == away:
        return None
    if status not in VALID_STATUS:
        if unmatched is not None:
            unmatched.append(f"status:{status}")
        return None
    m = NormalizedMatch(
        league_type=league_type,
        date=date,
        home_team=home,
        away_team=away,
        match_status=status,
    )
    m.season_label = derive_season_label(date)
    return m


def normalize_row(raw: dict, source: str, league_type: str) -> NormalizedMatch | None:
    """根据来源类型选择对应的清洗器。"""
    if source in ("bzzoiro_events", "bzzoiro"):
        return _normalize_bzzoiro(raw, league_type)
    if source == "fdco":
        return _normalize_fdco(raw, league_type)
    if source == "understat":
        return _normalize_understat(raw, league_type)
    return None


def _normalize_bzzoiro(raw: dict, league_type: str) -> NormalizedMatch | None:
    """bzzoiro event → NormalizedMatch。"""
    date = _parse_date(raw.get("event_date"))
    return _cleanse_common(
        league_type=league_type,
        date=date,
        home=raw.get("home_team", ""),
        away=raw.get("away_team", ""),
        status="finished" if raw.get("status") == "finished" else "scheduled",
    )


def _normalize_fdco(row: dict, league_type: str) -> NormalizedMatch | None:
    """fdco CSV 行 → NormalizedMatch。"""
    date_str = row.get("Date", "")
    date = _parse_date(date_str)
    if date is None:
        return None
    home = row.get("HomeTeam", row.get("Home", ""))
    away = row.get("AwayTeam", row.get("Away", ""))
    m = _cleanse_common(league_type, date, home, away, "finished")
    if m is None:
        return None
    # fdco 指标字段
    m.home_goals = _to_int(row.get("FTHG"))
    m.away_goals = _to_int(row.get("FTAG"))
    m.home_ht_goals = _to_int(row.get("HTHG"))
    m.away_ht_goals = _to_int(row.get("HTAG"))
    m.home_shots = _to_int(row.get("HS"))
    m.away_shots = _to_int(row.get("AS"))
    m.home_shots_on_target = _to_int(row.get("HST"))
    m.away_shots_on_target = _to_int(row.get("AST"))
    m.home_corners = _to_int(row.get("HC"))
    m.away_corners = _to_int(row.get("AC"))
    m.home_yellow_cards = _to_int(row.get("HY"))
    m.away_yellow_cards = _to_int(row.get("AY"))
    m.home_red_cards = _to_int(row.get("HR"))
    m.away_red_cards = _to_int(row.get("AR"))
    return m


def _normalize_understat(raw: dict, league_type: str) -> NormalizedMatch | None:
    """understat → NormalizedMatch (仅 xG 回填)。"""
    # understat 数据用于回填 xG,不创建新 match 记录
    return None
