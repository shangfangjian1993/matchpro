"""统一数据清洗层:任意数据源原始记录 → NormalizedMatch(规范中间格式)。

清洗职责(入库前完成,入库后不再改名/补数):
  1. 队名归一化(team_names.normalize,一次映射终身生效)
  2. 日期解析与标准化(ISO/双位年/时间戳 → datetime)
  3. 字段映射(各源列名 → 统一字段)
  4. 类型转换与校验(比分非负整数、状态合法)
  5. 分类信息生成(league_type 枚举值、赛季标签、match_status)
"""

from dataclasses import asdict, dataclass
from datetime import datetime

# 可入库的指标字段(与 matches 表列对应)
from app.core.config import MATCH_METRIC_COLUMNS as METRIC_FIELDS
from app.data.canonical.config import FDCO_COLUMN_MAP
from app.data.canonical.team_names import normalize

VALID_STATUS = {"finished", "scheduled", "postponed", "cancelled"}


@dataclass
class NormalizedMatch:
    """清洗后的统一比赛记录(入库中间格式)"""
    league_type: str                      # 小写枚举值,如 premier_league
    date: datetime
    home_team: str                        # 规范名(清洗后)
    away_team: str
    match_status: str = "finished"
    home_team_id: int | None = None       # 球队实体 id(ingest 阶段回填)
    away_team_id: int | None = None
    home_goals: int | None = None
    away_goals: int | None = None
    season_label: str = ""                # 分类用:"2016-2017"(推导或源提供)
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
        from app.core.timeutil import utcnow
        return utcnow().fromtimestamp(value)
    s = str(value).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
                "%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s[:19], fmt)
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


def _set_metric(m: NormalizedMatch, field_name: str, value):
    """按字段类型写入指标(仅限白名单字段)"""
    if field_name not in METRIC_FIELDS:
        return
    if field_name in ("match_stage",):
        setattr(m, field_name, str(value).strip() if value else None)
    elif field_name.endswith(("_xg", "possession", "accuracy")):
        setattr(m, field_name, _to_float(value))
    else:
        setattr(m, field_name, _to_int(value))


def _cleanse_common(league_type: str, date, home, away, status,
                    unmatched: list | None = None) -> NormalizedMatch | None:
    """公共校验:队名归一化 + 日期 + 状态;不合法返回 None"""
    if date is None:
        return None
    home = normalize(home or "", unmatched)
    away = normalize(away or "", unmatched)
    if not home or not away or home == away:
        return None
    if status not in VALID_STATUS:
        # 未知状态(含大小写变体)不再静默降级为 scheduled:记入 unmatched 供排查
        if unmatched is not None:
            unmatched.append(f"status:{status}")
        return None
    m = NormalizedMatch(
        league_type=league_type, date=date,
        home_team=home, away_team=away, match_status=status,
    )
    m.season_label = derive_season_label(date)
    return m


# ---------------------------------------------------------------- 各源清洗器

def cleanse_fdco_row(row: dict, league_type: str,
                     unmatched: list | None = None) -> NormalizedMatch | None:
    """football-data.co.uk CSV 行 → NormalizedMatch"""
    m = _cleanse_common(league_type, _parse_date(row.get("Date")),
                        row.get("HomeTeam"), row.get("AwayTeam"), "finished", unmatched)
    if m is None:
        return None
    m.home_goals = _to_int(row.get("FTHG"))
    m.away_goals = _to_int(row.get("FTAG"))
    for col, field in FDCO_COLUMN_MAP.items():
        _set_metric(m, field, row.get(col))
    if m.match_status == "finished" and (m.home_goals is None or m.away_goals is None):
        return None
    return m


def cleanse_fdo_row(row: dict, league_type: str,
                    unmatched: list | None = None) -> NormalizedMatch | None:
    """football-data.org API match 对象 → NormalizedMatch"""
    status_map = {
        "FINISHED": "finished",
        "AWARDED": "finished",          # 判负场:视为完赛,提取 fullTime 比分
        "SCHEDULED": "scheduled", "TIMED": "scheduled",
        "POSTPONED": "postponed", "CANCELLED": "cancelled",
        # IN_PLAY/PAUSED/SUSPENDED 等瞬时状态:归 scheduled,下次抓取刷新
    }
    status = status_map.get(row.get("status", ""), "scheduled")
    m = _cleanse_common(league_type, _parse_date(row.get("utcDate")),
                        (row.get("homeTeam") or {}).get("name"),
                        (row.get("awayTeam") or {}).get("name"), status, unmatched)
    if m is None:
        return None
    if status == "finished":
        score = (row.get("score") or {}).get("fullTime") or {}
        m.home_goals = _to_int(score.get("home"))
        m.away_goals = _to_int(score.get("away"))
    return m


def cleanse_understat_row(row: dict, league_type: str,
                          unmatched: list | None = None) -> NormalizedMatch | None:
    """understat dates 数组元素 → NormalizedMatch(仅回填 xG,其余字段 None)"""
    if not row.get("isResult"):
        return None
    m = _cleanse_common(league_type, _parse_date(row.get("datetime")),
                        (row.get("h") or {}).get("title"),
                        (row.get("a") or {}).get("title"), "finished", unmatched)
    if m is None:
        return None
    xg = row.get("xG") or {}
    m.home_xg = _to_float(xg.get("h"))
    m.away_xg = _to_float(xg.get("a"))
    goals = row.get("goals") or {}
    m.home_goals = _to_int(goals.get("h"))
    m.away_goals = _to_int(goals.get("a"))
    return m


def validate(m: NormalizedMatch) -> list:
    """入库前校验,返回错误列表(空 = 合法)"""
    errors = []
    if m.match_status == "finished" and (m.home_goals is None or m.away_goals is None):
        errors.append(f"{m.home_team} vs {m.away_team} {m.date}: 已完赛缺少比分")
    if m.match_status not in VALID_STATUS:
        errors.append(f"{m.home_team} vs {m.away_team}: 非法状态 {m.match_status}")
    return errors


def cleanse_apifootball_row(row: dict, league_type: str,
                            unmatched: list | None = None) -> NormalizedMatch | None:
    """api-football fixture → NormalizedMatch(与 fdo 结构适配)。

    api-football fixture 结构: {fixture: {date, status.short}, teams: {home/away.name},
                                goals: {home/away}};与 fdo(homeTeam/score.fullTime)不同。
    """
    status_map = {
        "FT": "finished", "AET": "finished", "PEN": "finished",
        "NS": "scheduled", "TBD": "scheduled", "PST": "postponed", "CANC": "cancelled",
    }
    fixture = row.get("fixture") or {}
    status = status_map.get((fixture.get("status") or {}).get("short", ""), "scheduled")
    m = _cleanse_common(
        league_type,
        _parse_date(fixture.get("date")),
        (row.get("teams") or {}).get("home", {}).get("name"),
        (row.get("teams") or {}).get("away", {}).get("name"),
        status, unmatched,
    )
    if m is None:
        return None
    if status == "finished":
        goals = row.get("goals") or {}
        m.home_goals = _to_int(goals.get("home"))
        m.away_goals = _to_int(goals.get("away"))
    return m

