"""Zafronix Sports API 接入(free key:世界杯/欧冠/欧洲杯历史全量)。

- 免费套餐实测可用:5 连发 200 无 429;支持 ?year= 过滤与 cursor 分页
- 覆盖:世界杯 1930-2026 / 欧冠 1955- / 欧洲杯 1960- / 欧联/欧协/国家联赛等
- 字段:date/kickoff/stage/homeTeam/awayTeam/scores/extraTime/penalties/
  venue/attendance/goals(进球事件级)
- 价值:大幅补全我们 DB 的欧战/国家队历史(当前 CL 503/EC 51/WC 168
  → 世界杯 64×N 届、欧洲杯 1960 起、欧冠 1955 起)

防泄漏:只入库"已完赛"场次;upsert 幂等(复用 canonical.ingest.upsert_matches)。
"""
from __future__ import annotations

import logging
import os
import time
import urllib.request

from app.data.canonical.cleanse import NormalizedMatch

logger = logging.getLogger(__name__)

BASE = "https://api.zafronix.com"
# LeagueType.value → 赛事路径
TOURNAMENT_PATHS = {
    "champions_league": "uefa/championsleague",
    "european_championship": "uefa/euro",
    "world_cup": "fifa/worldcup",
    "europa_league": "uefa/europaleague",
    "europa_conference_league": "uefa/conferenceleague",
    "nations_league": "uefa/nationsleague",
}

REQUEST_INTERVAL = 0.6  # 免费套餐稳健间隔


def _key() -> str:
    k = os.environ.get("ZAFRONIX_KEY") or ""
    if not k:
        for line in open(os.path.join(str(__import__("app.core.paths", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT), ".env"), encoding="utf-8"):
            if line.startswith("ZAFRONIX_KEY"):
                k = line.split("=", 1)[1].strip()
                break
    return k


def available() -> bool:
    return bool(_key())


def fetch_matches(league_type_value: str, year: int | None = None,
                  limit: int = 100, cursor: str | None = None) -> list[dict]:
    """拉赛事比赛(分页);返回 data 列表。"""
    path = TOURNAMENT_PATHS.get(league_type_value)
    if path is None:
        raise ValueError(f"Zafronix 未支持: {league_type_value}(可用 {list(TOURNAMENT_PATHS)})")
    url = f"{BASE}/{path}/v1/matches?limit={limit}"
    if year:
        url += f"&year={year}"
    if cursor:
        url += f"&cursor={cursor}"
    req = urllib.request.Request(url, headers={"X-API-Key": _key()})
    with urllib.request.urlopen(req, timeout=25) as r:
        import json
        d = json.load(r)
    return d.get("data") or []


def to_normalized(raw: dict, league_type_value: str, year: int) -> NormalizedMatch:
    """Zafronix 原始行 → NormalizedMatch。"""
    import datetime
    date = datetime.date.fromisoformat(str(raw["date"]))
    season_label = f"{year}-{year + 1}" if raw.get("season") is None else str(raw.get("season"))
    return NormalizedMatch(
        league_type=league_type_value,
        date=datetime.datetime.combine(date, datetime.time(12, 0)),
        home_team=str(raw["homeTeam"]).strip(),
        away_team=str(raw["awayTeam"]).strip(),
        home_goals=int(raw["homeScore"]) if raw.get("homeScore") is not None else None,
        away_goals=int(raw["awayScore"]) if raw.get("awayScore") is not None else None,
        season_label=season_label,
        match_stage=str(raw.get("stage") or ""),
    )


def ingest_year(league_type_value: str, year: int, verbose: bool = True) -> dict:
    """拉某赛事指定年份全部比赛并 upsert;返回 upsert 汇总。"""
    from app.data.canonical.ingest import upsert_matches
    rows, cursor = [], None
    while True:
        batch = fetch_matches(league_type_value, year=year, limit=100, cursor=cursor)
        rows.extend(batch)
        if len(batch) < 100:
            break
        cursor = batch[-1].get("id") if batch else None
        time.sleep(REQUEST_INTERVAL)
    normalized = [to_normalized(r, league_type_value, year) for r in rows]
    res = upsert_matches(normalized)
    if verbose:
        print(f"  {league_type_value} {year}: 拉取 {len(rows)} 场 → "
              f"新增 {res['inserted']} 更新 {res['updated']} 跳过 {res['skipped']} "
              f"错误 {len(res['errors'])}", flush=True)
    res["fetched"] = len(rows)
    return res
