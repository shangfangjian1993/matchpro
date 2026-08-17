"""api-football(api-sports.io)统计接入 —— 替代 FBref 回填比赛扩展列。

背景:FBref 在数据中心 IP 被 Cloudflare CAPTCHA 墙栏(容器/常规服务器不可用)。
api-football 是正式 API(无反爬、数据中心 IP 可用、已有 .env API_FOOTBALL_KEY)。

实测(曼联 vs 富勒姆 2024):返回 18 项统计 —— 射门/射正/控球/角球/犯规/黄红/
传球数/传球成功率;注意:**无 xG**(免费层)。xG 缺口仍由 understat 补(77% 覆盖),
StatsBomb 开放数据只到 2020/21(历史 xG 可补)。

限制:Free 套餐 100 次/天(每场 statistics 一次调用) → 分批回填。
防泄漏:只回填"已完赛且统计缺失"的历史场次(赛后补录)。
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request

logger = logging.getLogger(__name__)

# api-football league id(5 大联赛)
LEAGUE_IDS = {
    "premier_league": 39,
    "la_liga": 140,
    "bundesliga": 78,
    "serie_a": 135,
    "ligue_1": 61,
}

# statistic type → (matches 列 主, 客)
_STAT_MAP = {
    "Total Shots": ("home_shots", "away_shots"),
    "Shots on Goal": ("home_shots_on_target", "away_shots_on_target"),
    # 控球:maches 仅 home_possession 列(客队 = 100 - 主;由特征层推)
    "Ball Possession": ("home_possession", None),
    "Corner Kicks": ("home_corners", "away_corners"),
    "Yellow Cards": ("home_yellow_cards", "away_yellow_cards"),
    "Red Cards": ("home_red_cards", "away_red_cards"),
    "Passes %": ("home_passing_accuracy", "away_passing_accuracy"),
}

_TEAM_ALIAS = {
    "manchester united": "man united",
    "manchester city": "man city",
    "wolverhampton": "wolves",
    "nottingham": "nottingham",
    "brighton": "brighton",
    "west ham": "west ham",
}


def _key() -> str:
    k = os.environ.get("API_FOOTBALL_KEY") or ""
    if not k:
        for line in open(os.path.join(str(__import__("app.core.paths", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT), ".env"), encoding="utf-8"):
            if line.startswith("API_FOOTBALL_KEY"):
                k = line.split("=", 1)[1].strip()
                break
    return k


def _norm(name: str) -> str:
    n = re.sub(r"\b(?:fc|cf|sc|afc|acf|wanderers)\b", "", str(name).lower())
    n = re.sub(r"[^a-z0-9 ]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return _TEAM_ALIAS.get(n, n)


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"x-apisports-key": _key()})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def _season_of(date):
    """由场次日期推断 api 赛季起始年:8 月起算新赛季。"""
    if hasattr(date, "month"):
        return str(date.year if date.month >= 8 else date.year - 1)
    return str(date)[:4]


def _fixture_id(league_id: int, season: str, date: str, home: str, away: str) -> int | None:
    """按日期+球队找 api fixture id(1 次 fixtures 调用/日期,可缓存)。"""
    d = _get(f"https://v3.football.api-sports.io/fixtures?league={league_id}&season={season}"
             f"&from={date}&to={date}")
    for f in d.get("response", []):
        hn, an = _norm(f["teams"]["home"]["name"]), _norm(f["teams"]["away"]["name"])
        dbh, dba = _norm(home), _norm(away)
        if ((hn == dbh or (len(hn) >= 6 and len(dbh) >= 6 and (hn in dbh or dbh in hn)))
                and (an == dba or (len(an) >= 6 and len(dba) >= 6 and (an in dba or dba in an)))):
            return f["fixture"]["id"]
    return None


def available() -> bool:
    return bool(_key())


def enrich_matches(league_type, rows, season: str = "2024", verbose: bool = True) -> dict:
    """批量回填:rows 为已完赛且统计缺失的 Match 列表(同联赛)。

    返回 {"attempted": n, "updated": n, "unmatched": n, "errors": n, "calls": n}。
    注意:每场 statistics 计 1 次调用;free 套餐 100/天,分批执行。
    """
    league_id = LEAGUE_IDS.get(league_type.value)
    if league_id is None:
        return {"attempted": len(rows), "updated": 0, "unmatched": len(rows), "errors": 0, "calls": 0}
    from app.api.db import db
    updated = unmatched = errors = calls = 0
    key = _key()
    if not key:
        logger.error("缺少 API_FOOTBALL_KEY")
        return {"attempted": len(rows), "updated": 0, "unmatched": len(rows), "errors": 0, "calls": 0}
    for m in rows:
        date = str(m.match_date.date())
        try:
            fid = _fixture_id(league_id, _season_of(m.match_date), date, m.home_team, m.away_team)
            calls += 1
            if fid is None:
                unmatched += 1
                continue
            d = _get(f"https://v3.football.api-sports.io/fixtures/statistics?fixture={fid}")
            calls += 1
            resp = d.get("response") or []
            if len(resp) != 2:
                unmatched += 1
                continue
            home_stats = {s["type"]: s["value"] for s in resp[0]["statistics"]}
            away_stats = {s["type"]: s["value"] for s in resp[1]["statistics"]}
            def _num(v):
                try:
                    if v is None:
                        return None
                    s = str(v).strip().replace("%", "")
                    return float(s) if s else None
                except Exception:
                    return None
            existing = {col.name for col in m.__table__.columns}
            for stat_type, (hc, ac) in _STAT_MAP.items():
                if hc and hc in existing:
                    hv = _num(home_stats.get(stat_type))
                    if hv is not None:
                        setattr(m, hc, hv)
                if ac and ac in existing:
                    av = _num(away_stats.get(stat_type))
                    if av is not None:
                        setattr(m, ac, av)
            updated += 1
        except Exception as e:
            errors += 1
            if verbose and errors <= 3:
                logger.warning("回填失败 %s vs %s: %s", m.home_team, m.away_team, e)
    db.session.commit()
    return {"attempted": len(rows), "updated": updated, "unmatched": unmatched,
            "errors": errors, "calls": calls}
