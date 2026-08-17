"""SoccerData(FBref)enrich 接入 —— 补齐 matches 扩展统计列。

背景:soccerdata(Apache-2.0,probberechts/soccerdata)FBref 提供 5 大联赛
赛后全量统计。本模块用 read_team_match_stats(stat_type='shooting'/'misc')
回填 matches 的 xG/射门/射正/角球/黄红牌等扩展列(xG 覆盖 77%→95%+)。

防泄漏:enrich 只回填"已完赛且扩展列缺失"的历史场次(赛后补录),
与赛前预测无关 —— 预测仍只消费 `< 比赛日期` 数据。

依赖:soccerdata(可选 —— 未安装时模块可用性检查返回 False,系统正常降级)。
"""
from __future__ import annotations

import logging
import os
import re

import pandas as pd

logger = logging.getLogger(__name__)

LEAGUE_CODES = {
    "premier_league": "ENG-Premier League",
    "la_liga": "ESP-La Liga",
    "bundesliga": "GER-Bundesliga",
    "serie_a": "ITA-Serie A",
    "ligue_1": "FRA-Ligue 1",
}

# 已确认存在的 stat_type:schedule/keeper/shooting/misc
STAT_TYPES = ("shooting", "misc")

# FBref 常见队名后缀归一(足以匹配我们 DB 的中英映射后的英文名)
_TEAM_ALIAS = {
    "manchester united": "man united",
    "manchester city": "man city",
    "wolverhampton": "wolves",
    "brighton": "brighton",
    "nottingham": "nottingham",
    "leicester": "leicester",
    "west ham": "west ham",
}


def available() -> bool:
    try:
        import soccerdata  # noqa: F401
        return True
    except ImportError:
        return False


def _norm(name: str) -> str:
    """队名归一:去 {FC/CF/SC/Wanderers/Athletic 等}、小写、去非字母数字。"""
    n = re.sub(r"\b(?:fc|cf|sc|afc|acf|wanderers|athletic|club)\b", "", name.lower())
    n = re.sub(r"[^a-z0-9 ]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    n = _TEAM_ALIAS.get(n, n)
    return n


def _fetch_fbref(league_code: str, season: str):
    """拉取某联赛一赛季的球队比赛日志(合并 shooting + misc)。"""
    import soccerdata as sd
    fbref = sd.FBref(leagues=league_code, seasons=season)
    merged = None
    for st in STAT_TYPES:
        try:
            df = fbref.read_team_match_stats(stat_type=st, force_cache=False)
        except Exception as e:
            logger.warning("FBref %s 读取失败: %s", st, e)
            continue
        if merged is None:
            merged = df
        else:
            _drop = [c for c in df.columns
                     if c in merged.columns and c not in ("team", "opponent", "date")]
            merged = merged.join(df[_drop], how="outer")
    return merged


def _match_team(team_fb: str, known: dict) -> str | None:
    """FBref 队名 → 我们 DB 队名(归一后精确/子串)。"""
    n = _norm(team_fb)
    if not n:
        return None
    if n in known:
        return known[n]
    for db_name, norm_db in known.items():
        if n == norm_db or (len(n) >= 8 and (n in norm_db or norm_db in n)):
            return db_name
    return None


def enrich_matches(league_type, rows, season_start: str = "2024",
                   verbose: bool = True) -> dict:
    """用 FBref 回填给定聊条完赛场次的统计列。

    rows: Match ORM 列表(应已过滤为 '完赛且扩展列缺失' 且同 league)。
    返回 {"fetched": n, "updated": n, "unmatched": n, "errors": n}。
    """
    code = LEAGUE_CODES.get(league_type.value)
    if code is None:
        return {"fetched": 0, "updated": 0, "unmatched": len(rows), "errors": 0}
    df = _fetch_fbref(code, season_start)
    if df is None or df.empty:
        logger.warning("FBref 无数据: %s %s", code, season_start)
        return {"fetched": 0, "updated": 0, "unmatched": len(rows), "errors": 1}

    # 已知队名索引(归一化)
    known = {}
    for m in rows:
        known.setdefault(m.home_team, _norm(m.home_team))
        known.setdefault(m.away_team, _norm(m.away_team))

    # FBref:每队一行(venue=Home/Away)→ 重建比赛级 {date, home, away} 记录
    matches = {}
    for _, r in df.iterrows():
        try:
            team = _match_team(str(r.get("team", "")), known)
            opp = _match_team(str(r.get("opponent", "")), known)
            if team is None or opp is None:
                continue
            is_home = str(r.get("venue", "")).lower().startswith("home")
            key = (str(pd.Timestamp(r["date"]).date()), team, opp)
            side = "home" if is_home else "away"
            rec = matches.setdefault(key, {})
            side_rec = rec.setdefault(side, {})
            # 复制统计列(队级列,去元数据)
            for col, val in r.items():
                cs = str(col)
                if cs in ("team", "opponent", "venue", "date", "season", "league"):
                    continue
                side_rec[cs] = val
        except Exception:
            continue

    from app.api.db import db
    updated = unmatched = errors = 0
    for m in rows:
        key = (str(pd.Timestamp(m.match_date).date()), m.home_team, m.away_team)
        rec = matches.get(key)
        if not rec:
            unmatched += 1
            continue
        try:
            h, a = rec.get("home", {}), rec.get("away", {})
            # xG/射门/射正/角球/黄红牌(列名以 FBref shooting/misc 为准,get 防御)
            m.home_xg = _g(h.get("xg")) if _g(h.get("xg")) is not None else m.home_xg
            m.away_xg = _g(a.get("xg")) if _g(a.get("xg")) is not None else m.away_xg
            m.home_shots = _g(h.get("shots")) if _g(h.get("shots")) is not None else m.home_shots
            m.away_shots = _g(a.get("shots")) if _g(a.get("shots")) is not None else m.away_shots
            m.home_shots_on_target = _g(h.get("sot")) if _g(h.get("sot")) is not None else m.home_shots_on_target
            m.away_shots_on_target = _g(a.get("sot")) if _g(a.get("sot")) is not None else m.away_shots_on_target
            m.home_corners = _g(h.get("ck")) if _g(h.get("ck")) is not None else m.home_corners
            m.away_corners = _g(a.get("ck")) if _g(a.get("ck")) is not None else m.away_corners
            m.home_yellow_cards = _g(h.get("crd_y")) if _g(h.get("crd_y")) is not None else m.home_yellow_cards
            m.away_yellow_cards = _g(a.get("crd_y")) if _g(a.get("crd_y")) is not None else m.away_yellow_cards
            m.home_red_cards = _g(h.get("crd_r")) if _g(h.get("crd_r")) is not None else m.home_red_cards
            m.away_red_cards = _g(a.get("crd_r")) if _g(a.get("crd_r")) is not None else m.away_red_cards
            updated += 1
        except Exception:
            errors += 1
    db.session.commit()
    return {"fetched": len(df), "updated": updated, "unmatched": unmatched, "errors": errors}


def _g(v):
    """值清洗:NaN/str 空白 → None;float 化。"""
    try:
        import pandas as pd
        if v is None or (isinstance(v, float) and v != v) or (isinstance(v, str) and not v.strip()):
            return None
        return float(v)
    except Exception:
        return None
