"""StatsBomb 开放数据 xG 接入(免费,Apache-2.0;覆盖 2016/17-2020/21)。

用途:补充 understat 缺口的**历史**事件级 xG(每场全部射门事件的 xG 求和)。
不再使用 FBref(数据中心 IP 被验证码墙)。

用法:ingest 后人工/脚本回填史前赛季缺失 xG:
    python -m app.services.data.ingest --enrich-statsbomb --league premier_league
    或直接调用 enrich_matches(league_type, rows)

注意:StatsBomb 开放数据赛季有限(主流到 2020/21);最近赛季需付费 token。
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# 联赛英文名(与 StatsBomb competition_name 对齐)
_COMPETITION_NAMES = {
    "premier_league": "Premier League",
    "la_liga": "La Liga",
    "bundesliga": "Bundesliga",
    "serie_a": "Serie A",
    "ligue_1": "Ligue 1",
}


def available() -> bool:
    try:
        import statsbombpy  # noqa: F401

        return True
    except ImportError:
        return False


def _norm(name: str) -> str:
    import re

    n = re.sub(r"\b(?:fc|cf|sc|afc|acf|wanderers)\b", "", str(name).lower())
    n = re.sub(r"[^a-z0-9 ]", "", n)
    return re.sub(r"\s+", " ", n).strip()


def _season_matches(competition_id: int, season_id: int):
    """某赛事赛季的全部比赛:{(date, home_norm, away_norm): (home_xg, away_xg)}。"""
    from statsbombpy import sb

    ms = sb.matches(competition_id=competition_id, season_id=season_id)
    out = {}
    for _, r in ms.iterrows():
        date = str(pd.Timestamp(r["match_date"]).date())
        hn, an = _norm(r["home_team"]), _norm(r["away_team"])
        # shots 事件求和
        try:
            shots = sb.events(match_id=r["match_id"], split=True)["shot"]
            home_xg = float(
                shots.loc[shots["team"].apply(_norm) == hn, "shot_statsbomb_xg"].sum()
            )
            away_xg = float(
                shots.loc[shots["team"].apply(_norm) == an, "shot_statsbomb_xg"].sum()
            )
        except Exception as e:
            logger.debug("match %s 事件失败: %s", r["match_id"], e)
            continue
        out[(date, hn, an)] = (round(home_xg, 4), round(away_xg, 4))
    return out


def enrich_matches(league_type, rows, verbose: bool = True) -> dict:
    """回填 rows(缺失 xG 的历史场次)的 home_xg/away_xg。"""
    from statsbombpy import sb

    comp_name = _COMPETITION_NAMES.get(league_type.value)
    if comp_name is None:
        return {"updated": 0, "no_comp": len(rows), "unmatched": len(rows)}
    comps = sb.competitions()
    # 候选赛季:由场次年份推断(取该联赛覆盖范围,先全量拉一次该联赛所有赛季缓存)
    cand = comps[comps["competition_name"] == comp_name]
    if cand.empty:
        return {"updated": 0, "no_comp": len(rows), "unmatched": len(rows)}
    comp_id = int(cand.iloc[0]["competition_id"])
    year_set = {_season_year(m.match_date) for m in rows}
    from app.api.db import db

    updated = 0
    for _, srow in cand.iterrows():
        sid = int(srow["season_id"])
        sname = str(srow["season_name"])  # 形如 2019/2020
        sy = int(sname.split("/")[0])
        if not (year_set & {sy, sy + 1}):
            continue
        try:
            recs = _season_matches(comp_id, sid)
        except Exception as e:
            logger.warning("赛季 %s 拉取失败: %s", sname, e)
            continue
        for m in rows:
            if m.home_xg is not None and m.away_xg is not None:
                continue
            key = (str(m.match_date.date()), _norm(m.home_team), _norm(m.away_team))
            if key in recs:
                hx, ax = recs[key]
                if m.home_xg is None:
                    m.home_xg = hx
                if m.away_xg is None:
                    m.away_xg = ax
                updated += 1
        db.session.flush()
    db.session.commit()
    return {
        "updated": updated,
        "unmatched": len(rows) - updated,
        "checked_seasons": len(cand),
    }


def _season_year(dt):
    return dt.year if dt.month >= 8 else dt.year - 1
