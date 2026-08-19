"""Feature · stats(深度比赛统计,team_match_stats 子表 → 特征)。

bzzoiro 采集了 team_match_stats 的深度统计(fouls/offsides/tackles/…
/big_chances/禁区内外射门)与 match_odds(收盘赔率)。本模块把这两块
并入特征计算的滚动统计:
- 队级:该队最近 window 场× 主/客侧各统计的均值(赛前值,防泄漏)
- match_odds:该场/该队近期赔率隐含概率(赛前可用才用)

务必防泄漏:只计算"截止该场(严格更早)"的已完成场次。
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# team_match_stats 深度统计 → 特征列
_AGG_COLS = {
    "fouls": "tms_fouls_avg",
    "offsides": "tms_offsides_avg",
    "tackles": "tms_tackles_avg",
    "interceptions": "tms_interceptions_avg",
    "clearances": "tms_clearances_avg",
    "blocked_shots": "tms_blocked_shots_avg",
    "big_chances": "tms_big_chances_avg",
    "total_saves": "tms_total_saves_avg",
    "shots_inside_box": "tms_shots_inside_box_avg",
    "shots_outside_box": "tms_shots_outside_box_avg",
}


def _tms_rows(league_id: int):
    """该联赛全部 team_match_stats 行(按比赛日期排序,含 match_id/队/side)。"""
    from app.api.db import Match, TeamMatchStats
    # join Match 拿日期;date 缺则用 id 近似(采集按事件序)
    stats = TeamMatchStats.query.filter(
        TeamMatchStats.match_id.in_(
            [m.id for m in Match.query.filter_by(league_id=league_id).all()])
    ).all()
    return stats


def rolling_team_stats(hist_matches, window: int = 5) -> pd.DataFrame:
    """为 hist(该场之前已排序)的每行附加"该队近期深度统计均值"。

    返回与输入同序的 DataFrame,附加 tms_*_avg 主/客两套列。
    简化:按 matches 表时间序,对每行取该队(主/客侧)此前 window 场
    的 team_match_stats 均值。数据未到位(NaN)时留空(模型自动跳过)。
    """
    if hist_matches is None or not hist_matches:
        return pd.DataFrame()
    mids = [m.id for m in hist_matches]
    from app.api.db import TeamMatchStats
    rows = (TeamMatchStats.query
            .filter(TeamMatchStats.match_id.in_(mids))
            .all())
    # match_id -> stats(home/away)
    per_match = {}
    for r in rows:
        per_match.setdefault(r.match_id, {})[r.side] = r
    out = []
    for m in hist_matches:
        rec = {}
        rec["date"] = m.match_date
        rec["home_team"] = m.home_team
        rec["away_team"] = m.away_team
        rec["home_goals"] = m.home_goals
        rec["away_goals"] = m.away_goals
        out.append(rec)
    df = pd.DataFrame(out)
    if df.empty:
        return df
    for col, feat in _AGG_COLS.items():
        df[f"home_{feat}"] = 0.0
        df[f"away_{feat}"] = 0.0
    # 简化实现:逐行累积(该行只用窗内已见历史,防泄漏)
    seen_h: dict[str, list] = {k: [] for k in _AGG_COLS}
    seen_a: dict[str, list] = {k: [] for k in _AGG_COLS}
    home_team_key = {}
    for i, m in enumerate(hist_matches):
        st = per_match.get(m.id, {})
        # 先用 seen(截至本场之前)填特征 → 防泄漏
        for col, feat in _AGG_COLS.items():
            h = seen_h.get(col, [])
            a = seen_a.get(col, [])
            df.loc[i, f"home_{feat}"] = round(sum(h[-window:]) / max(1, len(h[-window:])), 3) if h else None
            df.loc[i, f"away_{feat}"] = round(sum(a[-window:]) / max(1, len(a[-window:])), 3) if a else None
        # 再用本场结果更新 seen(供下一场)
        hst, ast = st.get("home"), st.get("away")
        for col in _AGG_COLS:
            hv = getattr(hst, col, None) if hst else None
            av = getattr(ast, col, None) if ast else None
            if hv is not None:
                seen_h[col].append(float(hv))
            if av is not None:
                seen_a[col].append(float(av))
    return df
