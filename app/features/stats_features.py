"""Feature · stats(team_match_stats 深度比赛统计 → 球队状态特征)。

审查 ae724d5 P0 修复:按 **球队 × 主客侧** 维护状态(非联赛全局):
  本场特征 home_* = 主队在该场之前最近 window 场的"主场侧"统计均值;
  away_* = 客队"客场侧"均值。每队每侧独立滚动。

- 防泄漏:先取特征(用"该场之前"累积的状态),后更新(本场 stats 计入
  team_state —— 下一场才可见)。
- match_id 对齐:返回 DataFrame index = match_id,由 factory merge
  (不依赖 DataFrame 行序/index 对齐)。
- Odds 状态明确:match_odds(收盘赔率)于赛后收盘,赛前不可得 → 不入赛前
  特征;用于回测评估 / 后处理对比(不在本模块实现,勿误以为已接入)。

外部契约:
    rolling_team_stats(hist_matches, window=5) -> pd.DataFrame
        index = match_id; 列 = home_tms_* / away_tms_*;数据缺失 = NaN
"""
from __future__ import annotations

import collections
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


def _roll(hist_matches, per_match: dict, window: int = 5) -> dict:
    """纯计算:按 球队×主客侧 维护状态,输出 {match_id: {home_tms_*, away_tms_*}}。

    per_match: {match_id: {"home": stats_obj_or_none, "away": ...}}
    hist_matches: Match ORM 列表(已按时间序,该场之前的事件先到)。
    """
    # team_state[team][side][stat] = [有序统计值]
    team_state = collections.defaultdict(
        lambda: {"home": collections.defaultdict(list),
                 "away": collections.defaultdict(list)})
    out: dict = {}
    for m in hist_matches:
        st = per_match.get(m.id, {})
        home, away = m.home_team, m.away_team
        # ── 本场特征(严格用本场之前的状态)──
        rec: dict = {}
        for col, feat in _AGG_COLS.items():
            hs = team_state[home]["home"][col]
            as_ = team_state[away]["away"][col]
            rec[f"home_{feat}"] = (
                round(sum(hs[-window:]) / len(hs[-window:]), 3) if hs else None)
            rec[f"away_{feat}"] = (
                round(sum(as_[-window:]) / len(as_[-window:]), 3) if as_ else None)
        out[m.id] = rec
        # ── 本场结果计入状态(供下一场)──
        hst = st.get("home")
        ast = st.get("away")
        # team_match_stats 的 side=home 记录属于主队(m.home_team)的主场侧;
        # side=away 属于客队(m.away_team)的客场侧。
        if hst is not None:
            for col in _AGG_COLS:
                v = getattr(hst, col, None)
                if v is not None:
                    team_state[home]["home"][col].append(float(v))
        if ast is not None:
            for col in _AGG_COLS:
                v = getattr(ast, col, None)
                if v is not None:
                    team_state[away]["away"][col].append(float(v))
    return out


def rolling_team_stats(hist_matches, window: int = 5) -> pd.DataFrame:
    """训练/预测历史(match 序)的球队状态滚动特征(index = match_id)。

    hist_matches 为空 → 返回空 DataFrame(数据缺失可降级,调用方留 NaN)。
    实现/schema 异常 → 抛出(不做静默降级 —— 由上层决定 fail/invalid)。
    """
    stat_cols = [f"home_{v}" for v in _AGG_COLS.values()] +                 [f"away_{v}" for v in _AGG_COLS.values()]
    if not hist_matches:
        return pd.DataFrame(columns=stat_cols)
    mids = [m.id for m in hist_matches]
    from app.api.db import TeamMatchStats
    rows = (TeamMatchStats.query
            .filter(TeamMatchStats.match_id.in_(mids)).all())
    per_match: dict = {}
    for r in rows:
        per_match.setdefault(r.match_id, {})[r.side] = r
    mapping = _roll(hist_matches, per_match, window)
    df = pd.DataFrame.from_dict(mapping, orient="index")
    df.index.name = "match_id"
    # 统一列(缺失统计的列以 NaN 填充,模型自动跳过)
    for col in stat_cols:
        if col not in df.columns:
            df[col] = None
    return df[stat_cols]
