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

# team_match_stats 深度统计 → 特征列基础名(窗口均值/EWMA/对手调整均由其派生)
_AGG_COLS = {
    "fouls": "tms_fouls",
    "offsides": "tms_offsides",
    "tackles": "tms_tackles",
    "interceptions": "tms_interceptions",
    "clearances": "tms_clearances",
    "blocked_shots": "tms_blocked_shots",
    "big_chances": "tms_big_chances",
    "total_saves": "tms_total_saves",
    "shots_inside_box": "tms_shots_inside_box",
    "shots_outside_box": "tms_shots_outside_box",
}

# 特征族分组(审查下一阶段:Volume / Defensive / Chance / Discipline)
_GROUPS = {
    "grp_volume": ["shots_inside_box", "shots_outside_box", "tackles", "clearances"],
    "grp_defensive": ["interceptions", "blocked_shots", "total_saves", "fouls"],
    "grp_chance": ["big_chances"],
    "grp_discipline": ["offsides"],
}


def _ewma(vals, alpha):
    """加权均值:最近权重大(审查:EWMA)。alpha = 2/(window+1)。"""
    if not vals:
        return None
    n = len(vals)
    w = [(1 - alpha) ** (n - 1 - k) for k in range(n)]
    ws = sum(w)
    return round(sum(v * wi for v, wi in zip(vals, w)) / ws, 3) if ws else None


def _roll(hist_matches, per_match: dict, window: int | int | list = 5,
          windows: tuple = (5,)) -> dict:
    """纯计算:按 球队×主客侧 维护状态;输出 {match_id: {...特征列}}。

    per_match: {match_id: {"home": stats_obj_or_none, "away": ...}}
    输出列:
      home_tms_{stat}_avg / away_tms_{stat}_avg   (窗口均值,默认 windows 内)
      home_tms_{stat}_ewm / away_tms_{stat}_ewm   (指数加权,半衰期≈窗口)
      home_tms_{stat}_opp / away_tms_{stat}_opp   (对手调整 = 该侧均值 − 对手均值)
      home_tms_{grp}_avg / away_tms_{grp}_avg     (分组聚合均值)
    """
    from app.features.stats_features import _AGG_COLS, _GROUPS
    _w = list(windows) if isinstance(windows, (tuple, list)) else [windows]
    _w = _w or [5]
    team_state = collections.defaultdict(
        lambda: {"home": collections.defaultdict(list),
                 "away": collections.defaultdict(list)})
    out: dict = {}
    for m in hist_matches:
        st = per_match.get(m.id, {})
        home, away = m.home_team, m.away_team
        # 对手侧状态(用于 opp 调整)
        rec: dict = {}
        for col, base in _AGG_COLS.items():
            hs = team_state[home]["home"][col]
            as_ = team_state[away]["away"][col]
            # 窗口均值(windows 内各窗口)
            for w in _w:
                hw = hs[-w:] if w else hs
                aw = as_[-w:] if w else as_
                rec[f"home_{base}_avg_{w}"] = (
                    round(sum(hw) / len(hw), 3) if hw else None)
                rec[f"away_{base}_avg_{w}"] = (
                    round(sum(aw) / len(aw), 3) if aw else None)
            # EWMA(主窗口)
            alpha = 2.0 / (max(_w) + 1)
            rec[f"home_{base}_ewm"] = _ewma(hs, alpha)
            rec[f"away_{base}_ewm"] = _ewma(as_, alpha)
            # 对手调整(主窗口):本侧均值 − 对手侧均值(同窗)
            w0 = max(_w)
            hw0 = hs[-w0:]; aw0 = as_[-w0:]
            h_mean = sum(hw0) / len(hw0) if hw0 else None
            a_mean = sum(aw0) / len(aw0) if aw0 else None
            rec[f"home_{base}_opp"] = (
                round(h_mean - a_mean, 3) if h_mean is not None and a_mean is not None else None)
            rec[f"away_{base}_opp"] = (
                round(a_mean - h_mean, 3) if h_mean is not None and a_mean is not None else None)
        # 分组聚合均值(主窗口)
        w0 = max(_w)
        for grp, cols in _GROUPS.items():
            h_vals = []
            a_vals = []
            for col in cols:
                hvw = team_state[home]["home"][col][-w0:]
                avw = team_state[away]["away"][col][-w0:]
                if hvw:
                    h_vals.append(sum(hvw) / len(hvw))
                if avw:
                    a_vals.append(sum(avw) / len(avw))
            rec[f"home_tms_{grp}_avg"] = round(sum(h_vals) / len(h_vals), 3) if h_vals else None
            rec[f"away_tms_{grp}_avg"] = round(sum(a_vals) / len(a_vals), 3) if a_vals else None
        out[m.id] = rec
        # ── 本场计入状态(供下一场)──
        hst = st.get("home")
        ast = st.get("away")
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


def rolling_team_stats(hist_matches, window: int = 5, windows: tuple = (5,)) -> pd.DataFrame:
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
    mapping = _roll(hist_matches, per_match, window, windows=windows)
    df = pd.DataFrame.from_dict(mapping, orient="index")
    df.index.name = "match_id"
    # 统一列(缺失统计的列以 NaN 填充,模型自动跳过)
    for col in stat_cols:
        if col not in df.columns:
            df[col] = None
    return df[stat_cols]
