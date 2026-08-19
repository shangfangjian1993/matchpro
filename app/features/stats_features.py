"""Feature · stats(team_match_stats 深度比赛统计 → 球队状态特征)。

审查 21cd12b 修复:
- P0-1:rolling_team_stats 不再用手工白名单切列 —— 按 home_tms_/away_tms_
  前缀动态输出 _roll 的全部新列(avg_3/5/10、ewm、rel、group),杜绝
  "内部有新特征、公开接口被裁掉"。
- P1-3:仅保留 windows: tuple[int, ...](移除冗余 window 参数)。
- P1-自引用 import 已清(_roll 直接用模块顶层 _AGG_COLS/_GROUPS)。
- EWMA 语义明确:最近主窗口场上的指数加权(_ewma(hs[-w0:], alpha)),
  非"全历史 EWMA"。
- 对手差列命名为 _rel(relative_to_opponent):本侧均值 − 对手该侧均值,
  避免与"联赛基准对手调整强度"混淆。

Odds 状态:match_odds(收盘赔率)于赛后收盘,赛前不可得 → 不入赛前特征;
仅用于回测评估 / 后处理对比(本模块不实现)。
"""

from __future__ import annotations

import collections
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# team_match_stats 深度统计 → 特征列基础名(avg/ewm/rel/group 由其派生)
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

# 特征族分组(审查:Volume / Defensive / Chance / Discipline)
_GROUPS = {
    "grp_volume": ["shots_inside_box", "shots_outside_box", "tackles", "clearances"],
    "grp_defensive": ["interceptions", "blocked_shots", "total_saves", "fouls"],
    "grp_chance": ["big_chances"],
    "grp_discipline": ["offsides"],
}

# 该族所需最少历史场数(用于 hist_limit 校验/文档)
REQUIRED_HISTORY = 10


def _ewma(vals, alpha):
    """最近序列的指数加权均值(权重:最近最大,alpha 衰减)。空→None。"""
    if not vals:
        return None
    n = len(vals)
    w = [(1 - alpha) ** (n - 1 - k) for k in range(n)]
    ws = sum(w)
    return round(sum(v * wi for v, wi in zip(vals, w)) / ws, 3) if ws else None


def _roll(hist_matches, per_match: dict, windows: tuple = (5,)) -> dict:
    """纯计算:按 球队×主客侧 维护状态;输出 {match_id: {...特征列}}。

    per_match: {match_id: {"home": stats_obj_or_none, "away": ...}}
    输出列(全部以 home_tms_/away_tms_ 开头,供公开接口按前缀输出):
      *_avg_{w}         窗口均值(每个 windows 内窗口,side 主/客侧)
      *_overall_{w}     窗口均值(overall 全历史,side 并行)
      *_ewm        最近主窗口场的指数加权均值
      *_rel        相对对手(本侧均值 − 对手该侧均值)
      *_grp_*_avg  分组聚合均值
    """
    _w = list(windows) if isinstance(windows, (tuple, list)) else [windows]
    _w = _w or [5]
    team_state = collections.defaultdict(
        lambda: {
            "home": collections.defaultdict(list),
            "away": collections.defaultdict(list),
            # 审查 A70A601 P1-8:overall(全部比赛)滚动 —— side-only 历史在
            # "近 10 场含 5 主"等场景只取 5 场信息,量不足,需并行 overall。
            "all": collections.defaultdict(list),
        }
    )
    out: dict = {}
    w0 = max(_w)
    alpha = 2.0 / (w0 + 1)
    for m in hist_matches:
        if m is None:
            out[None] = {}  # 无落库 Match(如新预测场):stats 全 None,模型跳过
            continue
        st = per_match.get(m.id, {})
        home, away = m.home_team, m.away_team
        rec: dict = {}
        for col, base in _AGG_COLS.items():
            hs = team_state[home]["home"][col]
            as_ = team_state[away]["away"][col]
            ha = team_state[home]["all"][col]
            aa = team_state[away]["all"][col]
            for w in _w:
                hw = hs[-w:]
                aw = as_[-w:]
                haw = ha[-w:]
                aaw = aa[-w:]
                rec[f"home_{base}_avg_{w}"] = (
                    round(sum(hw) / len(hw), 3) if hw else None
                )
                rec[f"away_{base}_avg_{w}"] = (
                    round(sum(aw) / len(aw), 3) if aw else None
                )
                rec[f"home_{base}_overall_{w}"] = (
                    round(sum(haw) / len(haw), 3) if haw else None
                )
                rec[f"away_{base}_overall_{w}"] = (
                    round(sum(aaw) / len(aaw), 3) if aaw else None
                )
            # EWMA:最近主窗口场的加权(方案 B,语义明确)
            hw0, aw0 = hs[-w0:], as_[-w0:]
            rec[f"home_{base}_ewm"] = _ewma(hw0, alpha)
            rec[f"away_{base}_ewm"] = _ewma(aw0, alpha)
            # 相对对手(_rel):本侧均值 − 对手该侧均值
            h_mean = sum(hw0) / len(hw0) if hw0 else None
            a_mean = sum(aw0) / len(aw0) if aw0 else None
            if h_mean is not None and a_mean is not None:
                rec[f"home_{base}_rel"] = round(h_mean - a_mean, 3)
                rec[f"away_{base}_rel"] = round(a_mean - h_mean, 3)
        for grp, cols in _GROUPS.items():
            h_vals, a_vals = [], []
            for col in cols:
                hvw = team_state[home]["home"][col][-w0:]
                avw = team_state[away]["away"][col][-w0:]
                if hvw:
                    h_vals.append(sum(hvw) / len(hvw))
                if avw:
                    a_vals.append(sum(avw) / len(avw))
            rec[f"home_tms_{grp}_avg"] = (
                round(sum(h_vals) / len(h_vals), 3) if h_vals else None
            )
            rec[f"away_tms_{grp}_avg"] = (
                round(sum(a_vals) / len(a_vals), 3) if a_vals else None
            )
        out[m.id] = rec
        hst = st.get("home")
        ast = st.get("away")
        if hst is not None:
            for col in _AGG_COLS:
                v = getattr(hst, col, None)
                if v is not None:
                    _v = float(v)
                    team_state[home]["home"][col].append(_v)
                    team_state[home]["all"][col].append(_v)
        if ast is not None:
            for col in _AGG_COLS:
                v = getattr(ast, col, None)
                if v is not None:
                    _v = float(v)
                    team_state[away]["away"][col].append(_v)
                    team_state[away]["all"][col].append(_v)
    return out


def rolling_team_stats(hist_matches, windows: tuple = (5,)) -> pd.DataFrame:
    """公开 API:球队状态滚动特征(index = match_id)。

    - hist_matches 为空 → 空 DataFrame(调用方留 NaN,可降级)。
    - 实现/schema 异常 → 抛出(不做静默降级)。
    - 输出列 = 按 home_tms_/away_tms_ 前缀动态选出(新增特征自动带上,
      不再维护手工白名单 —— 防 schema drift)。
    """
    if not hist_matches:
        return pd.DataFrame()
    mids = [m.id if m is not None else None for m in hist_matches]
    from app.api.db import TeamMatchStats

    rows = TeamMatchStats.query.filter(TeamMatchStats.match_id.in_(mids)).all()
    per_match: dict = {}
    for r in rows:
        per_match.setdefault(r.match_id, {})[r.side] = r
    mapping = _roll(hist_matches, per_match, windows=windows)
    df = pd.DataFrame.from_dict(mapping, orient="index")
    df.index.name = "match_id"
    feature_cols = [c for c in df.columns if c.startswith(("home_tms_", "away_tms_"))]
    return df[feature_cols]


def version() -> str:
    """公式哈希(与 strength 同契约):规格 + 实现源码哈希。

    纳入 app.features.registry.logical_version —— stats 实现变化必须
    触发 feature version 变化(审查 21cd12b P1-2)。
    """
    import hashlib
    import inspect

    spec = (
        "_AGG_COLS="
        + repr(sorted(_AGG_COLS))
        + ";_GROUPS="
        + repr({k: sorted(v) for k, v in _GROUPS.items()})
    )
    impl = inspect.getsource(_roll) + inspect.getsource(rolling_team_stats)
    return hashlib.sha256((spec + "|" + impl).encode()).hexdigest()[:12]
