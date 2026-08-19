"""Bayesian/Hierarchical Team Strength(审查九 P1-10 + 深化:两阶段收缩)。

层次结构(审查二十八:Global → League → Team → Match State):
  league prior(联赛均值)
      ↓
  team historical prior(球队全部历史 → 收缩到 league,κ1=15)
      ↓
  recent state(近期 window 场,指数时间衰减,半衰期 200 场 → 收缩到
              team prior,κ2=8)
      ↓
  attack/defense posterior

主客场分离:进攻按主/客场侧估计(主场进攻 ≠ 客场进攻);
防守用全侧估计更稳。

时间安全:只用截止该场(严格早于)的历史。
新球队:无样本 → 完全收缩到 league prior(global→league),不报错。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LEAGUE_KAPPA = 15.0  # 球队历史先验收缩强度
RECENT_KAPPA = 8.0  # 近期状态收缩强度
HIST_WINDOW = 1000  # 球队历史先验窗口
RECENT_WINDOW = 200  # 近期窗口
DECAY_HALFLIFE = 200.0  # 时间衰减半衰期(场)


def _league_avg(hist_df: pd.DataFrame) -> float:
    hg = hist_df["home_goals"].to_numpy(dtype=float)
    ag = hist_df["away_goals"].to_numpy(dtype=float)
    hg = hg[~np.isnan(hg)]
    ag = ag[~np.isnan(ag)]
    if len(hg) + len(ag) == 0:
        return 1.5
    return max(float(np.mean(np.concatenate([hg, ag]))), 0.1)


def _team_stats(
    hist_df: pd.DataFrame,
    team: str,
    side=None,
    window: int = RECENT_WINDOW,
    decay: bool = True,
) -> tuple[float, float, int]:
    """该队(主/客/全侧)最近 window 场的加权场均进球/失球。

    side: "home" | "away" | None(全侧)。decay: 指数时间衰减。
    返回 (scored, conceded, n)。
    """
    mask = (hist_df["home_team"] == team) | (hist_df["away_team"] == team)
    if side == "home":
        mask = hist_df["home_team"] == team
    elif side == "away":
        mask = hist_df["away_team"] == team
    idx = np.where(mask.to_numpy())[0]
    if len(idx) == 0:
        return np.nan, np.nan, 0
    idx = idx[-window:]
    rows = hist_df.iloc[idx]
    gf = rows["home_goals"].to_numpy(dtype=float)
    ga = rows["away_goals"].to_numpy(dtype=float)
    is_home = rows["home_team"].to_numpy() == team
    scored = np.where(is_home, gf, ga)
    conceded = np.where(is_home, ga, gf)
    if decay and len(idx) > 1:
        # 时间衰减权重:最近场权重最高(半衰期 DECAY_HALFLIFE 场)
        ages = np.arange(len(idx) - 1, -1, -1, dtype=float)
        w = np.exp(-0.693 * ages / DECAY_HALFLIFE)
        w = w / w.sum()
        scored = np.nansum(scored * w) if not np.isnan(scored).all() else np.nan
        conceded = np.nansum(conceded * w) if not np.isnan(conceded).all() else np.nan
    else:
        scored = float(np.nanmean(scored)) if not np.isnan(scored).all() else np.nan
        conceded = (
            float(np.nanmean(conceded)) if not np.isnan(conceded).all() else np.nan
        )
    return scored, conceded, len(idx)


def team_posteriors(
    hist_df: pd.DataFrame,
    team: str,
    side: str | None = None,
    kappa_hist: float = LEAGUE_KAPPA,
    kappa_recent: float = RECENT_KAPPA,
) -> tuple[float, float]:
    """两阶段经验贝叶斯收缩 → (attack_posterior, defense_posterior)(相对 1)。

    阶段1:球队历史先验(全历史 → league);
    阶段2:近期状态(指数衰减 → 球队历史先验)。
    无样本 → (1.0, 1.0)(完全收缩到 league)。
    """
    if hist_df is None or hist_df.empty:
        return 1.0, 1.0
    league = _league_avg(hist_df)
    # 阶段1:球队历史先验(全侧,大窗口)
    hs, hc, n1 = _team_stats(hist_df, team, side=None, window=HIST_WINDOW, decay=False)
    if n1 == 0:
        return 1.0, 1.0
    w1 = n1 / (n1 + kappa_hist)
    prior_att = w1 * hs + (1 - w1) * league
    prior_def = w1 * hc + (1 - w1) * league
    # 阶段2:近期状态(按 side 侧,指数衰减)收缩到球队先验
    rs, rc, n2 = _team_stats(hist_df, team, side=side, window=RECENT_WINDOW, decay=True)
    if n2 == 0 or np.isnan(rs):
        return prior_att / league, prior_def / league
    w2 = n2 / (n2 + kappa_recent)
    post_att = w2 * rs + (1 - w2) * prior_att
    post_def = w2 * rc + (1 - w2) * prior_def
    return post_att / league, post_def / league


def bayes_lambda(
    hist_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    kappa_hist: float = LEAGUE_KAPPA,
    kappa_recent: float = RECENT_KAPPA,
) -> tuple[float, float]:
    """层次贝叶斯 λ:主队用主场进攻侧、客队用客场进攻侧;防守全侧。"""
    if hist_df is None or hist_df.empty:
        return 1.5, 1.4
    league = _league_avg(hist_df)
    att_h, def_h = team_posteriors(
        hist_df,
        home_team,
        side="home",
        kappa_hist=kappa_hist,
        kappa_recent=kappa_recent,
    )
    att_a, def_a = team_posteriors(
        hist_df,
        away_team,
        side="away",
        kappa_hist=kappa_hist,
        kappa_recent=kappa_recent,
    )
    lam_h = league * att_h * def_a
    lam_a = league * att_a * def_h
    return float(np.clip(lam_h, 0.05, 6.0)), float(np.clip(lam_a, 0.05, 6.0))


def version() -> str:
    """公式哈希(审查 A70A601 P1-4:Bayes 成员纳入版本冻结用)。

    规格 = 超参常量(κ₁/κ₂/窗口);实现 = 两阶段收缩 + bayes_lambda 源码。
    任何常数/公式变化 → version 变化 → 快照 model_set 变化。
    """
    import hashlib
    import inspect

    spec = (
        f"LEAGUE_KAPPA={LEAGUE_KAPPA};RECENT_KAPPA={RECENT_KAPPA};"
        f"HIST_WINDOW={HIST_WINDOW};RECENT_WINDOW={RECENT_WINDOW}"
    )
    impl = inspect.getsource(team_posteriors) + inspect.getsource(bayes_lambda)
    return hashlib.sha256((spec + "|" + impl).encode()).hexdigest()[:12]
