"""Bayesian/Hierarchical Team Strength(审查九 P1-10)。

经验贝叶斯收缩(无 MCMC 依赖,解析式):
  θ_team = (n/(n+κ))·obs + (κ/(n+κ))·prior

层次结构:global(联赛均值)→ league prior → team posterior:
  - league prior:联赛场均进球/失球(数据驱动)
  - team obs:该队最近 window 场场均进球/失球
  - κ:收缩强度(样本少 → 强收缩向联赛;样本多 → 信任球队自身)

输出 bayes λ(作为 Ensemble 新成员 "bayes"):
  λ_home = league_avg · attack_posterior_home · defense_posterior_away
  λ_away = league_avg · attack_posterior_away · defense_posterior_home

时间安全:只使用截止该场(严格早于)的历史。
新球队/升班马:无样本 → 完全收缩到 league prior(global→league),
不再"直接报错"(审查二十八)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def team_posteriors(hist_df: pd.DataFrame, team: str,
                    kappa: float = 8.0, window: int = 200):
    """该队最近 window 场(严格早于预测日的 hist)的经验贝叶斯强度。

    返回 (attack_posterior, defense_posterior) —— 相对 1 的系数:
      attack_posterior = θ_attack / league_avg_goals
      defense_posterior = θ_defense / league_avg_goals(对手视角:值>1 防守差)
    无样本 → (1.0, 1.0)(完全收缩到 league prior)。
    """
    if hist_df is None or hist_df.empty:
        return 1.0, 1.0
    hg = hist_df["home_goals"].to_numpy(dtype=float)
    ag = hist_df["away_goals"].to_numpy(dtype=float)
    hg = hg[~np.isnan(hg)]
    ag = ag[~np.isnan(ag)]
    if len(hg) == 0 and len(ag) == 0:
        return 1.0, 1.0
    league_avg = float(np.mean(np.concatenate([hg, ag]))) if len(hg) + len(ag) else 1.5
    league_avg = max(league_avg, 0.1)
    # 该队最近 window 场(按行序 = 时间序)
    mask = (hist_df["home_team"] == team) | (hist_df["away_team"] == team)
    idx = np.where(mask.to_numpy())[0]
    if len(idx) == 0:
        return 1.0, 1.0
    idx = idx[-window:]
    rows = hist_df.iloc[idx]
    gf = rows["home_goals"].to_numpy(dtype=float)
    ga = rows["away_goals"].to_numpy(dtype=float)
    scored = np.where(rows["home_team"].to_numpy() == team, gf, ga)
    conceded = np.where(rows["home_team"].to_numpy() == team, ga, gf)
    scored = scored[~np.isnan(scored)]
    conceded = conceded[~np.isnan(conceded)]
    n = max(len(scored), 1)
    obs_att = float(np.mean(scored)) if len(scored) else league_avg
    obs_def = float(np.mean(conceded)) if len(conceded) else league_avg
    # 收缩(样本多 → 信任球队;样本少 → 向联赛先验)
    w = n / (n + kappa)
    att = w * obs_att + (1 - w) * league_avg
    deff = w * obs_def + (1 - w) * league_avg
    return att / league_avg, deff / league_avg


def bayes_lambda(hist_df: pd.DataFrame, home_team: str, away_team: str,
                 kappa: float = 8.0, window: int = 200) -> tuple[float, float]:
    """层次贝叶斯 λ(联赛先验 × 主客队收缩后攻防)。"""
    if hist_df is None or hist_df.empty:
        return 1.5, 1.4
    hg = hist_df["home_goals"].to_numpy(dtype=float)
    ag = hist_df["away_goals"].to_numpy(dtype=float)
    hg = hg[~np.isnan(hg)]
    ag = ag[~np.isnan(ag)]
    if len(hg) + len(ag) == 0:
        return 1.5, 1.4
    league_avg = float(np.mean(np.concatenate([hg, ag])))
    league_avg = max(league_avg, 0.1)
    att_h, def_h = team_posteriors(hist_df, home_team, kappa, window)
    att_a, def_a = team_posteriors(hist_df, away_team, kappa, window)
    lam_h = league_avg * att_h * def_a
    lam_a = league_avg * att_a * def_h
    return float(np.clip(lam_h, 0.05, 6.0)), float(np.clip(lam_a, 0.05, 6.0))
