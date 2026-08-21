"""Bayesian/Hierarchical Team Strength (: 升班马/新队友加强收缩)。

层次结构: Global → League → Team → Match State
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
新球队/升班马:无样本或极少样本 → 加强收缩到 league prior,不报错。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEAGUE_KAPPA = 15.0 # 球队历史先验收缩强度
RECENT_KAPPA = 8.0 # 近期状态收缩强度
HIST_WINDOW = 1000 # 球队历史先验窗口
RECENT_WINDOW = 200 # 近期窗口
DECAY_HALFLIFE = 200
DECAY_HALFLIFE_DAYS = 400.0 # 时间衰减半衰期(场)


NEW_TEAM_MAX_MATCHES = 20 # 少于等于该场次视为新球队
NEW_TEAM_KAPPA_BOOST = 2.0 # 新球队额外收缩强度倍数


def _league_avg(hist_df: pd.DataFrame) -> float:
 hg = hist_df["home_goals"].to_numpy(dtype=float)
 ag = hist_df["away_goals"].to_numpy(dtype=float)
 hg = hg[~np.isnan(hg)]
 ag = ag[~np.isnan(ag)]
 if len(hg) + len(ag) == 0:
 return 1.5
 return max(float(np.mean(np.concatenate([hg, ag]))), 0.1)


def _is_new_team(hist_df: pd.DataFrame, team: str) -> bool:
 """: 判断是否为升班马/新队友(样本极少)。"""
 if hist_df is None or hist_df.empty:
 return True
 
 mask = (hist_df["home_team"] == team) | (hist_df["away_team"] == team)
 n_matches = mask.sum()
 
 return n_matches <= NEW_TEAM_MAX_MATCHES


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
 # 双重衰减:match-count × calendar
 ages = np.arange(len(idx) - 1, -1, -1, dtype=float)
 w = np.exp(-0.693 * ages / DECAY_HALFLIFE)
 if "date" in rows.columns and rows["date"].notna().all():
 _d = pd.to_datetime(rows["date"])
 days = (_d.iloc[-1] - _d).dt.days.to_numpy(dtype=float)
 w = w * np.exp(-0.693 * days / DECAY_HALFLIFE_DAYS)
 w = w / w.sum()
 scored = np.nansum(w * scored)
 conceded = np.nansum(w * conceded)
 else:
 scored = np.nanmean(scored)
 conceded = np.nanmean(conceded)
 return float(scored), float(conceded), len(idx)


def team_posteriors(
 hist_df: pd.DataFrame,
 team: str,
 side: str,
 kappa_hist: float = LEAGUE_KAPPA,
 kappa_recent: float = RECENT_KAPPA,
) -> tuple[float, float]:
 """球队进攻/防守后验(收缩估计)。

 : 新球队/升班马自动加强收缩(κ * NEW_TEAM_KAPPA_BOOST)。
 """
 if hist_df is None or hist_df.empty:
 return 1.0, 1.0
 
 
 is_new = _is_new_team(hist_df, team)
 if is_new:
 kappa_hist *= NEW_TEAM_KAPPA_BOOST
 kappa_recent *= NEW_TEAM_KAPPA_BOOST
 
 league = _league_avg(hist_df)
 
 # 历史先验(全侧)
 hist_scored, hist_conceded, hist_n = _team_stats(
 hist_df, team, side=None, window=HIST_WINDOW, decay=True
 )
 
 # 近期状态
 recent_scored, recent_conceded, recent_n = _team_stats(
 hist_df, team, side=side, window=RECENT_WINDOW, decay=True
 )
 
 # 收缩估计
 if hist_n > 0 and not np.isnan(hist_scored):
 prior_scored = hist_scored
 prior_conceded = hist_conceded
 else:
 prior_scored = league
 prior_conceded = league
 
 if recent_n > 0 and not np.isnan(recent_scored):
 # 向 prior 收缩
 w_recent = recent_n / (recent_n + kappa_recent)
 att = w_recent * recent_scored + (1 - w_recent) * prior_scored
 deff = w_recent * recent_conceded + (1 - w_recent) * prior_conceded
 else:
 att = prior_scored
 deff = prior_conceded
 
 # 向 league prior 收缩
 w_hist = hist_n / (hist_n + kappa_hist) if hist_n > 0 else 0
 att = w_hist * att + (1 - w_hist) * league
 deff = w_hist * deff + (1 - w_hist) * league
 
 return float(att / league), float(deff / league)


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
 hist_df, home_team, side="home",
 kappa_hist=kappa_hist, kappa_recent=kappa_recent,
 )
 att_a, def_a = team_posteriors(
 hist_df, away_team, side="away",
 kappa_hist=kappa_hist, kappa_recent=kappa_recent,
 )
 lam_h = league * att_h * def_a
 lam_a = league * att_a * def_h
 return float(np.clip(lam_h, 0.05, 6.0)), float(np.clip(lam_a, 0.05, 6.0))


def version() -> str:
 """公式哈希(Bayes 成员纳入版本冻结用)。"""
 import hashlib
 import inspect

 spec = (
 f"LEAGUE_KAPPA={LEAGUE_KAPPA};RECENT_KAPPA={RECENT_KAPPA};"
 f"HIST_WINDOW={HIST_WINDOW};RECENT_WINDOW={RECENT_WINDOW};"
 f"NEW_TEAM_MAX_MATCHES={NEW_TEAM_MAX_MATCHES};"
 f"NEW_TEAM_KAPPA_BOOST={NEW_TEAM_KAPPA_BOOST}"
 )
 impl = inspect.getsource(team_posteriors) + inspect.getsource(bayes_lambda)
 return hashlib.sha256((spec + "|" + impl).encode()).hexdigest()[:12]
