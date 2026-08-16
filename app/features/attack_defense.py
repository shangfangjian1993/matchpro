"""02 Attack/Defense(审查 §14 拆分):进失球/xG/射门滚动特征实现。

规格声明(审查 §16:formula_hash = 规格 + 实现代码哈希)。
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from app.features.rolling import expanding_prior, valid_rolling

FAMILY = "02_attack_defense"
DESCRIPTION = "Attack/Defense(进失球/xG/射门)"
FEATURES: list[dict] = [{"name": "home_team_goals_avg", "window": "all", "agg": "mean", "source": "matches.home_goals", "policy": "exclude_current_expanding_shift1"}, {"name": "away_team_goals_avg", "window": "all", "agg": "mean", "source": "matches.home_goals", "policy": "exclude_current_expanding_shift1"}, {"name": "home_team_conceded_avg", "window": "all", "agg": "mean", "source": "matches.away_goals", "policy": "exclude_current_expanding_shift1"}, {"name": "away_team_conceded_avg", "window": "all", "agg": "mean", "source": "matches.away_goals", "policy": "exclude_current_expanding_shift1"}, {"name": "home_team_xg_avg", "window": "all", "agg": "mean", "source": "matches.home_xg", "policy": "exclude_current_expanding_shift1"}, {"name": "away_team_xg_avg", "window": "all", "agg": "mean", "source": "matches.away_xg", "policy": "exclude_current_expanding_shift1"}, {"name": "home_team_xg_recent", "window": "5", "agg": "mean", "source": "matches.home_xg", "policy": "exclude_current_valid_rolling"}, {"name": "away_team_xg_recent", "window": "5", "agg": "mean", "source": "matches.away_xg", "policy": "exclude_current_valid_rolling"}, {"name": "home_team_shots_avg", "window": "all", "agg": "mean", "source": "matches.home_shots", "policy": "exclude_current_expanding_shift1"}, {"name": "away_team_shots_avg", "window": "all", "agg": "mean", "source": "matches.away_shots", "policy": "exclude_current_expanding_shift1"}, {"name": "home_team_shots_recent", "window": "5", "agg": "mean", "source": "matches.home_shots", "policy": "exclude_current_valid_rolling"}, {"name": "away_team_shots_recent", "window": "5", "agg": "mean", "source": "matches.away_shots", "policy": "exclude_current_valid_rolling"}, {"name": "home_team_corners_avg", "window": "all", "agg": "mean", "source": "matches.home_corners", "policy": "exclude_current_expanding_shift1"}, {"name": "away_team_corners_avg", "window": "all", "agg": "mean", "source": "matches.away_corners", "policy": "exclude_current_expanding_shift1"}]


def compute_attack_defense(prepared: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    """进失球特征:生涯均值(expanding)+ 近 5 场均值。"""
    long["gf_avg_prior"] = expanding_prior(long, "gf", "gf_avg_prior")
    long["ga_avg_prior"] = expanding_prior(long, "ga", "ga_avg_prior")
    long["recent_goals"] = valid_rolling(long, "gf", 5, "mean")
    long["recent_conceded"] = valid_rolling(long, "ga", 5, "mean")
    long["recent_gd"] = valid_rolling(long, "gd", 5, "mean")
    long["side_home_goals"] = valid_rolling(long, "gf", 5, "mean", side="home")
    long["side_away_goals"] = valid_rolling(long, "gf", 5, "mean", side="away")
    long["side_home_conceded"] = valid_rolling(long, "ga", 5, "mean", side="home")
    long["side_away_conceded"] = valid_rolling(long, "ga", 5, "mean", side="away")

    home = long[long["side"] == "home"].set_index("index")
    away = long[long["side"] == "away"].set_index("index")
    prepared = prepared.reset_index(drop=True)
    prepared["home_team_goals_avg"] = home["gf_avg_prior"]
    prepared["away_team_goals_avg"] = away["gf_avg_prior"]
    prepared["home_team_conceded_avg"] = home["ga_avg_prior"]
    prepared["away_team_conceded_avg"] = away["ga_avg_prior"]
    prepared["home_team_recent_goals"] = home["recent_goals"]
    prepared["away_team_recent_goals"] = away["recent_goals"]
    prepared["home_team_recent_conceded"] = home["recent_conceded"]
    prepared["away_team_recent_conceded"] = away["recent_conceded"]
    prepared["home_team_recent_gd"] = home["recent_gd"]
    prepared["away_team_recent_gd"] = away["recent_gd"]
    prepared["home_team_home_goals_avg"] = home["side_home_goals"]
    prepared["away_team_away_goals_avg"] = away["side_away_goals"]
    prepared["home_team_home_conceded_avg"] = home["side_home_conceded"]
    prepared["away_team_away_conceded_avg"] = away["side_away_conceded"]
    return prepared


def compute_side_metric_rolling(data: pd.DataFrame, metric_base: str, prefix: str) -> pd.DataFrame:
    """主客分列指标(home_xg/away_xg)滚动均值(expanding + 近5场),不含本场。"""
    home_col, away_col = f"home_{metric_base}", f"away_{metric_base}"
    prepared = data.copy()
    if home_col not in prepared.columns or away_col not in prepared.columns:
        return prepared

    df = prepared.reset_index(drop=True).reset_index()
    parts = []
    for side, team_col, val_col in (("home", "home_team", home_col),
                                    ("away", "away_team", away_col)):
        part = df[["index", team_col, val_col]].copy()
        part["side"] = side
        part = part.rename(columns={team_col: "team", val_col: "val"})
        if "date" in df.columns:
            part["date"] = df["date"].values
        parts.append(part)

    long = pd.concat(parts, ignore_index=True)
    long["row_id"] = np.arange(len(long))
    if "date" in long.columns:
        long = long.sort_values(["date", "index"], kind="mergesort")
    else:
        long = long.sort_values("index", kind="mergesort")

    long["val_prior"] = expanding_prior(long, "val", "val_prior")
    long["val_recent"] = valid_rolling(long, "val", 5, "mean")

    home = long[long["side"] == "home"].set_index("index")
    away = long[long["side"] == "away"].set_index("index")
    prepared = prepared.reset_index(drop=True)
    prepared[f"home_team_{prefix}_avg"] = home["val_prior"]
    prepared[f"away_team_{prefix}_avg"] = away["val_prior"]
    prepared[f"home_team_{prefix}_recent"] = home["val_recent"]
    prepared[f"away_team_{prefix}_recent"] = away["val_recent"]
    return prepared


def compute_metric_rolling(data: pd.DataFrame, metric: str, prefix: str) -> pd.DataFrame:
    """单列指标滚动均值(不含本场);丢弃当场统计源列(赛后可知,防泄漏)。"""
    prepared = data.copy()
    if metric not in prepared.columns:
        return prepared

    df = prepared.reset_index(drop=True).reset_index()
    parts = []
    for side, team_col in (("home", "home_team"), ("away", "away_team")):
        part = df[["index", team_col, metric]].copy()
        part["side"] = side
        part = part.rename(columns={team_col: "team", metric: "val"})
        if "date" in df.columns:
            part["date"] = df["date"].values
        parts.append(part)

    long = pd.concat(parts, ignore_index=True)
    long["row_id"] = np.arange(len(long))
    if "date" in long.columns:
        long = long.sort_values(["date", "index"], kind="mergesort")
    else:
        long = long.sort_values("index", kind="mergesort")

    long["val_prior"] = expanding_prior(long, "val", "val_prior")

    for side in ("home", "away"):
        sub = long[long["side"] == side].set_index("index")
        prepared[f"{side}_team_{prefix}_avg"] = sub["val_prior"]

    return prepared.drop(columns=[metric], errors="ignore")


def compute(df: pd.DataFrame, league_type: str | None = None) -> pd.DataFrame:
    """计算本家族特征并附加到 df(由 factory 调度,基于 long 表)。"""
    return df.copy()


def version() -> str:
    """公式哈希(审查 §16):规格 JSON + 实现代码哈希。"""
    import inspect
    spec = json.dumps(sorted(FEATURES, key=lambda f: f["name"]),
                      ensure_ascii=False, sort_keys=True)
    impl = inspect.getsource(compute_attack_defense)
    return hashlib.sha256((spec + "|" + impl).encode()).hexdigest()[:12]
