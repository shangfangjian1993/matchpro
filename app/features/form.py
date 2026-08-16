"""03 Form & Momentum(审查 §14 拆分):胜率/近期状态/主客分离实现。"""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from app.features.rolling import expanding_prior, valid_rolling

FAMILY = "03_form_momentum"
DESCRIPTION = "Form & Momentum(近期/主客/胜率)"
FEATURES: list[dict] = [{"name": "home_team_win_rate", "window": "all", "agg": "mean", "source": "matches.result", "policy": "exclude_current_expanding_shift1"}, {"name": "away_team_win_rate", "window": "all", "agg": "mean", "source": "matches.result", "policy": "exclude_current_expanding_shift1"}, {"name": "home_team_form", "window": "5", "agg": "sum_norm15", "source": "matches.points", "policy": "exclude_current_valid_rolling"}, {"name": "away_team_form", "window": "5", "agg": "sum_norm15", "source": "matches.points", "policy": "exclude_current_valid_rolling"}, {"name": "home_team_home_form", "window": "5", "agg": "sum_norm15", "source": "matches.points_home_only", "policy": "exclude_current_valid_rolling"}, {"name": "away_team_away_form", "window": "5", "agg": "sum_norm15", "source": "matches.points_away_only", "policy": "exclude_current_valid_rolling"}, {"name": "home_team_home_goals_avg", "window": "5", "agg": "mean", "source": "matches.home_goals_home_only", "policy": "exclude_current_valid_rolling"}, {"name": "away_team_away_goals_avg", "window": "5", "agg": "mean", "source": "matches.home_goals_away_only", "policy": "exclude_current_valid_rolling"}]


def compute_form(prepared: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    """胜率(expanding)+ 近5场积分归一(form)+ 主客分离 form。"""
    long["win_rate_prior"] = expanding_prior(long, "win", "win_rate_prior")
    long["form_prior"] = valid_rolling(long, "points", 5, "sum") / 15.0
    long["side_home_form"] = valid_rolling(long, "points", 5, "sum", side="home") / 15.0
    long["side_away_form"] = valid_rolling(long, "points", 5, "sum", side="away") / 15.0

    home = long[long["side"] == "home"].set_index("index")
    away = long[long["side"] == "away"].set_index("index")
    prepared = prepared.reset_index(drop=True)
    prepared["home_team_win_rate"] = home["win_rate_prior"]
    prepared["away_team_win_rate"] = away["win_rate_prior"]
    prepared["home_team_form"] = home["form_prior"]
    prepared["away_team_form"] = away["form_prior"]
    prepared["home_team_home_form"] = home["side_home_form"]
    prepared["away_team_away_form"] = away["side_away_form"]
    return prepared


def compute(df: pd.DataFrame, league_type: str | None = None) -> pd.DataFrame:
    """计算本家族特征并附加到 df(由 factory 调度)。"""
    return df.copy()


def version() -> str:
    """公式哈希(审查 §16):规格 JSON + 实现代码哈希。"""
    import inspect
    spec = json.dumps(sorted(FEATURES, key=lambda f: f["name"]),
                      ensure_ascii=False, sort_keys=True)
    impl = inspect.getsource(compute_form)
    return hashlib.sha256((spec + "|" + impl).encode()).hexdigest()[:12]
