"""06 Opponent Interaction(审查 §14 拆分):H2H 交手胜率(已降权,可选关闭)。"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

FAMILY = "06_opponent_interaction"
DESCRIPTION = "Opponent Interaction(H2H,已降权)"
FEATURES: list[dict] = [{"name": "home_team_h2h_win_rate", "window": "all", "agg": "mean", "source": "matches.same_pair", "policy": "exclude_current_expanding_shift1"}]


def _feature_flag(name: str, default: bool = True) -> bool:
    """读取 configs/models.yaml 的 features 开关(§3 可选关闭)。"""
    try:
        import os as _os

        import yaml as _yaml
        _root = _os.path.dirname(_os.path.dirname(_os.path.dirname(
            _os.path.dirname(_os.path.abspath(__file__)))))
        _path = _os.path.join(_root, "configs", "models.yaml")
        if _os.path.exists(_path):
            with open(_path, encoding="utf-8") as _f:
                _cfg = _yaml.safe_load(_f) or {}
            return bool((_cfg.get("features") or {}).get(name, default))
    except Exception:
        pass
    return default


def compute_h2h(data: pd.DataFrame) -> pd.DataFrame:
    """交手胜率(home 视角,同对阵历史);feature_flags.h2h=false 时完全移除。"""
    prepared = data.copy()
    if not _feature_flag("h2h", True):
        return prepared.drop(columns=["home_team_h2h_win_rate"], errors="ignore")
    if "home_goals" not in prepared.columns or "away_goals" not in prepared.columns:
        return prepared

    df = prepared.reset_index(drop=True).reset_index()
    pair = df[["index", "home_team", "away_team", "home_goals", "away_goals"]].copy()
    pair["pair"] = pair["home_team"] + "||" + pair["away_team"]
    if "date" in df.columns:
        pair["date"] = df["date"].values

    pair["h2h_home_win_rate"] = np.where(
        pair["home_goals"].isna(), np.nan,
        (pair["home_goals"] > pair["away_goals"]).astype(float))
    pair = pair.sort_values(["pair", "date", "index"], kind="mergesort") \
        if "date" in pair.columns else pair.sort_values(["pair", "index"], kind="mergesort")
    pair["cum"] = pair.groupby("pair")["h2h_home_win_rate"].expanding().mean().reset_index(drop=True)
    pair["prior"] = pair.groupby("pair")["cum"].shift(1)
    pair = pair.set_index("index")

    prepared = prepared.reset_index(drop=True)
    prepared["home_team_h2h_win_rate"] = pair["prior"].reindex(range(len(prepared))).values
    return prepared


def compute(df: pd.DataFrame, league_type: str | None = None) -> pd.DataFrame:
    """计算本家族特征并附加到 df(由 factory 调度)。"""
    return df.copy()


def version() -> str:
    """公式哈希(审查 §16):规格 JSON + 实现代码哈希。"""
    import inspect
    spec = json.dumps(sorted(FEATURES, key=lambda f: f["name"]),
                      ensure_ascii=False, sort_keys=True)
    impl = inspect.getsource(compute_h2h)
    return hashlib.sha256((spec + "|" + impl).encode()).hexdigest()[:12]
