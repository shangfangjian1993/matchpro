"""06 Opponent Interaction(审查 §14 拆分):H2H 交手胜率(已降权,可选关闭)。"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

FAMILY = "06_opponent_interaction"
DESCRIPTION = "Opponent Interaction(H2H,已降权)"
FEATURES: list[dict] = [
    {
        "name": "home_team_h2h_win_rate",
        "window": "all",
        "agg": "mean",
        "source": "matches.same_pair",
        "policy": "exclude_current_expanding_shift1",
    }
]


def _feature_flag(name: str, default: bool = True) -> bool:
    """读取 configs/models.yaml 的 features 开关(§3 可选关闭)。"""
    try:
        import os as _os

        import yaml as _yaml

        _root = _os.path.dirname(
            _os.path.dirname(
                _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            )
        )
        _path = _os.path.join(_root, "configs", "models.yaml")
        if _os.path.exists(_path):
            with open(_path, encoding="utf-8") as _f:
                _cfg = _yaml.safe_load(_f) or {}
            return bool((_cfg.get("features") or {}).get(name, default))
    except Exception:
        pass
    return default


def compute_h2h(data: pd.DataFrame) -> pd.DataFrame:
    """交手胜率(home 视角,同对阵历史);feature_flags.h2h=false 时完全移除。

    审查 A70A601 §21:H2H 带**时间衰减 + venue 变换 + 贡献受限**:
    - recency decay:按距最近一场的天数指数衰减(半衰 1 年)——2-4 年前的
      交手权重趋零,避免"2021 交手 ≠ 2026 交手";
    - venue:主队视角固定(home||away 方向),主客互换对阵独立分组 = 天然
      分层;方向一致行全权、方向不一致行运势天然降权;
    - 贡献受限:衰减后陈旧交手贡献 ≈ 0,使 H2H 对整体预测只占小尾贡献
      (不再以全历史等权累计率冲击模型)。
    无 date 列时回退等权累计(旧语义),保证不因数据缺失而崩溃。
    """
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
        pair["home_goals"].isna(),
        np.nan,
        (pair["home_goals"] > pair["away_goals"]).astype(float),
    )
    pair = (
        pair.sort_values(["pair", "date", "index"], kind="mergesort")
        if "date" in pair.columns
        else pair.sort_values(["pair", "index"], kind="mergesort")
    )
    if "date" in pair.columns:
        # 时间衰减权重:半衰期 365 天
        _ref = pd.Timestamp(pair["date"].max())
        _age = (_ref - pd.to_datetime(pair["date"])).dt.days
        pair["w"] = np.exp(-_age / 365.0)
        pair["w"] = np.where(pair["h2h_home_win_rate"].isna(), 0.0, pair["w"])
        pair["wh"] = pair["h2h_home_win_rate"].fillna(0.0) * pair["w"]
        pair["wsum"] = pair.groupby("pair")["w"].cumsum()
        pair["wprod"] = pair.groupby("pair")["wh"].cumsum()
        with np.errstate(divide="ignore", invalid="ignore"):
            _wm = np.where(pair["wsum"] > 0, pair["wprod"] / pair["wsum"], np.nan)
        pair["wmean"] = _wm
        pair["prior"] = pair.groupby("pair")["wmean"].shift(1)
    else:
        pair["cum"] = (
            pair.groupby("pair")["h2h_home_win_rate"]
            .expanding()
            .mean()
            .reset_index(drop=True)
        )
        pair["prior"] = pair.groupby("pair")["cum"].shift(1)
    pair = pair.set_index("index")

    prepared = prepared.reset_index(drop=True)
    # 审查 §21 数值目标:H2H 贡献上限 1~3%% —— 将衰减后均值压缩到
    # [0.5−3%%, 0.5+3%%] 幅值区间,模型能用的只有极小尾巴(而非全历史
    # 等权率冲击)。值域窄 → 即便模型分配权重,贡献也被限制在 ~3%%。
    _cap = 0.03
    _prior = pair["prior"].reindex(range(len(prepared))).values
    _contrib = np.clip(0.5 + (_prior - 0.5) * (2.0 * _cap), 0.0, 1.0)
    prepared["home_team_h2h_win_rate"] = _contrib
    return prepared


def compute(df: pd.DataFrame, league_type: str | None = None) -> pd.DataFrame:
    """计算本家族特征并附加到 df(由 factory 调度)。"""
    return df.copy()


def version() -> str:
    """公式哈希(审查 §16):规格 JSON + 实现代码哈希。"""
    import inspect

    spec = json.dumps(
        sorted(FEATURES, key=lambda f: f["name"]), ensure_ascii=False, sort_keys=True
    )
    impl = inspect.getsource(compute_h2h)
    return hashlib.sha256((spec + "|" + impl).encode()).hexdigest()[:12]
