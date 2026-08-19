"""Feature Factory 统一入口(审查 §14/§18:ELO 也由 Factory 管理,单一体系)。

顺序:strength(ELO)→ attack_defense(进失球/指标)→ form(胜率/近期)→ h2h(交手)。
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

from app.features.rolling import build_long_table
from app.features.strength import compute as compute_strength
from app.features.attack_defense import (compute_attack_defense,
                                         compute_metric_rolling,
                                         compute_side_metric_rolling)
from app.features.form import compute_form
from app.features.h2h import compute_h2h


def compute_all(df: pd.DataFrame,
                league_type: str | None = None,
                metric_columns: tuple = (),
                side_metric_columns: tuple = (),
                hist_matches=None) -> pd.DataFrame:
    """统一特征计算(审查 §18:ELO 注入也在 Factory 内,外部不再 with_elo_features)。"""
    out = compute_strength(df, league_type)  # 01 Team Strength(ELO)
    long, _ = build_long_table(out)
    out = compute_attack_defense(out, long)  # 02 Attack/Defense
    out = compute_form(out, long)             # 03 Form & Momentum
    # 审查七 V7-3:H2H 默认关闭(configs/models.yaml features.h2h)——
    # 样本少+阵容/教练变化大,2021 交手≠2026 交手;特殊场景再开
    try:
        from app.core.config import load_yaml
        _h2h_enabled = (load_yaml("models.yaml").get("features") or {}).get("h2h", True)
    except Exception:
        _h2h_enabled = True
    if _h2h_enabled:
        out = compute_h2h(out)                # 06 Opponent Interaction
    for metric in metric_columns:
        out = compute_metric_rolling(out, metric, metric)
    for metric in side_metric_columns:
        out = compute_side_metric_rolling(out, metric, metric)
    # stats 特征族(team_match_stats 深度统计;数据未到位时附空列,模型自动跳过)
    if hist_matches is not None:
        try:
            from app.features.stats_features import rolling_team_stats as _rolling_stats
            _stats_df = _rolling_stats(hist_matches)
            if not _stats_df.empty:
                _stat_cols = [c for c in _stats_df.columns
                              if c.startswith("home_tms") or c.startswith("away_tms")]
                if _stat_cols:
                    out = out.join(_stats_df[_stat_cols], how="left")
        except Exception as _se:
            logger.warning("stats 特征附加失败(降级,不影响主链路): %s", _se)
    return out
