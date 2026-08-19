"""Feature Factory 统一入口(审查 §14/§18:ELO 也由 Factory 管理,单一体系)。

顺序:strength(ELO)→ attack_defense(进失球/指标)→ form(胜率/近期)→ h2h(交手)。
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

from app.features.attack_defense import (
    compute_attack_defense,
    compute_metric_rolling,
    compute_opponent_adjusted_xg,
    compute_side_metric_rolling,
)
from app.features.form import compute_form
from app.features.h2h import compute_h2h
from app.features.rolling import build_long_table
from app.features.strength import compute as compute_strength


def compute_all(
    df: pd.DataFrame,
    league_type: str | None = None,
    metric_columns: tuple = (),
    side_metric_columns: tuple = (),
    hist_matches=None,
) -> pd.DataFrame:
    """统一特征计算(审查 §18:ELO 注入也在 Factory 内,外部不再 with_elo_features)。"""
    out = compute_strength(df, league_type)  # 01 Team Strength(ELO)
    long, _ = build_long_table(out)
    out = compute_attack_defense(out, long)  # 02 Attack/Defense
    out = compute_opponent_adjusted_xg(out, long)  # Opponent-adjusted xG(A70A601 §20)
    out = compute_form(out, long)  # 03 Form & Momentum
    # 审查七 V7-3:H2H 默认关闭(configs/models.yaml features.h2h)——
    # 样本少+阵容/教练变化大,2021 交手≠2026 交手;特殊场景再开
    try:
        from app.core.config.features import feature_flags

        _h2h_enabled = feature_flags().get("h2h", True)
    except Exception:
        _h2h_enabled = True
    if _h2h_enabled:
        out = compute_h2h(out)  # 06 Opponent Interaction
    for metric in metric_columns:
        out = compute_metric_rolling(out, metric, metric)
    for metric in side_metric_columns:
        out = compute_side_metric_rolling(out, metric, metric)
    # stats 特征族(team_match_stats 深度统计)
    # 审查 ae724d5:
    #   - 按 match_id 显式 merge(不依赖 DataFrame 行序/index 对齐)
    #   - 异常分级:数据缺失(hist 空/无 stats)→ NaN 列自动跳过(可降级);
    #     实现/schema 异常 → 抛出(不静默降级,由上层 fail/invalid 标记)
    if hist_matches:
        # 审查 ae724d5:按 match_id 显式 merge(不依赖行序/index 对齐)
        from app.features.stats_features import rolling_team_stats as _rolling_stats

        _stats_df = _rolling_stats(hist_matches)  # index=match_id;空或 NaN 列则跳过
        _stat_cols = [
            col for col in _stats_df.columns if col.startswith(("home_tms", "away_tms"))
        ]
        if _stat_cols and len(_stats_df):
            _idx = out.index
            _tmp = out.reset_index(drop=True)
            _tmp["match_id"] = [
                m.id if m is not None else None for m in hist_matches
            ]  # 与 hist 顺序一致(调用方保证);None=无落库行(新预测)→ 左连不足留 NaN
            _stats_right = _stats_df[_stat_cols].copy()
            _stats_right = _stats_right.reset_index()  # index(name=match_id) → 列
            out = pd.merge(_tmp, _stats_right, on="match_id", how="left")
            out = out.drop(columns=["match_id"])  # 仅作对齐键,不作特征
            out = out.set_index(_idx)
    return out
