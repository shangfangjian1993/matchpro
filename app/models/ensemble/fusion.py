"""融合层:概率融合 + 两层 Goal/Outcome。

Layer-3 使用 shape_weight + gbm_weight(由 optimize_outcome_weights 学习)。
"""
from __future__ import annotations

import numpy as np


def fuse_probs(
    member_probs_list: dict[str, tuple[float, float, float]],
    weights: dict | None = None,
) -> tuple[float, float, float]:
    """概率层融合:P_final = Σ w_i · P_i。"""
    w = weights or {"hgbr": 1.0, "dc": 0.0, "nb": 0.0, "elo": 0.0, "gbm": 0.0}
    out = np.zeros(3)
    for name, p in member_probs_list.items():
        out += w.get(name, 0.0) * np.asarray(p)
    s = out.sum()
    if s <= 0:
        return tuple(member_probs_list.get("hgbr", (0.5, 0.25, 0.25)))
    return tuple(float(x / s) for x in out)


def fuse_goal_outcome(
    goal_probs: tuple[float, float, float],
    gbm_probs: tuple[float, float, float] | None,
    weights: dict | None = None,
) -> tuple[float, float, float]:
    """Layer-3 融合:Shape 1X2 与 GBM 1X2 融合。
    
    使用 learned shape_weight + gbm_weight(由 optimize_outcome_weights 学习)。
    weights 中必须包含 "shape_weight" 和 "gbm_weight"。
    """
    if gbm_probs is None:
        return tuple(goal_probs)
    
    # P0-6 FIX: 使用 learned shape_weight + gbm_weight
    shape_weight = weights.get("shape_weight", 1.0) if weights else 1.0
    gbm_weight = weights.get("gbm_weight", 0.0) if weights else 0.0
    
    total = shape_weight + gbm_weight
    if total <= 0:
        return tuple(goal_probs)
    
    return tuple(
        round(shape_weight / total * g + gbm_weight / total * b, 6)
        for g, b in zip(goal_probs, gbm_probs)
    )
