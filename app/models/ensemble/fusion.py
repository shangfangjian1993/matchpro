"""融合层(审查 §36:ensemble 拆分):概率融合 + 两层 Goal/Outcome。"""

from __future__ import annotations

import numpy as np

_GOAL_MEMBERS = ("hgbr", "dc", "nb", "elo", "bayes")


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
    """两层融合(审查 P0-9):Goal Engine(4 成员)1X2 与 GBM(Outcome)融合。"""
    w = weights or {"hgbr": 1.0, "dc": 0.0, "nb": 0.0, "elo": 0.0, "gbm": 0.0}
    w_goal = sum(w.get(k, 0.0) for k in _GOAL_MEMBERS)
    w_gbm = w.get("gbm", 0.0)
    if gbm_probs is None or (w_goal + w_gbm) <= 0:
        return tuple(goal_probs)
    total = w_goal + w_gbm
    return tuple(
        round(w_goal / total * g + w_gbm / total * b, 6)
        for g, b in zip(goal_probs, gbm_probs)
    )
