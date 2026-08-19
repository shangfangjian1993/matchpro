"""ELO-Goal 模型(§1.1 app/models/elo_goal):纯 ELO 期望进球。"""

from __future__ import annotations

# 期望进球基线 + 每 400 分差对应球数(与 attack ELO 一致)
ELO_BASE = 1.6
ELO_SPREAD = 1.0


def elo_goal_lambda(attack_elo_diff: float, home: bool = True) -> float:
    """由攻击 ELO 差推导期望进球(独立于 HGBR 的信息源)。"""
    if home:
        return max(0.3, ELO_BASE + attack_elo_diff / 400.0 * ELO_SPREAD)
    return max(0.3, ELO_BASE - attack_elo_diff / 400.0 * ELO_SPREAD)
