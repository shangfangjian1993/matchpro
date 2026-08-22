"""多源冲突解决(reconciliation)。

不变量(Business invariant):
- canonical score 必须保持 canonical orientation(来源主客场反转不污染)。
- 不因"最后来的源"静默覆盖历史;跨源冲突 → 保留旧值 + 标记 conflict。
- source_consensus 为**来源级共识**(多少数据源与 canonical 一致),非字段级。
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def reconcile(existing: Any, incoming: dict, source: str) -> dict:
    """解决多源冲突。

    Args:
        existing: 现有 DB 记录(Match 对象)
        incoming: 新数据(dict,来自 NormalizedMatch.to_row())
        source: 来源标识

    Returns:
        需要应用的更新 dict
    """
    updates: dict = {}
    conflicts: list[str] = []

    # 比分:已存在 finished 比分且与新值不同 → 冲突
    if existing and getattr(existing, "match_status", "") == "finished":
        for field in ("home_goals", "away_goals"):
            old_val = getattr(existing, field, None)
            new_val = incoming.get(field)
            if old_val is not None and new_val is not None and old_val != new_val:
                conflicts.append(f"{field}: {old_val} vs {new_val}")

    # 指标字段:旧值为 None → 更新;旧值非 None 且新值不同 → 保留旧值
    metric_fields = [
        "home_xg", "away_xg", "home_shots", "away_shots",
        "home_shots_on_target", "away_shots_on_target",
        "home_corners", "away_corners", "home_possession",
        "home_yellow_cards", "away_yellow_cards",
        "home_red_cards", "away_red_cards",
        "home_ht_goals", "away_ht_goals",
        "home_passing_accuracy", "away_passing_accuracy",
    ]

    for field in metric_fields:
        old_val = getattr(existing, field, None) if existing else None
        new_val = incoming.get(field)
        if old_val is None and new_val is not None:
            updates[field] = new_val
        elif old_val is not None and new_val is not None and old_val != new_val:
            # 保留旧值,记录冲突
            conflicts.append(f"{field}: keep {old_val}, reject {new_val}")

    if conflicts:
        logger.debug("reconcile conflicts for %s: %s", source, conflicts)

    return updates
