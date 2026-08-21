"""Uncertainty Engine(审查九 二十四/二十五/三十二 拆分)。

heuristic confidence(概率 × 一致度 × 数据质量)+ entropy + disagreement。
注意:confidence_score 是 heuristic 强度分,非统计学置信度(审查二十五:
真正的 uncertainty 应来自 ensemble distribution + calibration + data +
parameter uncertainty —— 后续深化)。
"""

from __future__ import annotations

import math

import numpy as np


def compute(final_probs, member_probs, gbm_probs, feat, feature_columns) -> dict:
    """计算不确定性指标。

    final_probs: 最终 1X2(调整后);member_probs: goal 成员概率 dict;
    gbm_probs: GBM 概率或 None;feat: prepare 特征(白名单 NaN 统计)。
    返回 {entropy, disagreement, data_quality, agreement, confidence_score,
          confidence}。
    """
    hw, dr, aw = (float(x) for x in final_probs)
    entropy = -sum(p * math.log(p) for p in (hw, dr, aw) if p > 0)
    disagreement = 0.0
    all_p = [list(v) for v in (member_probs or {}).values() if v is not None]
    if gbm_probs is not None:
        all_p.append(list(gbm_probs))
    if len(all_p) >= 2:
        arr = np.array(all_p, dtype=float)
        means = arr.mean(axis=0)
        disagreement = float(np.abs(arr - means).mean())  # 平均绝对偏差(0=一致)
    data_quality = 1.0
    try:
        if feat is not None and len(feat) >= 2:
            fcols = [c for c in feature_columns if c in feat.columns]
            if fcols:
                nan_frac = float(feat.iloc[-2:][fcols].isna().mean().mean())
                data_quality = round(max(0.0, 1.0 - nan_frac), 4)
    except Exception:
        data_quality = 1.0
    agreement = round(max(0.0, 1.0 - disagreement), 4)
    confidence = max(hw, dr, aw)
    confidence_score = round(confidence * agreement * data_quality, 4)
    return {
        "entropy": round(entropy, 4),
        "disagreement": disagreement,
        "data_quality": data_quality,
        "agreement": agreement,
        "confidence": round(confidence, 4),
        "confidence_score": confidence_score,
    }


def recompute_after_adjust(final_probs, agreement, data_quality) -> dict:
    """Regime 调整后重算置信度(最终概率变化)。"""
    hw, dr, aw = (float(x) for x in final_probs)
    entropy = -sum(p * math.log(p) for p in (hw, dr, aw) if p > 0)
    confidence = max(hw, dr, aw)
    return {
        "confidence": round(confidence, 4),
        "confidence_score": round(confidence * agreement * data_quality, 4),
        "entropy": round(entropy, 4),
    }
