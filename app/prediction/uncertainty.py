""": Uncertainty 量化与校准深化。

当前模型已有基础 uncertainty 计算,此处深化:
- 模型分歧 (inter-member disagreement)
- 数据质量 (特征 NaN 率)
- 预测置信度 (entropy-based)
"""
from __future__ import annotations

import math

import numpy as np


def compute(probs_1x2, members, gbm_probs, feat, feature_columns):
 """Backward compatibility: old uncertainty_compute signature。
 
 Args:
 probs_1x2: (home_win, draw, away_win) tuple
 members: dict of member probabilities
 gbm_probs: GBM 1X2 probs (or None)
 feat: feature DataFrame
 feature_columns: list of feature column names
 """
 import numpy as np
 import pandas as pd
 
 # Compute feature NaN rate
 feature_nan_rate = 0.0
 if feat is not None and hasattr(feat, 'isna'):
 feature_nan_rate = float(feat.isna().mean().mean()) if feat.size > 0 else 0.0
 
 # History count from members
 history_count = 0
 if members and len(members) > 0:
 first_member = list(members.values())[0]
 if hasattr(first_member, '__len__') and len(first_member) > 0:
 history_count = len(first_member)
 
 # Build a dummy matrix from 1X2 probs
 matrix = np.zeros((10, 10))
 matrix[0, 0] = probs_1x2[1] # draw
 matrix[1, 0] = probs_1x2[0] / 2 # home win
 matrix[0, 1] = probs_1x2[2] / 2 # away win
 matrix[1, 1] = 0.1
 matrix = matrix / matrix.sum()
 
 result = compute_uncertainty(members, {}, matrix, feature_nan_rate, history_count)
 # Return with old key names for backward compatibility
 return {
 "agreement": 1.0 - result["disagreement"], "disagreement": result["disagreement"],
 "data_quality": 1.0 - result["data_quality_penalty"],
 "confidence": result["confidence"], "confidence_score": result["confidence"],
 "entropy": result["entropy"],
 "feature_nan_rate": result["feature_nan_rate"],
 "history_factor": result["history_factor"],
 }


def recompute_after_adjust(uncertainty_dict, adjustment_factor):
 """Backward compatibility: recompute uncertainty after adjustment。"""
 result = dict(uncertainty_dict)
 if "confidence" in result:
 result["confidence"] = max(0.0, min(1.0, result["confidence"] * (1.0 - adjustment_factor)))
 return result


def compute_uncertainty(
 members: dict,
 weights: dict,
 matrix: np.ndarray,
 feature_nan_rate: float = 0.0,
 history_count: int = 0,
) -> dict:
 """计算预测的不确定性。
 
 Args:
 members: 各成员 1X2 概率 {name: (hw, dr, aw)}
 weights: 各成员权重
 matrix: score matrix
 feature_nan_rate: 特征 NaN 率 (0-1)
 history_count: 历史比赛场数
 
 Returns:
 uncertainty dict
 """
 # 模型分歧 (inter-member std)
 member_names = [n for n in ["hgbr", "dc", "nb", "elo", "bayes"] if n in members]
 if not member_names:
 member_names = list(members.keys())
 
 probs = np.array([members[n] for n in member_names])
 disagreement = float(np.mean(np.std(probs, axis=0)))
 
 # 预测熵
 fused = np.zeros(3)
 w_sum = 0.0
 for n in member_names:
 w = weights.get(n, 0.0)
 fused += w * np.array(members[n])
 w_sum += w
 if w_sum > 0:
 fused /= w_sum
 
 # Shannon entropy (normalized to [0, 1])
 entropy = -sum(p * math.log(max(p, 1e-12)) for p in fused)
 max_entropy = math.log(3) # max entropy for 3 outcomes
 norm_entropy = entropy / max_entropy
 
 # 数据质量惩罚
 data_quality_penalty = feature_nan_rate * 0.5 # up to 50% penalty
 
 # 历史数据充分性
 history_factor = min(1.0, history_count / 100) # 100+ games = full confidence
 
 # 综合置信度
 confidence = 1.0 - norm_entropy
 confidence *= (1.0 - data_quality_penalty)
 confidence *= (0.5 + 0.5 * history_factor) # at least 50% confidence
 
 return {
 "disagreement": round(disagreement, 6),
 "entropy": round(norm_entropy, 6),
 "confidence": round(max(0.0, min(1.0, confidence)), 6),
 "data_quality_penalty": round(data_quality_penalty, 6),
 "history_factor": round(history_factor, 6),
 "feature_nan_rate": round(feature_nan_rate, 6),
 }
