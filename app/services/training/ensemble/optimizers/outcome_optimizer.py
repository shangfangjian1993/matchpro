"""Layer-3 Outcome GBM 权重优化(连续 bounded optimization)。"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from . import EnsembleTrainingConfig


def optimize_outcome_weights(
 shape_probs: list,
 gbm_probs: list,
 actuals: list,
 config: EnsembleTrainingConfig | None = None,
) -> tuple[float, float]:
 """Layer-3: 学习 Outcome GBM 权重。
 
 输入: shape_1x2, gbm_1x2, actual
 输出: (shape_weight, gbm_weight), 和为 1
 
 使用 bounded scalar optimization (连续,非 grid search)。
 """
 if config is None:
 config = EnsembleTrainingConfig()
 
 if not shape_probs or not gbm_probs:
 return 1.0, 0.0
 
 shape_probs = np.asarray(shape_probs, dtype=float)
 gbm_probs = np.asarray(gbm_probs, dtype=float)
 actuals = np.asarray(actuals, dtype=int)
 
 def _nll(alpha: float) -> float:
 """负对数似然。"""
 p = alpha * shape_probs + (1 - alpha) * gbm_probs
 p = np.clip(p, 1e-12, None)
 p = p / p.sum(axis=1, keepdims=True)
 return -np.mean(np.log(p[np.arange(len(actuals)), actuals]))
 
 if config.outcome_method == "bounded":
 # 连续 bounded optimization
 result = minimize_scalar(_nll, bounds=(0.0, 1.0), method="bounded",
 options={"xatol": 1e-4, "maxiter": 100})
 best_alpha = result.x
 else:
 # fallback: grid search
 best_alpha = 1.0
 best_ll = np.inf
 for alpha in np.arange(0.0, 1.01, 0.05):
 ll = _nll(alpha)
 if ll < best_ll:
 best_ll = ll
 best_alpha = alpha
 
 return float(best_alpha), float(1.0 - best_alpha)
