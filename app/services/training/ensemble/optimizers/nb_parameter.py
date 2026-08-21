"""Negative Binomial φ 参数拟合。"""
from __future__ import annotations

import warnings

from . import EnsembleTrainingConfig


def fit_phi(samples: list[dict], config: EnsembleTrainingConfig | None = None) -> float:
    """φ 拟合:基于 overdispersion 估计。
    
    φ → ∞ 时 NB 退化为 Poisson。
    φ 估计为 Var/Mean - 1 的倒数。
    """
    if config is None:
        config = EnsembleTrainingConfig()
    
    goals = []
    for s in samples:
        goals.append(s.get("home_goals", 0))
        goals.append(s.get("away_goals", 0))
    
    if len(goals) < 10:
        return 1e9  # 样本太少,退化为 Poisson
    
    import numpy as np
    mean = np.mean(goals)
    var = np.var(goals)
    
    if var <= mean or mean <= 0:
        return 1e9  # 无过离散
    
    phi = mean / (var - mean)
    
    # Bounds check
    phi = max(config.phi_min, min(config.phi_max, phi))
    
    # Warning
    if phi < config.phi_warning_threshold:
        warnings.warn(f"NB φ={phi:.2f} indicates strong overdispersion")
    
    return float(phi)
