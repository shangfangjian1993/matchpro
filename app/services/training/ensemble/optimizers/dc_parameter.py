"""Dixon-Coles τ 参数拟合。"""
from __future__ import annotations

import numpy as np
import warnings

from . import EnsembleTrainingConfig


def fit_tau(samples: list[dict], config: EnsembleTrainingConfig | None = None,
            use_fused_lambda: bool = False) -> float:
    """τ 拟合:在低比分格点上最大化 log-likelihood。"""
    if config is None:
        config = EnsembleTrainingConfig()
    
    from app.models.distributions import pois_pmf as _pois_pmf
    from app.models.dixon_coles.dc import _dc_tau

    best_t, best_ll = 0.0, -np.inf
    for t in np.arange(config.tau_min, config.tau_max + config.tau_step, config.tau_step):
        ll = 0.0
        for s in samples:
            x, y = s["home_goals"], s["away_goals"]
            if x > 1 or y > 1:
                continue
            if use_fused_lambda and "fused_lam_h" in s:
                lam_h, lam_a = s["fused_lam_h"], s["fused_lam_a"]
            else:
                lam_h, lam_a = s["hgbr_lam_h"], s["hgbr_lam_a"]
            p = (
                _dc_tau(x, y, lam_h, lam_a, t)
                * _pois_pmf(lam_h, x)
                * _pois_pmf(lam_a, y)
            )
            ll += np.log(max(1e-12, p))
        if ll > best_ll:
            best_t, best_ll = t, ll
    
    # Parameter sanity check
    if abs(best_t) > config.tau_warning_threshold:
        warnings.warn(f"DC τ={best_t:.3f} exceeds threshold ±{config.tau_warning_threshold}")
    
    return float(best_t)
