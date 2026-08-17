"""τ/φ 拟合 + 权重优化(审查九 P1-9 拆分)。"""
from __future__ import annotations

import numpy as np


def fit_tau(samples):
    """τ 拟合:在 (0,0)/(0,1)/(1,0)/(1,1) 低比分格点上最大似然。"""
    from app.models.distributions import pois_pmf as _pois_pmf
    from app.models.dixon_coles.dc import _dc_tau
    best_t, best_ll = 0.0, float("inf")
    for t in np.arange(-0.2, 0.201, 0.01):
        ll = 0.0
        for s in samples:
            x, y = s["home_goals"], s["away_goals"]
            if x > 1 or y > 1:
                continue
            p = _dc_tau(x, y, s["hgbr_lam_h"], s["hgbr_lam_a"], t) * \
                _pois_pmf(s["hgbr_lam_h"], x) * _pois_pmf(s["hgbr_lam_a"], y)
            ll += np.log(max(1e-12, p))
        if ll < best_ll:
            best_t, best_ll = t, ll
    return float(best_t)


def optimize(samples, tau, phi, shrinkage: float = 0.15):
    """SLSQP 学习权重(动态成员,shrinkage 向 hgbr 先验收缩)。"""
    from app.models.ensemble import fit_nb_phi, learn_weights
    phi = fit_nb_phi(samples) if phi is None else phi
    w = learn_weights(samples, tau, phi, shrinkage=shrinkage)
    return phi, w
