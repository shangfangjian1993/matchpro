"""τ/φ 拟合 + 权重优化。

Layer-1: Poisson Goal NLL (λ fusion)
Layer-2: 1X2 LogLoss (score distribution,基于 fused λ)
Layer-3: Outcome GBM (单独学习)
"""
from __future__ import annotations

import numpy as np


def fit_tau(samples, use_fused_lambda: bool = False):
    """τ 拟合:在低比分格点上最大化 log-likelihood。
    
    Args:
        samples: OOF 样本列表
        use_fused_lambda: 是否使用 fused λ(而非 HGBR λ)
    """
    from app.models.distributions import pois_pmf as _pois_pmf
    from app.models.dixon_coles.dc import _dc_tau

    best_t, best_ll = 0.0, -np.inf  # P0-1 FIX: 最大化 log-likelihood
    for t in np.arange(-0.2, 0.201, 0.01):
        ll = 0.0
        for s in samples:
            x, y = s["home_goals"], s["away_goals"]
            if x > 1 or y > 1:
                continue
            # P0-4: 使用 fused λ(如果可用)
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
        if ll > best_ll:  # P0-1 FIX: 最大化
            best_t, best_ll = t, ll
    return float(best_t)


def optimize(oof_samples, tau, phi, shrinkage: float = 0.15, use_fused_lambda: bool = False):
    """分层权重学习。
    
    Layer-1: Poisson Goal NLL (λ fusion)
    Layer-2: 1X2 LogLoss (shape)
    Layer-3: Outcome GBM (α * shape + (1-α) * gbm)
    """
    from app.models.ensemble import fit_nb_phi, learn_weights

    from .member_builder import build_member_samples

    # P0-4: τ/φ 在 fused λ 上估计
    if use_fused_lambda:
        # 先用 preliminary weights 计算 fused λ
        preliminary_samples = build_member_samples(oof_samples, tau, phi)
        phi = fit_nb_phi(preliminary_samples) if phi is None else phi
        if tau is None:
            tau = fit_tau(preliminary_samples, use_fused_lambda=True)
    
    samples = build_member_samples(oof_samples, tau, phi)
    w = learn_weights(samples, tau, phi, shrinkage=shrinkage)
    return phi, w, samples


def optimize_outcome_weights(oof_samples):
    """Layer-3: 学习 Outcome GBM 权重。
    
    输入:shape_1x2, gbm_1x2, actual
    输出:shape_weight, gbm_weight
    """
    shape_probs = []
    gbm_probs = []
    actuals = []
    
    for s in oof_samples:
        if "shape_1x2" not in s or "gbm" not in s:
            continue
        shape_probs.append(s["shape_1x2"])
        gbm_probs.append(s["gbm"])
        actuals.append(s["actual"])
    
    if not shape_probs:
        return 1.0, 0.0  # fallback: shape only
    
    shape_probs = np.array(shape_probs)
    gbm_probs = np.array(gbm_probs)
    actuals = np.array(actuals)
    
    best_alpha = 1.0
    best_ll = np.inf
    
    for alpha in np.arange(0.0, 1.01, 0.05):
        # 融合概率
        p = alpha * shape_probs + (1 - alpha) * gbm_probs
        p = np.clip(p, 1e-12, None)
        p = p / p.sum(axis=1, keepdims=True)
        # LogLoss
        ll = -np.mean(np.log(p[np.arange(len(actuals)), actuals]))
        if ll < best_ll:
            best_ll = ll
            best_alpha = alpha
    
    return float(best_alpha), float(1.0 - best_alpha)
