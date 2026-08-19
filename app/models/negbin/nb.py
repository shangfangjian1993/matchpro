"""Negative Binomial 模型(§1.1 app/models/negbin):过离散修正。"""

from __future__ import annotations

import math

import numpy as np

from app.models.distributions import pois_pmf as _pois_pmf


def _nb_logpmf(k: int, mu: float, phi: float) -> float:
    """负二项 log-pmf:均值 μ,离散参数 φ(Var = μ + μ²/φ);φ→∞ 退化为泊松。"""
    if phi > 1e6:
        return math.log(_pois_pmf(mu, k))
    r = max(1e-6, phi)
    try:
        return (
            math.lgamma(k + r)
            - math.lgamma(k + 1)
            - math.lgamma(r)
            + r * math.log(r / (r + mu))
            + k * math.log(mu / (r + mu))
        )
    except (ValueError, OverflowError):
        return -1e9


def nb_probs(
    lam_home: float, lam_away: float, phi: float = 1e9
) -> tuple[float, float, float]:
    """负二项联合分布 → 胜平负。φ 大时退化为泊松。"""
    m = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            m[i, j] = math.exp(
                _nb_logpmf(i, lam_home, phi) + _nb_logpmf(j, lam_away, phi)
            )
    m = m / m.sum()
    ph = float(np.tril(m, -1).sum())
    pd_ = float(np.diag(m).sum())
    pa = float(np.triu(m, 1).sum())
    s = ph + pd_ + pa
    return ph / s, pd_ / s, pa / s


def fit_nb_phi(samples: list[dict]) -> float:
    """φ 拟合:进球数的过离散度(Var/Mean-1)倒数;≈0 时取大值=泊松。"""
    goals = [s["home_goals"] for s in samples] + [s["away_goals"] for s in samples]
    mean = float(np.mean(goals))
    var = float(np.var(goals))
    if var <= mean or mean <= 0:
        return 1e9
    return float(mean / (var - mean))
