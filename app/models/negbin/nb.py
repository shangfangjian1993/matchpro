"""Negative Binomial 模型(1.1 app/models/negbin):过离散修正。"""

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
 """φ 拟合(委托给新实现)。"""
 from app.services.training.ensemble.optimizers.nb_parameter import fit_phi
 return fit_phi(samples)
