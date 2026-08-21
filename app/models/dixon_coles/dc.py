"""Dixon-Coles 模型(1.1 app/models/dixon_coles):低比分相关性 τ 修正。"""

from __future__ import annotations

import numpy as np

from app.models.distributions import pois_pmf as _pois_pmf


def _dc_tau(x: int, y: int, lam: float, mu: float, tau: float) -> float:
 """Dixon-Coles 低比分修正因子(标准形式)。"""
 if x == 0 and y == 0:
 return 1.0 - lam * mu * tau
 if x == 0 and y == 1:
 return 1.0 + lam * tau
 if x == 1 and y == 0:
 return 1.0 + mu * tau
 if x == 1 and y == 1:
 return 1.0 - tau
 return 1.0


def dc_probs(
 lam_home: float, lam_away: float, tau: float = 0.0
) -> tuple[float, float, float]:
 """DC 修正联合分布 → 胜平负。τ=0 退化为普通泊松。"""
 m = np.zeros((10, 10))
 ph = [_pois_pmf(lam_home, i) for i in range(10)]
 pa = [_pois_pmf(lam_away, j) for j in range(10)]
 for i in range(10):
 for j in range(10):
 m[i, j] = ph[i] * pa[j]
 for i in range(min(10, 2)):
 for j in range(min(10, 2)):
 m[i, j] *= _dc_tau(i, j, lam_home, lam_away, tau)
 m = m / m.sum()
 phw = float(np.tril(m, -1).sum())
 pd_ = float(np.diag(m).sum())
 paw = float(np.triu(m, 1).sum())
 s = phw + pd_ + paw
 return phw / s, pd_ / s, paw / s


def fit_dc_tau(samples: list[dict]) -> float:
 """τ 最大似然拟合(委托给新实现)。"""
 from app.services.training.ensemble.optimizers.dc_parameter import fit_tau
 return fit_tau(samples)
