"""概率分布唯一实现(REUSE 统一:消除 5 份 PMF / 7 处矩阵 / 5 处归约重复)。

- pois_pmf:标量泊松 PMF
- pois_pmf_vec:向量化泊松分布(对数空间,数值稳定)
- pois_matrix:10×10 双泊松联合概率矩阵
- matrix_to_probs:概率矩阵 → 胜/平/负
"""

from __future__ import annotations

import math

import numpy as np

MAX_GOALS = 10


def pois_pmf(lam: float, k: int) -> float:
    """泊松 PMF: P(X=k)。"""
    return math.exp(-lam) * lam**k / math.factorial(k)


def pois_pmf_vec(lam: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """向量化泊松分布(0..max_goals),对数空间计算并归一化。"""
    ks = np.arange(max_goals + 1)
    logp = (
        ks * math.log(max(lam, 1e-12))
        - lam
        - np.array([math.lgamma(i + 1) for i in ks])
    )
    p = np.exp(logp)
    return p / p.sum()


def pois_matrix(
    lam_home: float, lam_away: float, max_goals: int = MAX_GOALS
) -> np.ndarray:
    """10×10 双泊松联合概率矩阵(向量化:外积)。"""
    ks = np.arange(max_goals)
    logp_h = (
        ks * math.log(max(lam_home, 1e-12))
        - lam_home
        - np.array([math.lgamma(i + 1) for i in ks])
    )
    logp_a = (
        ks * math.log(max(lam_away, 1e-12))
        - lam_away
        - np.array([math.lgamma(i + 1) for i in ks])
    )
    m = np.exp(logp_h[:, None] + logp_a[None, :])
    return m / m.sum()


def matrix_to_probs(m: np.ndarray) -> tuple[float, float, float]:
    """概率矩阵 → (home_win, draw, away_win)。"""
    phw = float(np.tril(m, -1).sum())
    pd_ = float(np.diag(m).sum())
    paw = float(np.triu(m, 1).sum())
    s = phw + pd_ + paw
    return phw / s, pd_ / s, paw / s


def poisson_loss(lam: float, actual: float) -> float:
    """负对数似然(训练评估用)。"""
    return -math.log(max(1e-12, pois_pmf(lam, round(actual))))
