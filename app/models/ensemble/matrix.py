"""比分矩阵与派生输出(审查 §36:ensemble 拆分)。"""

from __future__ import annotations

import numpy as np

from app.models.distributions import MAX_GOALS
from app.models.distributions import pois_matrix as _pois_matrix


def _dc_matrix(lam_home: float, lam_away: float, tau: float = 0.0) -> np.ndarray:
    """Dixon-Coles 修正比分矩阵(τ=0 退化为泊松)。"""
    from app.models.dixon_coles.dc import _dc_tau

    m = _pois_matrix(lam_home, lam_away)
    for i in range(min(10, 2)):
        for j in range(min(10, 2)):
            m[i, j] *= _dc_tau(i, j, lam_home, lam_away, tau)
    return m / m.sum()


def _nb_matrix(lam_home: float, lam_away: float, phi: float = 1e9) -> np.ndarray:
    """负二项比分矩阵(φ→∞ 退化为泊松)。"""
    import math

    from app.models.negbin.nb import _nb_logpmf

    m = np.zeros((MAX_GOALS, MAX_GOALS))
    for i in range(MAX_GOALS):
        for j in range(MAX_GOALS):
            m[i, j] = math.exp(
                _nb_logpmf(i, lam_home, phi) + _nb_logpmf(j, lam_away, phi)
            )
    return m / m.sum()


def fuse_score_matrix(
    member_matrices: dict[str, np.ndarray], weights: dict | None = None
) -> np.ndarray:
    """比分矩阵融合:M_final = Σ w_i · M_i。"""
    w = weights or {"hgbr": 1.0, "dc": 0.0, "nb": 0.0, "elo": 0.0, "gbm": 0.0}
    _size = MAX_GOALS
    if member_matrices:
        _size = max((m.shape[0] for m in member_matrices.values()), default=MAX_GOALS)
    out = np.zeros((_size, _size))
    for name, m in member_matrices.items():
        out += w.get(name, 0.0) * m
    s = out.sum()
    if s <= 0:
        return member_matrices.get("hgbr", _pois_matrix(1.5, 1.2))
    return out / s


def score_outputs(matrix: np.ndarray) -> dict:
    """从比分矩阵派生 Top5 / Over-Under / BTTS / xG。"""
    m = np.asarray(matrix, dtype=float)
    n = m.shape[0]
    flat = [(i, j, float(m[i, j])) for i in range(n) for j in range(n)]
    flat.sort(key=lambda t: -t[2])
    top5 = [{"home": i, "away": j, "probability": round(p, 5)} for i, j, p in flat[:5]]
    over = sum(m[i, j] for i in range(n) for j in range(n) if i + j > 2)
    btts = sum(m[i, j] for i in range(n) for j in range(n) if i > 0 and j > 0)
    xg_home = sum(i * m[i, j] for i in range(n) for j in range(n))
    xg_away = sum(j * m[i, j] for i in range(n) for j in range(n))
    return {
        "top_scores": top5,
        "over_2_5": round(float(over), 4),
        "under_2_5": round(1.0 - float(over), 4),
        "btts": round(float(btts), 4),
        "expected_xg": [round(float(xg_home), 3), round(float(xg_away), 3)],
    }
