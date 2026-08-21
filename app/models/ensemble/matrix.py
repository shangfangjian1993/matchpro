"""比分矩阵与派生输出(Tail Mass Tracking)。"""
from __future__ import annotations

import numpy as np

from app.models.distributions import MAX_GOALS
from app.models.distributions import pois_matrix as _pois_matrix


def _dc_matrix(lam_home: float, lam_away: float, tau: float = 0.0) -> np.ndarray:
    """Dixon-Coles 修正比分矩阵(τ=0 退化为泊松)。"""
    from app.models.dixon_coles.dc import _dc_tau
    
    m = _pois_matrix(lam_home, lam_away)
    for i in range(min(MAX_GOALS, 2)):
        for j in range(min(MAX_GOALS, 2)):
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


def compute_tail_mass(matrix: np.ndarray, max_goals: int = 10) -> dict:
    """计算 score matrix 的 tail mass (opt-1)。
    
    返回:
        raw_mass_10x10: 10x10 截断前的原始质量
        tail_mass: 被截断的尾部质量
        renormalized: 是否经过重新归一化
    """
    m = np.asarray(matrix, dtype=float)
    total_mass = float(m.sum())
    
    # 计算 10x10 范围内的质量
    mass_10x10 = float(m[:max_goals, :max_goals].sum()) if m.shape[0] >= max_goals else total_mass
    
    # 尾部质量 = 总质量 - 10x10 质量
    tail_mass = total_mass - mass_10x10
    
    return {
        "total_mass": round(total_mass, 6),
        "mass_10x10": round(mass_10x10, 6),
        "tail_mass": round(tail_mass, 6),
        "tail_fraction": round(tail_mass / total_mass, 6) if total_mass > 0 else 0.0,
    }


def extract_dc_low_score_probs(matrix: np.ndarray) -> dict:
    """计算 DC 低比分校准指标 (opt-2)。
    
    返回 P(0-0), P(1-0), P(0-1), P(1-1) 的预测概率。
    """
    m = np.asarray(matrix, dtype=float)
    n = min(m.shape[0], 10)
    
    p_00 = float(m[0, 0]) if n > 0 else 0.0
    p_10 = float(m[1, 0]) if n > 1 else 0.0
    p_01 = float(m[0, 1]) if n > 1 else 0.0
    p_11 = float(m[1, 1]) if n > 1 else 0.0
    
    return {
        "p_00": round(p_00, 6),
        "p_10": round(p_10, 6),
        "p_01": round(p_01, 6),
        "p_11": round(p_11, 6),
        "p_low_score": round(p_00 + p_10 + p_01 + p_11, 6),
    }


def extract_nb_tail_probs(matrix: np.ndarray) -> dict:
    """计算 NB 尾部校准指标 (opt-3)。
    
    返回 P(total>=4), P(total>=5) 的预测概率。
    """
    m = np.asarray(matrix, dtype=float)
    n = min(m.shape[0], 10)
    
    p_ge4 = 0.0
    p_ge5 = 0.0
    
    for i in range(n):
        for j in range(n):
            total = i + j
            if total >= 4:
                p_ge4 += m[i, j]
            if total >= 5:
                p_ge5 += m[i, j]
    
    return {
        "p_total_ge4": round(p_ge4, 6),
        "p_total_ge5": round(p_ge5, 6),
    }
