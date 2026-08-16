"""概率基础(审查 §36:ensemble 拆分)。"""
from __future__ import annotations

from app.models.distributions import matrix_to_probs, pois_matrix


def match_probs(lam_home: float, lam_away: float) -> tuple[float, float, float]:
    """双泊松卷积 → (home_win, draw, away_win)——统一走 distributions。"""
    return matrix_to_probs(pois_matrix(lam_home, lam_away))
