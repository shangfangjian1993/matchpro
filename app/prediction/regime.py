"""Regime Detector + Score Matrix 级调整(审查九 P0-3/P1-12)。

P0-3:Prior Blend 只改 1X2 而 score matrix 不变 → 两套输出统计不一致。
修复:先算目标 1X2(α·P_model + (1-α)·近期频率),再用 IPF(迭代比例拟合)
调整 score matrix 使其边缘 = 目标 1X2,所有输出(Top5/O/U/BTTS/xG/1X2)
从同一最终矩阵导出 —— 彻底消除"1X2 与比分分布不一致"。

P1-12:Regime Detector —— 检测平局率/进球率/主场优势漂移,输出 regime
标签;α 动态化:稳定赛季 α≈0.85,剧烈 shift α≈0.55(审查十三)。
"""
from __future__ import annotations

import numpy as np


# ── IPF:按 1X2 类别边缘调整 score matrix ──────────────────────────────────
def ipf_to_target(matrix: np.ndarray, target: tuple[float, float, float],
                  max_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    """迭代调整矩阵使 1X2 边缘 = target,保持类内(胜/平/负格点)相对结构。

    matrix: 10x10 概率矩阵;target = (home_win, draw, away_win)。
    每轮对下三角/对角/上三角格点施加同一乘数(类内比例不变),归一化后
    重复,直到边缘收敛 —— 标准 IPF 的类别边缘变体。
    """
    m = np.asarray(matrix, dtype=float).copy()
    s = m.sum()
    if s <= 0:
        return m
    m = m / s
    n = len(m)
    tril = np.tril(np.ones((n, n), dtype=bool), -1)
    diag = np.eye(n, dtype=bool)
    triu = np.triu(np.ones((n, n), dtype=bool), 1)
    th, td, ta = (float(x) for x in target)
    for _ in range(max_iter):
        ph = m[tril].sum()
        pd_ = np.trace(m)
        pa = m[triu].sum()
        if (abs(ph - th) < tol and abs(pd_ - td) < tol and abs(pa - ta) < tol):
            break
        mult = np.ones_like(m)
        mult[tril] = th / max(ph, 1e-12)
        mult[diag] = td / max(pd_, 1e-12)
        mult[triu] = ta / max(pa, 1e-12)
        m = m * mult
        m = m / m.sum()
    return m


# ── Regime Detector(审查 P1-12)────────────────────────────────────────────
REGIMES = ("NORMAL", "HIGH_DRAW", "LOW_SCORING", "HIGH_SCORING",
           "HOME_ADV_SHIFT", "STRENGTH_COMPRESSED")


def detect(league_id, cutoff_dt, window: int = 100,
           baseline_window: int = 1000, min_samples: int = 50) -> dict:
    """检测截止该场的 regime 漂移(draw/goal/home-adv)。

    返回 {regime, shift_score, draw_rate, goal_rate, home_rate, baseline_*}。
    shift_score ∈ [0, 1]:0=无漂移,1=剧烈漂移。
    """
    from app.api.db import Match
    rows = (Match.query.filter(
                Match.league_id == league_id,
                Match.match_status == "finished",
                Match.match_date < cutoff_dt)
            .order_by(Match.match_date.desc()).limit(window + baseline_window).all())
    if len(rows) < min_samples:
        return {"regime": "NORMAL", "shift_score": 0.0, "sample": len(rows)}

    def _rates(ms):
        n = len(ms)
        if n == 0:
            return 0.0, 0.0, 0.0
        h = sum(1 for m in ms if (m.home_goals or 0) > (m.away_goals or 0))
        d = sum(1 for m in ms if (m.home_goals or 0) == (m.away_goals or 0))
        goals = sum((m.home_goals or 0) + (m.away_goals or 0) for m in ms) / n
        return h / n, d / n, goals

    recent = rows[:window]
    baseline = rows[window:window + baseline_window]
    if len(baseline) < min_samples:
        baseline = rows[min(window, len(rows) - min_samples):]
    rh, rd, rg = _rates(recent)
    bh, bd, bg = _rates(baseline)
    # 漂移评分:各维度偏离的归一化和(5pp 平局率 = 0.5,0.5 球 = 0.5)
    shift = max(
        abs(rd - bd) / 0.10,      # 平局率 ±10pp 满量程
        abs(rg - bg) / 0.60,      # 场均进球 ±0.6 满量程
        abs(rh - bh) / 0.12,      # 主胜率 ±12pp 满量程
    )
    shift = float(min(1.0, shift))
    if shift < 0.15:
        regime = "NORMAL"
    elif rd - bd > 0.03:
        regime = "HIGH_DRAW"
    elif rg - bg < -0.25:
        regime = "LOW_SCORING"
    elif rg - bg > 0.25:
        regime = "HIGH_SCORING"
    elif abs(rh - bh) > 0.05:
        regime = "HOME_ADV_SHIFT"
    else:
        regime = "NORMAL"
    return {"regime": regime, "shift_score": shift,
            "draw_rate": round(rd, 4), "goal_rate": round(rg, 3),
            "home_rate": round(rh, 4),
            "base_draw": round(bd, 4), "base_goal": round(bg, 3),
            "base_home": round(bh, 4), "sample": len(recent)}


def dynamic_alpha(shift_score: float, alpha_stable: float = 0.85,
                  alpha_shift: float = 0.55) -> float:
    """审查十三:α 动态 —— 稳定赛季 ≈0.85,剧烈 regime shift ≈0.55。"""
    return float(np.clip(
        alpha_stable - (alpha_stable - alpha_shift) * shift_score,
        alpha_shift, alpha_stable))
