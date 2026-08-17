"""Probability Invariants(审查十 P1-4):统一概率不变量检查。

所有输出(1X2/score matrix/Top5/xG)必须满足:
  0 ≤ p ≤ 1;sum(1X2) = 1;matrix.sum() = 1;matrix 边缘 == 1X2;
  xG == matrix 期望;Top5 概率 ≤ 1 且递减。

engine 核心路径已内联 _check_matrix/_check_probs(抛 CorePredictionError);
本模块为统一入口(测试/审计/快照校验共用)。
"""
from __future__ import annotations

import numpy as np

EPS = 1e-4


def check_probability_vector(probs, name: str = "1X2") -> list[str]:
    """0 ≤ p ≤ 1 且 sum=1。返回违规列表(空 = 通过)。"""
    v = np.asarray(probs, dtype=float)
    vios = []
    if v.size == 0:
        return [f"{name}: 空向量"]
    if not np.all(np.isfinite(v)):
        vios.append(f"{name}: 含 NaN/Inf")
    if np.any(v < -EPS) or np.any(v > 1 + EPS):
        vios.append(f"{name}: 越界 [0,1]: {v}")
    if not np.isclose(v.sum(), 1.0, atol=EPS):
        vios.append(f"{name}: 和={v.sum():.6f} ≠ 1")
    return vios


def check_matrix(matrix, name: str = "score_matrix") -> list[str]:
    """非负/有限/和为 1。返回违规列表。"""
    m = np.asarray(matrix, dtype=float)
    vios = []
    if m.size == 0:
        return [f"{name}: 空矩阵"]
    if not np.all(np.isfinite(m)):
        vios.append(f"{name}: 含 NaN/Inf")
    if np.any(m < -EPS):
        vios.append(f"{name}: 含负概率")
    if not np.isclose(m.sum(), 1.0, atol=EPS):
        vios.append(f"{name}: 和={m.sum():.6f} ≠ 1")
    return vios


def check_matrix_marginal(matrix, probs, name: str = "score_matrix") -> list[str]:
    """矩阵 1X2 边缘 == 输出 1X2。"""
    m = np.asarray(matrix, dtype=float)
    if m.size == 0:
        return [f"{name}: 空矩阵,无法校验边缘"]
    hw = float(m[np.tril_indices_from(m, -1)].sum())
    dr = float(np.trace(m))
    aw = float(m[np.triu_indices_from(m, 1)].sum())
    p = np.asarray(probs, dtype=float)
    vios = []
    for got, want, label in ((hw, p[0], "home"), (dr, p[1], "draw"), (aw, p[2], "away")):
        if abs(got - want) > EPS:
            vios.append(f"{name} 边缘 {label}: {got:.6f} ≠ {want:.6f}")
    return vios


def check_xg(matrix, lam_home, lam_away, tol: float = 0.05) -> list[str]:
    """xG(矩阵期望)≈ fused λ。"""
    m = np.asarray(matrix, dtype=float)
    if m.size == 0:
        return ["矩阵为空,无法校验 xG"]
    grid = np.arange(m.shape[0], dtype=float)
    xg_h = float((m * grid[:, None]).sum())
    xg_a = float((m * grid[None, :]).sum())
    vios = []
    if abs(xg_h - lam_home) > tol:
        vios.append(f"xG_home {xg_h:.3f} ≠ λ_home {lam_home:.3f}")
    if abs(xg_a - lam_away) > tol:
        vios.append(f"xG_away {xg_a:.3f} ≠ λ_away {lam_away:.3f}")
    return vios


def check_top_scores(top_scores) -> list[str]:
    """Top5 概率:0-1、递减、和 ≤ 1。"""
    vios = []
    if not top_scores:
        return ["top_scores 为空"]
    prev = 1.01
    total = 0.0
    for i, t in enumerate(top_scores):
        p = float(t.get("probability", 0))
        total += p
        if not (0.0 <= p <= 1.0 + EPS):
            vios.append(f"top_scores[{i}] 概率越界: {p}")
        if p > prev + EPS:
            vios.append(f"top_scores[{i}] 概率未递减: {p} > {prev}")
        prev = p
    if total > 1.0 + EPS:
        vios.append(f"top_scores 概率和 {total:.4f} > 1")
    return vios


def validate_prediction(result: dict) -> list[str]:
    """对预测 result 做全套不变量检查;返回违规列表(空 = 全部通过)。"""
    vios = []
    probs = [result.get("home_win_probability"), result.get("draw_probability"),
             result.get("away_win_probability")]
    if any(p is None for p in probs):
        vios.append("1X2 概率缺失")
    else:
        vios += check_probability_vector(probs)
        matrix = result.get("_internal", {}).get("fused_matrix")
        if matrix is not None:
            vios += check_matrix(matrix)
            vios += check_matrix_marginal(matrix, probs)
        xg = result.get("expected_xg")
        if xg and len(xg) == 2:
            vios += check_xg(matrix if matrix is not None else np.zeros((10, 10)),
                             xg[0], xg[1], tol=0.15)
    vios += check_top_scores(result.get("top_scores") or [])
    return vios
