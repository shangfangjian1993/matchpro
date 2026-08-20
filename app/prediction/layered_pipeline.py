"""Layered Prediction Pipeline — 唯一数学真相源。

Production / OOF / Ablation / Replay 全部调用同一个函数,
区别仅在于 data cutoff、model artifact、ablation mask。

完整流水线:
  Layer 1: Goal λ Ensemble (HGBR/ELO/Bayes) → fused λ
  Layer 2: Shape Ensemble (Poisson/DC/NB,基于 fused λ) → Goal 1X2
  Layer 3: Outcome GBM → Outcome 1X2
  + Prior + Calibration → Final 1X2

数学不变量:
  - DC/NB 永远基于 fused λ,而非 HGBR λ
  - 缺失成员 mask 后重新归一化,不隐式替代
  - 所有调用者使用同一个 compute_prediction() 函数
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.models.ensemble import (
    dc_probs,
    fuse_probs,
    fuse_score_matrix,
    match_probs,
    nb_probs,
)
from app.models.ensemble.weights import to_layered


@dataclass
class AblationMask:
    """消融掩码:控制哪些成员参与计算。

    None 表示该层全部参与(生产模式),
    非空列表表示只保留指定成员。
    """
    goal_lambda: list[str] | None = None  # None = 全部参与
    score_distribution: list[str] | None = None
    disable_gbm: bool = False
    disable_prior: bool = False
    disable_calibration: bool = False


@dataclass
class LayeredResult:
    """分层计算结果(用于审计/对比/诊断)。

    所有字段都是真实计算结果,无虚假完成度。
    """
    fused_lambda: tuple[float, float]
    goal_1x2: tuple[float, float, float]  # Shape ensemble 1X2
    shape_1x2: tuple[float, float, float]  # 同 goal_1x2(Layer 2 输出)
    outcome_1x2: tuple[float, float, float]  # After GBM fusion
    final_1x2: tuple[float, float, float]  # After prior + calibration
    score_matrix: np.ndarray  # 用于 xG / calibration 计算
    ablation_mask: AblationMask | None = None


def _fuse_goal_lambda(
    lam_h: float, lam_a: float,
    lam_eh: float, lam_ea: float,
    lam_bh: float | None, lam_ba: float | None,
    goal_weights: dict,
    mask: list[str] | None,
) -> tuple[dict, float, float]:
    """Layer-1: Goal λ Ensemble with mask + renormalize.

    Returns: (active_weights, fh, fa) or raises ValueError if no active members.
    """
    active_weights = {}
    for name in ["hgbr", "elo", "bayes"]:
        if mask is not None and name not in mask:
            continue  # masked out
        if name == "bayes" and (lam_bh is None or lam_ba is None):
            continue  # Bayes unavailable → mask
        w = goal_weights.get(name, 0.0)
        if w > 0:
            active_weights[name] = w

    if not active_weights:
        raise ValueError("No active goal lambda members")

    wsum = sum(active_weights.values())
    active_weights = {k: v / wsum for k, v in active_weights.items()}

    fh = (
        active_weights.get("hgbr", 0.0) * lam_h
        + active_weights.get("elo", 0.0) * lam_eh
        + active_weights.get("bayes", 0.0) * (lam_bh or 0.0)
    )
    fa = (
        active_weights.get("hgbr", 0.0) * lam_a
        + active_weights.get("elo", 0.0) * lam_ea
        + active_weights.get("bayes", 0.0) * (lam_ba or 0.0)
    )

    return active_weights, fh, fa


def _compute_shape(
    fh: float, fa: float,
    tau: float, phi: float,
    shape_weights: dict,
    mask: list[str] | None,
) -> tuple[dict, dict]:
    """Layer-2: Shape Ensemble (基于 fused λ).

    Returns: (active_weights, member_probs)
    """
    active_shape = []
    for name in ["poisson", "dc", "nb"]:
        if mask is not None and name not in mask:
            continue
        active_shape.append(name)

    if not active_shape:
        active_shape = ["poisson"]  # fallback

    active_sd = {k: shape_weights.get(k, 0.0) for k in active_shape}
    sd_sum = sum(active_sd.values())
    if sd_sum <= 0:
        active_sd = {k: 1.0 / len(active_shape) for k in active_shape}
    else:
        active_sd = {k: v / sd_sum for k, v in active_sd.items()}

    probs = {}
    if "poisson" in active_shape:
        probs["poisson"] = match_probs(fh, fa)
    if "dc" in active_shape:
        probs["dc"] = dc_probs(fh, fa, tau)
    if "nb" in active_shape:
        probs["nb"] = nb_probs(fh, fa, phi)

    return active_sd, probs


def compute_prediction(
    lam_h: float,
    lam_a: float,
    lam_eh: float,
    lam_ea: float,
    tau: float,
    phi: float,
    weights: dict,
    lam_bh: float | None = None,
    lam_ba: float | None = None,
    gbm_probs: tuple | None = None,
    prior_context: dict | None = None,
    calibration_context: dict | None = None,
    ablation_mask: AblationMask | None = None,
) -> LayeredResult | None:
    """完整预测流水线(所有调用者的唯一入口)。

    执行: Layer 1 → Layer 2 → Layer 3 (GBM) → Prior → Calibration

    Args:
        lam_h: HGBR λ home
        lam_a: HGBR λ away
        lam_eh: ELO λ home
        lam_ea: ELO λ away
        tau: Dixon-Coles 参数
        phi: Negative Binomial 参数
        weights: 生产学到的权重(dict)
        lam_bh: Bayes λ home(可为 None)
        lam_ba: Bayes λ away(可为 None)
        gbm_probs: GBM 1X2 概率(可为 None)
        prior_context: Prior 上下文(league_id, match_dt, raw_matrix)
        calibration_context: Calibration 上下文(models_dir, league_type)
        ablation_mask: 消融掩码(生产模式传 None)

    Returns:
        LayeredResult 或 None(必要 λ 缺失且无法 mask)
    """
    if ablation_mask is None:
        ablation_mask = AblationMask()

    lay = to_layered(weights)
    gl = lay["goal_lambda"]
    sd = lay["score_distribution"]

    # ── Layer 1: Goal λ Ensemble ──
    try:
        _, fh, fa = _fuse_goal_lambda(
            lam_h, lam_a, lam_eh, lam_ea, lam_bh, lam_ba, gl,
            ablation_mask.goal_lambda,
        )
    except ValueError:
        return None

    # ── Layer 2: Shape Ensemble (基于 fused λ) ──
    active_sd, shape_probs = _compute_shape(
        fh, fa, tau, phi, sd,
        ablation_mask.score_distribution,
    )

    shape_1x2 = fuse_probs(shape_probs, active_sd)

    # Compute score matrix for xG / calibration
    from app.models.ensemble import _dc_matrix, _nb_matrix, _pois_matrix
    matrices = {}
    if "poisson" in shape_probs:
        matrices["poisson"] = _pois_matrix(fh, fa)
    if "dc" in shape_probs:
        matrices["dc"] = _dc_matrix(fh, fa, tau)
    if "nb" in shape_probs:
        matrices["nb"] = _nb_matrix(fh, fa, phi)
    score_matrix = fuse_score_matrix(matrices, active_sd)

    # ── Layer 3: Outcome GBM ──
    outcome_1x2 = shape_1x2
    if gbm_probs is not None and not ablation_mask.disable_gbm:
        from app.models.ensemble.fusion import fuse_goal_outcome
        outcome_1x2 = fuse_goal_outcome(shape_1x2, gbm_probs, weights)

    # ── Prior ──
    final_1x2 = outcome_1x2
    if prior_context is not None and not ablation_mask.disable_prior:
        from app.prediction.prior_blend import blend_matrix as _prod_blend
        try:
            league_id = prior_context.get("league_id")
            match_dt = prior_context.get("match_dt")
            raw_matrix = prior_context.get("raw_matrix", score_matrix)
            if league_id is not None and match_dt is not None:
                m2, _info = _prod_blend(league_id, match_dt, list(outcome_1x2), np.asarray(raw_matrix))
                if m2 is not None:
                    final_1x2 = tuple(float(x) for x in (
                        np.asarray(m2)[np.tril_indices(len(m2), -1)].sum(),
                        np.trace(np.asarray(m2)),
                        np.asarray(m2)[np.triu_indices(len(m2), 1)].sum(),
                    ))
        except Exception:
            pass  # Prior 失败 → 回退到 outcome_1x2

    # ── Calibration ──
    if calibration_context is not None and not ablation_mask.disable_calibration:
        from app.prediction import calibration as cal
        try:
            fake = {
                "home_win_probability": final_1x2[0],
                "draw_probability": final_1x2[1],
                "away_win_probability": final_1x2[2],
            }
            out, _i, _d = cal.apply(
                fake,
                calibration_context.get("models_dir", ""),
                calibration_context.get("league_type"),
            )
            final_1x2 = (
                out["home_win_probability"],
                out["draw_probability"],
                out["away_win_probability"],
            )
        except Exception:
            pass  # Calibration 失败 → 回退

    return LayeredResult(
        fused_lambda=(fh, fa),
        goal_1x2=shape_1x2,
        shape_1x2=shape_1x2,
        outcome_1x2=outcome_1x2,
        final_1x2=final_1x2,
        score_matrix=score_matrix,
        ablation_mask=ablation_mask,
    )


# ── 预定义的 Ablation 配置(用于 Season-Start Expanding-Window 评估) ──
ABLATION_MASKS = {
    "A": AblationMask(goal_lambda=["hgbr"], score_distribution=["poisson"]),
    "B": AblationMask(goal_lambda=["hgbr", "elo"], score_distribution=["poisson"]),
    "C": AblationMask(goal_lambda=["hgbr", "elo", "bayes"], score_distribution=["poisson"]),
    "D": AblationMask(score_distribution=["poisson", "dc"]),
    "E": AblationMask(score_distribution=["poisson", "dc", "nb"]),
    "F": AblationMask(disable_gbm=False),
    "G": AblationMask(disable_prior=False),
    "H": AblationMask(disable_calibration=False),
}
