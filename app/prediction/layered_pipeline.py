"""Layered Prediction Pipeline — 唯一数学真相源。

Production / OOF / Ablation / Replay 全部调用同一个函数,
区别仅在于 data cutoff、model artifact、ablation mask。

数学契约:
  Layer 1: Goal λ Ensemble (HGBR/ELO/Bayes, mask+归一) → fused λ
  Layer 2: Shape Ensemble (Poisson/DC/NB,基于 fused λ) → Goal 1X2
  Layer 3: Outcome GBM → Outcome 1X2
  Layer 4: Prior Blend
  Layer 5: Calibration
"""
from __future__ import annotations

from dataclasses import dataclass, field

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

    None = 该层全部参与(生产模式),
    非空列表 = 只保留指定成员。
    """
    goal_lambda: list[str] | None = None
    score_distribution: list[str] | None = None
    disable_gbm: bool = False
    disable_prior: bool = False
    disable_calibration: bool = False


@dataclass
class LayeredResult:
    """分层计算结果。所有字段都是真实计算结果,无虚假完成度。"""
    fused_lambda: tuple[float, float]
    goal_1x2: tuple[float, float, float]
    shape_1x2: tuple[float, float, float]
    outcome_1x2: tuple[float, float, float]
    final_1x2: tuple[float, float, float]
    score_matrix: np.ndarray
    ablation_mask: AblationMask | None = None
    diagnostics: dict = field(default_factory=dict)


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
    """完整预测流水线(唯一入口)。

    执行: Layer 1 → Layer 2 → Layer 3 (GBM) → Prior → Calibration
    """
    if ablation_mask is None:
        ablation_mask = AblationMask()
    diagnostics: dict = {}

    # to_layered 现在是幂等的(DEFAULT_WEIGHTS 已经是 layered)
    lay = to_layered(weights)
    gl = lay["goal_lambda"]
    sd = lay["score_distribution"]

    # ── Layer 1: Goal λ Ensemble ──
    active_goal = []
    _gl = ablation_mask.goal_lambda  # None = all, [] = none, ["x"] = specific
    if _gl is None or "hgbr" in _gl:
        active_goal.append("hgbr")
    if _gl is None or "elo" in _gl:
        active_goal.append("elo")
    if (_gl is None or "bayes" in _gl) and lam_bh is not None and lam_ba is not None:
        active_goal.append("bayes")

    if not active_goal:
        diagnostics["error"] = "No active goal lambda members"
        return None

    active_weights = {k: gl.get(k, 0.0) for k in active_goal}
    wsum = sum(active_weights.values())
    if wsum <= 0:
        diagnostics["error"] = "Goal lambda weights sum to 0"
        return None
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

    # ── Layer 2: Shape Ensemble (基于 fused λ) ──
    active_shape = []
    _sd = ablation_mask.score_distribution
    if _sd is None or "poisson" in _sd:
        active_shape.append("poisson")
    if _sd is None or "dc" in _sd:
        active_shape.append("dc")
    if _sd is None or "nb" in _sd:
        active_shape.append("nb")

    if not active_shape:
        active_shape = ["poisson"]

    active_sd = {k: sd.get(k, 0.0) for k in active_shape}
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

    shape_1x2 = fuse_probs(probs, active_sd)

    # Score matrix for xG / calibration
    from app.models.ensemble import _dc_matrix, _nb_matrix, _pois_matrix
    matrices = {}
    if "poisson" in probs:
        matrices["poisson"] = _pois_matrix(fh, fa)
    if "dc" in probs:
        matrices["dc"] = _dc_matrix(fh, fa, tau)
    if "nb" in probs:
        matrices["nb"] = _nb_matrix(fh, fa, phi)
    score_matrix = fuse_score_matrix(matrices, active_sd)

    # ── Layer 3: Outcome GBM ──
    outcome_1x2 = shape_1x2
    if gbm_probs is not None and not ablation_mask.disable_gbm:
        from app.models.ensemble.fusion import fuse_goal_outcome
        outcome_1x2 = fuse_goal_outcome(shape_1x2, gbm_probs, weights)

    # ── Layer 4: Prior ──
    final_1x2 = outcome_1x2
    if prior_context is not None and not ablation_mask.disable_prior:
        from app.prediction.prior_blend import blend_matrix as _prod_blend
        league_id = prior_context.get("league_id")
        match_dt = prior_context.get("match_dt")
        raw_matrix = prior_context.get("raw_matrix", score_matrix)
        if league_id is not None and match_dt is not None:
            try:
                m2, _info = _prod_blend(league_id, match_dt, list(outcome_1x2), np.asarray(raw_matrix))
                if m2 is not None:
                    final_1x2 = tuple(float(x) for x in (
                        np.asarray(m2)[np.tril_indices(len(m2), -1)].sum(),
                        np.trace(np.asarray(m2)),
                        np.asarray(m2)[np.triu_indices(len(m2), 1)].sum(),
                    ))
                diagnostics["prior"] = "applied"
            except Exception as e:
                diagnostics["prior"] = f"fallback: {type(e).__name__}"
                diagnostics["degraded"] = True

    # ── Layer 5: Calibration ──
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
            diagnostics["calibration"] = "applied"
        except Exception as e:
            diagnostics["calibration"] = f"fallback: {type(e).__name__}"
            diagnostics["degraded"] = True

    return LayeredResult(
        fused_lambda=(fh, fa),
        goal_1x2=shape_1x2,
        shape_1x2=shape_1x2,
        outcome_1x2=outcome_1x2,
        final_1x2=final_1x2,
        score_matrix=score_matrix,
        ablation_mask=ablation_mask,
        diagnostics=diagnostics,
    )


ABLATION_MASKS = {
    # A-E: 只执行到 Layer 2(禁用 GBM/Prior/Calibration)
    "A": AblationMask(goal_lambda=["hgbr"], score_distribution=["poisson"],
                     disable_gbm=True, disable_prior=True, disable_calibration=True),
    "B": AblationMask(goal_lambda=["hgbr", "elo"], score_distribution=["poisson"],
                     disable_gbm=True, disable_prior=True, disable_calibration=True),
    "C": AblationMask(goal_lambda=["hgbr", "elo", "bayes"], score_distribution=["poisson"],
                     disable_gbm=True, disable_prior=True, disable_calibration=True),
    "D": AblationMask(score_distribution=["poisson", "dc"],
                     disable_gbm=True, disable_prior=True, disable_calibration=True),
    "E": AblationMask(score_distribution=["poisson", "dc", "nb"],
                     disable_gbm=True, disable_prior=True, disable_calibration=True),
    # F: + GBM
    "F": AblationMask(disable_prior=True, disable_calibration=True),
    # G: + Prior (无 Calibration)
    "G": AblationMask(disable_calibration=True),
    # H: 全量 (Prior + Calibration)
    "H": AblationMask(),
}
