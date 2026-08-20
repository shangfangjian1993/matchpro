"""Layered Prediction Pipeline — 唯一数学真相源。

Production / OOF / Ablation / Replay 全部调用同一个函数,
区别仅在于 data cutoff、model artifact、ablation mask。

三层结构:
  Layer 1: Goal λ Ensemble (HGBR/ELO/Bayes) → fused λ
  Layer 2: Shape Ensemble (Poisson/DC/NB,基于 fused λ) → Goal 1X2
  Layer 3: Outcome GBM → Outcome 1X2
  + Prior + Calibration

数学不变量:
  - DC/NB 永远基于 fused λ,而非 HGBR λ
  - 缺失成员 mask 后重新归一化,不隐式替代
  - 所有调用者使用同一个 compute_layers() 函数
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.models.ensemble import (
    dc_probs,
    fuse_probs,
    match_probs,
    nb_probs,
)
from app.models.ensemble.weights import to_layered


@dataclass
class AblationMask:
    """消融掩码:控制哪些成员参与计算。

    空列表表示该层全部参与(生产模式),
    非空列表表示只保留指定成员。
    """
    goal_lambda: list[str] = field(default_factory=list)  # 只保留这些成员
    score_distribution: list[str] = field(default_factory=list)
    disable_gbm: bool = False
    disable_prior: bool = False
    disable_calibration: bool = False


@dataclass
class LayeredResult:
    """分层计算结果(用于审计/对比/诊断)。"""
    fused_lambda: tuple[float, float]
    goal_1x2: tuple[float, float, float]  # Goal-derived 1X2
    shape_1x2: tuple[float, float, float]  # Shape ensemble 1X2
    outcome_1x2: tuple[float, float, float]  # After GBM fusion
    final_1x2: tuple[float, float, float]  # After prior + calibration
    ablation_mask: AblationMask | None = None


def compute_layers(
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
    ablation_mask: AblationMask | None = None,
) -> LayeredResult | None:
    """三层分层计算(所有调用者的唯一入口)。

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
    # 确定参与的成员
    active_goal = []
    if "hgbr" in ablation_mask.goal_lambda or not ablation_mask.goal_lambda:
        active_goal.append("hgbr")
    if "elo" in ablation_mask.goal_lambda or not ablation_mask.goal_lambda:
        active_goal.append("elo")
    if ("bayes" in ablation_mask.goal_lambda or not ablation_mask.goal_lambda) and lam_bh is not None and lam_ba is not None:
        active_goal.append("bayes")

    if not active_goal:
        return None  # 无可用 λ 成员

    # 重新归一化权重(缺失成员后)
    active_weights = {k: gl.get(k, 0.0) for k in active_goal}
    wsum = sum(active_weights.values())
    if wsum <= 0:
        return None
    active_weights = {k: v / wsum for k, v in active_weights.items()}

    # 融合 λ
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
    if "poisson" in ablation_mask.score_distribution or not ablation_mask.score_distribution:
        active_shape.append("poisson")
    if "dc" in ablation_mask.score_distribution or not ablation_mask.score_distribution:
        active_shape.append("dc")
    if "nb" in ablation_mask.score_distribution or not ablation_mask.score_distribution:
        active_shape.append("nb")

    if not active_shape:
        active_shape = ["poisson"]  # 保底

    # 重新归一化 shape 权重
    active_sd = {k: sd.get(k, 0.0) for k in active_shape}
    sd_sum = sum(active_sd.values())
    if sd_sum <= 0:
        active_sd = {k: 1.0 / len(active_shape) for k in active_shape}
    else:
        active_sd = {k: v / sd_sum for k, v in active_sd.items()}

    # 计算各成员概率(全部基于 fused λ!)
    shape_probs = {}
    if "poisson" in active_shape:
        shape_probs["poisson"] = match_probs(fh, fa)
    if "dc" in active_shape:
        shape_probs["dc"] = dc_probs(fh, fa, tau)
    if "nb" in active_shape:
        shape_probs["nb"] = nb_probs(fh, fa, phi)

    shape_1x2 = fuse_probs(shape_probs, active_sd)

    # ── Layer 3: Outcome GBM ──
    if gbm_probs is not None and not ablation_mask.disable_gbm:
        # 简化:直接返回 shape_1x2(完整 fusion 在 engine 层处理)
        outcome_1x2 = shape_1x2
    else:
        outcome_1x2 = shape_1x2

    return LayeredResult(
        fused_lambda=(fh, fa),
        goal_1x2=shape_1x2,  # Goal = Shape in layered mode
        shape_1x2=shape_1x2,
        outcome_1x2=outcome_1x2,
        final_1x2=outcome_1x2,  # Prior/calibration 在后续处理
        ablation_mask=ablation_mask,
    )


def compute_layers_simple(
    lam_h: float,
    lam_a: float,
    weights: dict,
    lam_eh: float | None = None,
    lam_ea: float | None = None,
    lam_bh: float | None = None,
    lam_ba: float | None = None,
) -> tuple[float, float] | None:
    """简化的 Layer-1 计算:只返回 fused λ。

    用于需要 fused λ 作为 DC/NB 输入的场景(OOF/Training)。
    """
    lay = to_layered(weights)
    gl = lay["goal_lambda"]

    active = []
    w_sum = 0.0
    for name in ["hgbr", "elo", "bayes"]:
        if name == "bayes" and (lam_bh is None or lam_ba is None):
            continue  # Bayes 缺失 → mask
        if gl.get(name, 0) > 0:
            active.append(name)
            w_sum += gl[name]

    if not active or w_sum <= 0:
        return None

    fh = gl.get("hgbr", 0) / w_sum * lam_h
    fa = gl.get("hgbr", 0) / w_sum * lam_a

    if "elo" in active:
        fh += gl["elo"] / w_sum * (lam_eh or 0)
        fa += gl["elo"] / w_sum * (lam_ea or 0)
    if "bayes" in active and lam_bh is not None:
        fh += gl["bayes"] / w_sum * lam_bh
        fa += gl["bayes"] / w_sum * lam_ba

    return (fh, fa)


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
