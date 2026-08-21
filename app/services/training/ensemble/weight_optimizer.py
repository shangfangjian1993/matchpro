"""分层权重优化入口(严格DAG顺序 + 拆分 Optimizers)。

Stage 1: OOF λ → Layer-1 weights → fused λ
Stage 2: fused λ → fit τ/φ → Layer-2 weights
Stage 3: shape + GBM → Layer-3 weights
"""
from __future__ import annotations

from dataclasses import dataclass

from .optimizers import DEFAULT_CONFIG, EnsembleTrainingConfig
from .optimizers.dc_parameter import fit_tau
from .optimizers.nb_parameter import fit_phi
from .optimizers.outcome_optimizer import optimize_outcome_weights


@dataclass(frozen=True)
class EnsembleTrainingResult:
    """训练结果(不可变)。"""
    tau: float
    phi: float
    weights: dict
    metadata: dict


def optimize(
    oof_samples: list[dict],
    tau: float | None = None,
    phi: float | None = None,
    config: EnsembleTrainingConfig | None = None,
) -> EnsembleTrainingResult:
    """分层权重学习(严格DAG + 拆分 Optimizers)。"""
    if config is None:
        config = DEFAULT_CONFIG
    
    from app.models.ensemble.weights import (
        optimize_goal_lambda_weights,
        optimize_score_distribution_weights,
    )

    from .member_builder import build_member_samples
    
    # Stage 1: Layer-1 λ weights (Poisson Goal NLL)
    samples_stage1 = build_member_samples(oof_samples, tau=0.0, phi=1e9)
    w_layer1 = optimize_goal_lambda_weights(samples_stage1, shrinkage=config.shrinkage)
    
    # Stage 2: τ/φ on fused λ (使用 learned w_layer1)
    samples_fused = build_member_samples(oof_samples, tau=0.0, phi=1e9, weights=w_layer1)
    if tau is None:
        tau = fit_tau(samples_fused, config=config, use_fused_lambda=True)
    if phi is None:
        phi = fit_phi(samples_fused, config=config)
    
    # Stage 3: Layer-2 shape weights (使用拟合后的 τ/φ 和 learned w_layer1)
    samples_stage2 = build_member_samples(oof_samples, tau, phi, weights=w_layer1)
    w_layer2 = optimize_score_distribution_weights(samples_stage2, shrinkage=config.shrinkage)
    
    # Stage 4: Layer-3 outcome weights (使用 learned shape weights)
    shape_probs = []
    gbm_probs_list = []
    actuals = []
    for s in samples_stage2:
        if "gbm" in s:
            p_pois = s.get("poisson", [0.33, 0.34, 0.33])
            p_dc = s.get("dc", [0.33, 0.34, 0.33])
            p_nb = s.get("nb", [0.33, 0.34, 0.33])
            wp = w_layer2.get("poisson", 0.33)
            wd = w_layer2.get("dc", 0.33)
            wn = w_layer2.get("nb", 0.33)
            wsum = wp + wd + wn or 1.0
            shape_1x2 = [
                (wp * p_pois[0] + wd * p_dc[0] + wn * p_nb[0]) / wsum,
                (wp * p_pois[1] + wd * p_dc[1] + wn * p_nb[1]) / wsum,
                (wp * p_pois[2] + wd * p_dc[2] + wn * p_nb[2]) / wsum,
            ]
            shape_probs.append(shape_1x2)
            gbm_probs_list.append(s["gbm"])
            actuals.append(s["actual"])
    
    if gbm_probs_list and len(gbm_probs_list) == len(actuals):
        shape_weight, gbm_weight = optimize_outcome_weights(
            shape_probs, gbm_probs_list, actuals, config
        )
    else:
        shape_weight, gbm_weight = 1.0, 0.0
    
    # 组合输出(全精度)
    out = {
        "hgbr": w_layer1.get("hgbr", 0.0),
        "elo": w_layer1.get("elo", 0.0),
        "bayes": w_layer1.get("bayes", 0.0),
        "poisson": w_layer2.get("poisson", 0.0),
        "dc": w_layer2.get("dc", 0.0),
        "nb": w_layer2.get("nb", 0.0),
        "shape_weight": shape_weight,
        "gbm_weight": gbm_weight,
        "log_loss": w_layer2.get("log_loss", 0.0),
        "n": len(samples_stage2),
        "shrinkage": config.shrinkage,
        "tau": tau,
        "phi": phi,
    }
    
    metadata = {
        "tau": tau,
        "phi": phi,
        "shape_weight": shape_weight,
        "gbm_weight": gbm_weight,
        "config": config.to_dict(),
    }
    
    return EnsembleTrainingResult(tau=tau, phi=phi, weights=out, metadata=metadata)
