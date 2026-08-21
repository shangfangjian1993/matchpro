"""分层权重优化入口(严格DAG顺序)。

Stage 1: OOF λ → Layer-1 weights → fused λ
Stage 2: fused λ → fit τ/φ → Layer-2 weights
Stage 3: shape + GBM → Layer-3 weights
"""
from __future__ import annotations

import warnings

from . import EnsembleTrainingConfig, DEFAULT_CONFIG
from .optimizers.dc_parameter import fit_tau
from .optimizers.nb_parameter import fit_phi
from .optimizers.outcome_optimizer import optimize_outcome_weights


def optimize(
    oof_samples: list[dict],
    tau: float | None = None,
    phi: float | None = None,
    config: EnsembleTrainingConfig | None = None,
) -> tuple[float, float, dict]:
    """分层权重学习(严格DAG)。
    
    Stage 1: Layer-1 λ weights (Poisson Goal NLL)
    Stage 2: τ/φ on fused λ → Layer-2 shape weights (1X2 LogLoss)
    Stage 3: Layer-3 outcome weights (bounded optimization)
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    from app.models.ensemble import learn_weights
    from .member_builder import build_member_samples
    
    # Stage 1: Layer-1 λ weights
    # (τ/φ 使用默认值,因为 Layer-1 不依赖它们)
    samples_stage1 = build_member_samples(oof_samples, tau=0.0, phi=1e9)
    w_layer1 = learn_weights(samples_stage1, tau=0.0, phi=1e9, shrinkage=config.shrinkage)
    
    # Stage 2: τ/φ on fused λ
    if tau is None:
        # 使用 preliminary weights 计算 fused λ
        tau = fit_tau(samples_stage1, config=config, use_fused_lambda=True)
    if phi is None:
        phi = fit_phi(samples_stage1, config=config)
    
    # Stage 3: Layer-2 shape weights (使用拟合后的 τ/φ)
    samples_stage2 = build_member_samples(oof_samples, tau, phi)
    w_layer2 = learn_weights(samples_stage2, tau, phi, shrinkage=config.shrinkage)
    
    # Stage 4: Layer-3 outcome weights
    shape_probs = [s.get("shape_1x2", s.get("poisson", [0.33, 0.34, 0.33])) for s in samples_stage2]
    gbm_probs_list = [s.get("gbm") for s in samples_stage2 if "gbm" in s]
    actuals = [s["actual"] for s in samples_stage2]
    
    if gbm_probs_list and len(gbm_probs_list) == len(actuals):
        shape_weight, gbm_weight = optimize_outcome_weights(
            shape_probs, gbm_probs_list, actuals, config
        )
    else:
        shape_weight, gbm_weight = 1.0, 0.0
    
    # 组合为 flat 格式(兼容旧接口)
    out = {
        "hgbr": w_layer1.get("hgbr", 0.0),
        "elo": w_layer1.get("elo", 0.0),
        "bayes": w_layer1.get("bayes", 0.0),
        "dc": w_layer2.get("dc", 0.0),
        "nb": w_layer2.get("nb", 0.0),
        "gbm": gbm_weight,
        "log_loss": w_layer2.get("log_loss", 0.0),
        "n": len(samples_stage2),
        "shrinkage": config.shrinkage,
        "tau": tau,
        "phi": phi,
        "shape_weight": shape_weight,
        "gbm_weight": gbm_weight,
    }
    return tau, phi, out
