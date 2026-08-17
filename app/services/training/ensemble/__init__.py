"""Ensemble 权重学习包(审查九 P1-9 拆分)。

模块:oof_generator(样本)/ temporary_trainer(临时模型)/ member_builder(成员)/
weight_optimizer(τφ+SLSQP)/ artifact_writer(落盘)。
"""
from .oof_generator import (K_SEG, MIN_OOF_SAMPLES as MIN_OOF_SAMPLES,
                             SAMPLE_PER_SEG as SAMPLE_PER_SEG, generate)


def run_all(leagues, verbose: bool = True) -> dict:
    """对 5 联赛执行 OOF 权重学习;返回 (weights, params, meta)。"""
    from app.core.paths import ARTIFACTS_DIR as _AD

    from .artifact_writer import write_all
    from .weight_optimizer import fit_tau, optimize

    _ens_dir = str(_AD / "ensemble")
    weights_out, params_out, meta_out = {}, {}, {}
    for lt, league, matches in leagues:
        oof_samples = generate(lt, league, matches, verbose=verbose)
        if not oof_samples:
            continue
        tau = fit_tau(oof_samples)
        phi, w, samples = optimize(oof_samples, tau, phi=None, shrinkage=0.15)
        params_out[lt.value] = {"tau": tau, "phi": phi}
        # optimize 内已 learn_weights(shrinkage);samples 用于记录
        weights_out[lt.value] = {k: round(v, 4) for k, v in w.items()
                                 if k not in ("log_loss", "n", "shrinkage")}
        meta_out[lt.value] = {"log_loss": round(w["log_loss"], 4), "n": len(samples),
                              "segments": K_SEG, "method": "time-segmented-oof",
                              "k_seg": K_SEG, "shrinkage": 0.15}
        if verbose:
            print(f"  {lt.value}: τ={tau:.3f} φ={phi:.1f} "
                  f"w={ {k: round(v,3) for k,v in weights_out[lt.value].items()} } "
                  f"ll={w['log_loss']:.4f} n={len(samples)}", flush=True)
    write_all(weights_out, params_out, meta_out, _ens_dir)
    return weights_out, params_out, meta_out
