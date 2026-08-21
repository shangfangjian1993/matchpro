"""Ensemble 权重学习包。

模块:oof_generator(样本)/ temporary_trainer(临时模型)/ member_builder(成员)/
weight_optimizer(τφ+SLSQP)/ artifact_writer(落盘)/ artifact(ProductionArtifact)。
"""
from .oof_generator import K_SEG, generate
from .oof_generator import MIN_OOF_SAMPLES as MIN_OOF_SAMPLES
from .oof_generator import SAMPLE_PER_SEG as SAMPLE_PER_SEG


def run_all(leagues, verbose: bool = True) -> dict:
    """对 5 联赛执行 OOF 权重学习;返回 (weights, params, meta)。
    
    P0-1 FIX: 同时创建和写入 ProductionArtifact。
    """
    from app.core.paths import ARTIFACTS_DIR as _AD
    
    from .artifact import create_production_artifact
    from .artifact_writer import write_all, write_production_artifact
    from .weight_optimizer import optimize
    
    _ens_dir = str(_AD / "ensemble")
    weights_out, params_out, meta_out = {}, {}, {}
    artifacts_out = {}
    
    for lt, league, matches in leagues:
        oof_samples = generate(lt, league, matches, verbose=verbose)
        if not oof_samples:
            continue
        result = optimize(oof_samples)
        w = result.weights
        params_out[lt.value] = {"tau": result.tau, "phi": result.phi}
        weights_out[lt.value] = {
            k: round(v, 4)
            for k, v in w.items()
            if k not in ("log_loss", "n", "shrinkage")
        }
        meta_out[lt.value] = {
            "log_loss": round(w["log_loss"], 4),
            "n": w.get("n", 0),
            "segments": K_SEG,
            "method": "time-segmented-oof",
            "k_seg": K_SEG,
            "shrinkage": w.get("shrinkage", 0.15),
            "config": result.metadata.get("config", {}),
        }
        
        # P0-1: 创建 ProductionArtifact
        artifact = create_production_artifact(
            weights=w,
            tau=result.tau,
            phi=result.phi,
            oof_n=w.get("n", 0),
            shrinkage=w.get("shrinkage", 0.15),
        )
        artifacts_out[lt.value] = artifact
        
        if verbose:
            print(
                f"  {lt.value}: τ={result.tau:.3f} φ={result.phi:.1f} "
                f"w={ {k: round(v, 3) for k, v in weights_out[lt.value].items()} } "
                f"ll={w['log_loss']:.4f} n={w.get('n', 0)} "
                f"hash={artifact.model_hash()[:8]}",
                flush=True,
            )
    
    # 写入旧格式(兼容)
    write_all(weights_out, params_out, meta_out, _ens_dir)
    
    # P0-1: 写入 ProductionArtifact (主格式)
    for lt, artifact in artifacts_out.items():
        write_production_artifact(artifact, _ens_dir)
    
    return weights_out, params_out, meta_out
