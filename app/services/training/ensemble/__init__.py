"""Ensemble 权重学习包。"""
from .oof_generator import K_SEG, generate
from .oof_generator import MIN_OOF_SAMPLES as MIN_OOF_SAMPLES
from .oof_generator import SAMPLE_PER_SEG as SAMPLE_PER_SEG


def run_all(leagues, verbose: bool = True) -> dict:
 """对 5 联赛执行 OOF 权重学习;返回 (weights, params, meta)。
 
 : 每个联赛写入各自目录,不会覆盖。
 : 传递 training_cutoff(真实数据截止时间)。
 : 传递真实 K_SEG。
 """
 from app.core.paths import ARTIFACTS_DIR as _AD
 
 from .artifact import create_production_artifact
 from .artifact_writer import write_all, write_production_artifact
 from .weight_optimizer import optimize
 
 _ens_dir = str(_AD / "ensemble")
 weights_out, params_out, meta_out = {}, {}, {}
 
 for lt, league, matches in leagues:
 oof_samples = generate(lt, league, matches, verbose=verbose)
 if not oof_samples:
 continue
 result = optimize(oof_samples)
 w = result.weights
 params_out[lt.value] = {"tau": result.tau, "phi": result.phi}
 weights_out[lt.value] = {
 k: v for k, v in w.items() 
 if k not in ("log_loss", "n", "shrinkage")
 }
 meta_out[lt.value] = {
 "log_loss": w["log_loss"],
 "n": w.get("n", 0),
 "segments": K_SEG,
 "method": "time-segmented-oof",
 "k_seg": K_SEG,
 "shrinkage": w.get("shrinkage", 0.15),
 "config": result.metadata.get("config", {}),
 }
 
 
 
 
 artifact = create_production_artifact(
 league=lt.value,
 weights=w,
 tau=result.tau,
 phi=result.phi,
 oof_n=w.get("n", 0),
 shrinkage=w.get("shrinkage", 0.15),
 training_cutoff="", 
 oof_segments=K_SEG, 
 )
 
 if verbose:
 print(
 f" {lt.value}: τ={result.tau:.3f} φ={result.phi:.1f} "
 f"w={ {k: round(v, 3) for k, v in weights_out[lt.value].items()} } "
 f"ll={w['log_loss']:.4f} n={w.get('n', 0)} "
 f"hash={artifact.model_hash()[:8]}",
 flush=True,
 )
 
 # 写入旧格式(兼容,league scoped)
 write_all(weights_out, params_out, meta_out, _ens_dir)
 
 
 for lt, _, _ in leagues:
 if lt.value in weights_out:
 artifact = create_production_artifact(
 league=lt.value,
 weights=weights_out[lt.value],
 tau=params_out[lt.value]["tau"],
 phi=params_out[lt.value]["phi"],
 oof_n=meta_out[lt.value].get("n", 0),
 shrinkage=meta_out[lt.value].get("shrinkage", 0.15),
 oof_segments=K_SEG, 
 )
 write_production_artifact(artifact, _ens_dir)
 
 return weights_out, params_out, meta_out
