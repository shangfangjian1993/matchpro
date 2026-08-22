"""Ensemble 权重学习包。"""
from .oof_generator import K_SEG, generate
from .oof_generator import MIN_OOF_SAMPLES as MIN_OOF_SAMPLES
from .oof_generator import SAMPLE_PER_SEG as SAMPLE_PER_SEG


def run_all(leagues, verbose: bool = True, *, 
              train_dfs=None, cal_artifacts=None, prior_artifacts=None,
              gbm_model_hashes=None, training_cutoff=None) -> dict:
    """对 5 联赛执行 OOF 权重学习;返回 (weights, params, meta)。
    
    P0-1: 每个联赛写入各自目录,不会覆盖。
    P1-1: 传递 training_cutoff(真实数据截止时间)。
    P1-2: 传递真实 K_SEG。
    P1-4: lineage 参数由调用方注入。
    """
    from app.core.paths import ARTIFACTS_DIR as _AD
    
    from .artifact import create_production_artifact
    from .artifact_writer import write_all, write_production_artifact
    from .weight_optimizer import optimize
    
    _train_dfs = train_dfs or {}
    _cal_artifacts = cal_artifacts or {}
    _prior_artifacts = prior_artifacts or {}
    _gbm_model_hashes = gbm_model_hashes or {}
    _training_cutoff = training_cutoff or ""
    
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
            k: v for k, v in w.items()  # P1-6: 全精度,无 round
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
        
        if verbose:
            print(
                f"  {lt.value}: τ={result.tau:.3f} φ={result.phi:.1f} "
                f"w={ {k: round(v, 3) for k, v in weights_out[lt.value].items()} } "
                f"ll={w['log_loss']:.4f} n={w.get('n', 0)}",
                flush=True,
            )
    
    # 写入旧格式(兼容,league scoped)
    write_all(weights_out, params_out, meta_out, _ens_dir)
    
    # P0-1: 写入 ProductionArtifact (league scoped)
    for lt, _, _ in leagues:
        if lt.value in weights_out:
            # P1-4: Compute lineage hashes from training data
            import hashlib
            _train_df = _train_dfs.get(lt.value)
            _training_data_hash = ""
            if _train_df is not None and len(_train_df) > 0:
                _train_bytes = _train_df.to_json().encode("utf-8")
                _training_data_hash = hashlib.sha256(_train_bytes).hexdigest()[:16]
            
            _cal_hash = ""
            if lt.value in _cal_artifacts and cal_artifacts[lt.value]:
                _cal_hash = _cal_artifacts[lt.value].artifact_hash
            
            _prior_hash = ""
            if lt.value in _prior_artifacts and prior_artifacts[lt.value]:
                _prior_bytes = str(_prior_artifacts[lt.value].__dict__).encode("utf-8")
                _prior_hash = hashlib.sha256(_prior_bytes).hexdigest()[:16]
            
            _gbm_hash = _gbm_model_hashes.get(lt.value, "")
            
            artifact = create_production_artifact(
                league=lt.value,
                weights=weights_out[lt.value],
                tau=params_out[lt.value]["tau"],
                phi=params_out[lt.value]["phi"],
                oof_n=meta_out[lt.value].get("n", 0),
                shrinkage=meta_out[lt.value].get("shrinkage", 0.15),
                oof_segments=K_SEG,
                training_cutoff=_training_cutoff,
                calibration=_cal_artifacts.get(lt.value),
                prior=_prior_artifacts.get(lt.value),
                training_data_hash=_training_data_hash,
                calibration_hash=_cal_hash,
                prior_hash=_prior_hash,
            )
            write_production_artifact(artifact, _ens_dir)
    
    return weights_out, params_out, meta_out
