"""Artifact 写入(League Scoped + 全精度)。"""
from __future__ import annotations

import json
import os


def _artifact_dir(artifacts_dir: str, league: str) -> str:
    """联赛隔离目录。"""
    return os.path.join(artifacts_dir, league)


def write_all(weights, params, meta, artifacts_dir: str) -> None:
    """落盘(兼容旧格式,league scoped)。"""
    from app.models.ensemble.weights import to_layered
    
    os.makedirs(artifacts_dir, exist_ok=True)
    
    for lt, w in weights.items():
        league_dir = _artifact_dir(artifacts_dir, lt)
        os.makedirs(league_dir, exist_ok=True)
        
        # 旧格式(兼容)
        layered = to_layered(w)
        with open(os.path.join(league_dir, "ensemble_weights.json"), "w", encoding="utf-8") as f:
            json.dump(layered, f, ensure_ascii=False, indent=2)
    
    # params 和 meta 按联赛隔离
    for lt, p in params.items():
        league_dir = _artifact_dir(artifacts_dir, lt)
        os.makedirs(league_dir, exist_ok=True)
        with open(os.path.join(league_dir, "dc_nb_params.json"), "w", encoding="utf-8") as f:
            json.dump(p, f, ensure_ascii=False, indent=2)
    
    for lt, m in meta.items():
        league_dir = _artifact_dir(artifacts_dir, lt)
        os.makedirs(league_dir, exist_ok=True)
        with open(os.path.join(league_dir, "oof_meta.json"), "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)


def write_production_artifact(artifact, artifacts_dir: str) -> str:
    """写入 ProductionArtifact (atomic publish + league scoped)。
    
    P1-5: 原子发布(先写 tmp,再 rename,保证 READY 语义)。
    """
    import tempfile
    league_dir = _artifact_dir(artifacts_dir, artifact.league)
    os.makedirs(league_dir, exist_ok=True)
    
    path = os.path.join(league_dir, "production_artifact.json")
    # Atomic write: write to temp, then rename
    fd, tmp_path = tempfile.mkstemp(dir=league_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(artifact.to_json())
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return path
