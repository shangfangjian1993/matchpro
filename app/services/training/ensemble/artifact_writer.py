"""Artifact 写入(ProductionArtifact 为主)。"""
from __future__ import annotations

import json
import os


def write_all(weights, params, meta, artifacts_dir: str) -> None:
    """落盘(兼容旧格式 + ProductionArtifact)。"""
    from app.models.ensemble.weights import to_layered
    
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # 旧格式(兼容)
    layered = {
        lt: to_layered(w)
        for lt, w in weights.items()
    }
    with open(
        os.path.join(artifacts_dir, "ensemble_weights.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(layered, f, ensure_ascii=False, indent=2)
    with open(
        os.path.join(artifacts_dir, "dc_nb_params.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    with open(os.path.join(artifacts_dir, "oof_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def write_production_artifact(artifact, artifacts_dir: str) -> str:
    """写入 ProductionArtifact (主格式)。"""
    os.makedirs(artifacts_dir, exist_ok=True)
    path = os.path.join(artifacts_dir, "production_artifact.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(artifact.to_json())
    return path
