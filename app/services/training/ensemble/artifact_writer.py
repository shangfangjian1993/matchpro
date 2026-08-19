"""Artifact 写入(审查九 P1-9 拆分):权重/τφ/OOF meta。"""

from __future__ import annotations

import json
import os


def write_all(weights, params, meta, artifacts_dir: str) -> None:
    """落盘(审查 f01d7e4 P1-5:权重按 Goal/Score/Outcome 三层视图存储)。"""
    from app.models.ensemble.weights import to_layered

    os.makedirs(artifacts_dir, exist_ok=True)
    layered = {
        lt: (to_layered(w) if not isinstance(w.get("goal_lambda", None), dict) else w)
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
