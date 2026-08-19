"""Artifact 写入(审查九 P1-9 拆分):权重/τφ/OOF meta。"""

from __future__ import annotations

import json
import os


def write_all(weights, params, meta, artifacts_dir: str) -> None:
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(
        os.path.join(artifacts_dir, "ensemble_weights.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)
    with open(
        os.path.join(artifacts_dir, "dc_nb_params.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    with open(os.path.join(artifacts_dir, "oof_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
