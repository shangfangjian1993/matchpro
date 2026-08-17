"""预测管线版本(审查九 P1-7):pipeline_hash = 预测逻辑源码聚合哈希。

快照必须冻结"预测逻辑版本"—— 仅冻结 model/feature/ensemble/calibration
哈希还不够:goal_engine/outcome_engine/prior_blend/info_fusion 等编排代码
变化时,同样会改变预测结果,快照应能识别。

pipeline_version:人工维护的语义版本(每次预测逻辑变更 +1)。
pipeline_hash:自动源码哈希(任何相关模块改动即变化)。
"""
from __future__ import annotations

import hashlib
import inspect

PIPELINE_VERSION = "v7.5.0"

_PIPELINE_MODULES = (
    "app.prediction.goal_engine",
    "app.prediction.outcome_engine",
    "app.prediction.calibration",
    "app.prediction.prior_blend",
    "app.prediction.info_fusion",
    "app.prediction.regime",
    "app.models.ensemble.fusion",
    "app.models.ensemble.weights",
    "app.models.ensemble.probabilities",
)


def pipeline_hash() -> str:
    """源码聚合哈希:任何相关预测模块改动 → 哈希变化。"""
    parts = []
    for name in _PIPELINE_MODULES:
        try:
            mod = __import__(name, fromlist=["*"])
            parts.append(inspect.getsource(mod))
        except Exception:
            parts.append(f"{name}=missing")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
