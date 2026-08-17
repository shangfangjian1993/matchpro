"""上下文动态权重(审查三十七 P2:Dynamic Ensemble Context Weight)。

w_final = f(league, team_strength_gap, uncertainty):
  - 强弱悬殊(att_diff 大)→ 统计模型 HGBR 更可信(权重↑)
  - 势均力敌(att_diff 小)→ 状态/先验模型(bayes/elo)更有信息
  - 模型分歧大(disagreement 高)→ 向 HGBR 收缩(保守)

未离线验证前默认关闭(configs/models.yaml → ensemble.context_weight.enabled)。
审计信息写入 result["context_weights"]。
"""
from __future__ import annotations

import numpy as np


def _cfg() -> dict:
    try:
        from app.core.config import load_yaml
        return ((load_yaml("models.yaml").get("ensemble") or {})
                .get("context_weight") or {})
    except Exception:
        return {}


def adjust(base_w: dict, att_diff: float, disagreement: float) -> tuple[dict, dict | None]:
    """返回 (调整后权重, 审计信息);禁用或无调整时返回 (base_w, None)。"""
    cfg = _cfg()
    if not cfg.get("enabled", False):
        return dict(base_w), None
    max_delta = float(cfg.get("max_delta", 0.10))
    w = {k: float(base_w.get(k, 0.0)) for k in ("hgbr", "dc", "nb", "elo", "gbm", "bayes")}
    gap = float(np.clip(abs(att_diff) / 200.0, 0.0, 1.0))
    # 强弱悬殊(gap>0.5)→ hgbr↑;势均力敌(gap<0.5)→ hgbr↓(让先验/状态成员发挥)
    delta = max_delta * (gap - 0.5)
    changes = {"gap": round(gap, 3), "delta": round(delta, 4)}
    w["hgbr"] = float(np.clip(w["hgbr"] + delta, 0.0, 0.7))
    others = [k for k in ("bayes", "elo", "gbm", "nb", "dc") if w[k] > 0]
    if others and abs(delta) > 1e-9:
        rem = sum(w[k] for k in others)
        for k in others:
            w[k] = float(np.clip(w[k] - delta * w[k] / max(rem, 1e-9), 0.0, 0.7))
    # 模型分歧大 → 向 HGBR 收缩(保守)
    if disagreement > float(cfg.get("disagreement_threshold", 0.08)):
        bump = float(cfg.get("disagreement_bump", 0.05))
        w["hgbr"] = float(np.clip(w["hgbr"] + bump, 0.0, 0.7))
        others2 = [k for k in ("bayes", "elo", "gbm", "nb", "dc") if w[k] > 0]
        if others2:
            rem2 = sum(w[k] for k in others2)
            for k in others2:
                w[k] = float(np.clip(w[k] - bump * w[k] / max(rem2, 1e-9), 0.0, 0.7))
        changes["disagreement_bump"] = bump
    s = sum(w.values())
    if s <= 0:
        return dict(base_w), None
    w = {k: v / s for k, v in w.items()}
    return w, {"method": "context_adjust", **changes,
               "weights": {k: round(v, 4) for k, v in w.items()}}
