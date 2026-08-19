"""Outcome Engine(审查 §12/§14 拆分):GBM 分类器 + Goal 1X2 融合。

GBM 直接建模胜平负(与泊松类低相关);只参与 1X2,不参与比分矩阵。
"""

from __future__ import annotations

import os

from app.core.cache import ArtifactCache
from app.models.ensemble import fuse_goal_outcome

_GBM_CACHE = ArtifactCache(8)


def load_gbm(league_type, models_dir: str):
    """加载 GBM 分类成员(统一 ArtifactCache);不存在返回 None。"""
    path = os.path.join(
        models_dir, league_type.value, "gbm.pkl"
    )  # 审查 P0-4:models_dir=模型根
    if not os.path.exists(path):
        return None
    try:
        cached = _GBM_CACHE.get(path)
        if cached is not None:
            return cached
        from app.models.ensemble.gbm import GbmClassifier

        gbm = GbmClassifier.load(path)
        _GBM_CACHE.put(path, gbm)
        return gbm
    except Exception:
        return None


def gbm_probs(gbm, pred_df, model) -> tuple | None:
    """GBM 三分类概率(预测行主队视角);失败返回 None(降级)。"""
    try:
        gfeat = model.prepare_features(pred_df)
        gcols = [col for col in gbm.feature_columns_ if col in gfeat.columns]
        # 审查 P1:GBM 只需主队预测行(-2 行);客队行(-1)不使用
        gp = gbm.predict_proba(gfeat[gcols].iloc[[-2]])
        return (float(gp[0][0]), float(gp[0][1]), float(gp[0][2]))
    except Exception:
        return None


def fuse(goal_probs, gbm_probs, weights):
    """Goal 1X2 + GBM 融合(审查 P0-9:两层)。"""
    return fuse_goal_outcome(goal_probs, gbm_probs, weights)
