"""Outcome Engine(

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
 )
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
 gp = gbm.predict_proba(gfeat[gcols].iloc[[-2]])
 return (float(gp[0][0]), float(gp[0][1]), float(gp[0][2]))
 except Exception:
 return None


def fuse(goal_probs, gbm_probs, weights):
 """Goal 1X2 + GBM 融合(
 return fuse_goal_outcome(goal_probs, gbm_probs, weights)
