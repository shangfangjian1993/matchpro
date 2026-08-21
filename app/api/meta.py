"""V2 元数据端点:实验追踪 / 特征注册表 / 健康(新设计)"""

from fastapi import APIRouter, Query

from app.api.db import Experiment, FeatureStore

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/health")
def health():
 return {"status": "ok", "service": "football-prediction", "version": "4.0.0"}


@router.get("/experiments")
def experiments(limit: int = Query(50, ge=1, le=500)):
 rows = Experiment.query.order_by(Experiment.id.desc()).limit(limit).all()
 return {
 "items": [
 {
 "id": e.id,
 "model_version": getattr(e, "model_version", None),
 "league": getattr(e, "league_type", None),
 "hyperparams": getattr(e, "hyperparameters_json", None),
 "metrics": getattr(e, "metrics_json", None),
 "data_rows": getattr(e, "dataset_version", None),
 "data_hash": getattr(e, "data_hash", None),
 "created_at": str(getattr(e, "created_at", "") or ""),
 }
 for e in rows
 ],
 "total": len(rows),
 }


@router.get("/features")
def features(league: str | None = None, family: str | None = None):
 q = FeatureStore.query
 if league:
 q = q.filter_by(league_type=league)
 if family:
 q = q.filter_by(family=family)
 rows = q.order_by(FeatureStore.id.desc()).limit(1000).all()
 return {
 "items": [
 {
 "id": f.id,
 "league": getattr(f, "league", None),
 "family": getattr(f, "family", None),
 "formula_hash": getattr(f, "formula_hash", None),
 "feature_version": getattr(f, "feature_version", None),
 "created_at": str(getattr(f, "created_at", "") or ""),
 }
 for f in rows
 ],
 "total": len(rows),
 }
