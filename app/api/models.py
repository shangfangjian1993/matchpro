"""V2 模型与训练端点(新设计)"""

import glob
import os
import subprocess
import sys

from fastapi import APIRouter, Depends, HTTPException

from app.api.db import ModelRecord, TrainingTask, db
from app.api.helpers import require_admin
from app.api.schemas import TrainingSubmitReq
from app.api.security import get_current_user
from app.core.timeutil import utcnow
from app.data.adapters import _resolve_league_type

router = APIRouter(prefix="/api", tags=["models"])

MODELS_DIR = os.environ.get(
 "MODELS_DIR", str(__import__("app.core.paths", fromlist=["MODELS_DIR"]).MODELS_DIR)
)


def _worker_script() -> str:
 root = os.path.dirname(
 os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 )
 return os.path.join(root, "app", "services", "system", "train_worker.py")


def submit_training_task(
 task_id, league_type_value, target_column, cross_validation, cv_folds
) -> None:
 env = dict(os.environ)
 cmd = [
 sys.executable,
 _worker_script(),
 "--task-id",
 str(task_id),
 "--league-type",
 league_type_value,
 "--target-column",
 target_column,
 "--cv",
 str(bool(cross_validation)).lower(),
 "--cv-folds",
 str(cv_folds),
 ]
 log_dir = "/tmp/hermes-train-logs"
 os.makedirs(log_dir, exist_ok=True)
 try:
 with open(os.path.join(log_dir, f"train_{task_id}.log"), "ab") as log_f:
 subprocess.Popen(
 cmd,
 env=env,
 start_new_session=True,
 stdout=log_f,
 stderr=subprocess.STDOUT,
 )
 except OSError as e:
 task = db.session.get(TrainingTask, task_id)
 if task is not None:
 task.status = "failed"
 task.message = f"训练子进程启动失败: {e}"
 task.finished_at = utcnow()
 db.session.commit()


# ---------------- 模型 ----------------


@router.get("/models")
def list_models(user=Depends(get_current_user)):
 records = ModelRecord.query.order_by(ModelRecord.training_date.desc()).all()
 return {"items": [r.to_dict() for r in records], "total": len(records)}


@router.get("/models/performance")
def model_performance():
 records = ModelRecord.query.order_by(ModelRecord.training_date.desc()).all()
 return [r.to_dict() for r in records]


@router.post("/models/{model_id}/retrain", status_code=202)
def retrain_model(model_id: int, admin=Depends(require_admin)):
 record = db.session.get(ModelRecord, model_id)
 if record is None:
 raise HTTPException(404, "模型不存在")
 task = TrainingTask(
 user_id=admin.id,
 league_type=record.league_type,
 status="pending",
 message=f"重新训练模型 #{model_id}",
 )
 db.session.add(task)
 db.session.commit()
 submit_training_task(task.id, record.league_type, "goals", True, 5)
 return {"message": "重训练任务已提交", "task": task.to_dict()}


@router.delete("/models/{model_id}")
def delete_model(model_id: int, admin=Depends(require_admin)):
 record = db.session.get(ModelRecord, model_id)
 if record is None:
 raise HTTPException(404, "模型不存在")
 removed = []
 _art_dir = os.path.join(MODELS_DIR, record.league_type)
 for pat in (
 os.path.join(_art_dir, "*.pkl"),
 os.path.join(_art_dir, "*.pkl.sha256"),
 os.path.join(_art_dir, "*.pkl.json"),
 ):
 for fp in glob.glob(pat):
 try:
 os.remove(fp)
 removed.append(os.path.basename(fp))
 except OSError:
 pass
 # active_models.json 同步移除该联赛指针
 try:
 _am = os.path.join(
 str(__import__("app.core.paths", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT),
 "runtime",
 "active_models.json",
 )
 if os.path.exists(_am):
 import json as _json

 with open(_am, encoding="utf-8") as _f:
 _data = _json.load(_f)
 _data.pop(record.league_type, None)
 with open(_am, "w", encoding="utf-8") as _f:
 _json.dump(_data, _f, ensure_ascii=False, indent=2)
 except Exception:
 pass
 db.session.delete(record)
 db.session.commit()
 return {"message": "模型已删除", "removed_files": removed}


# ---------------- 训练 ----------------


@router.post("/training", status_code=202)
def train_model(body: TrainingSubmitReq, admin=Depends(require_admin)):
 try:
 league_type = _resolve_league_type(body.league_type)
 except ValueError as e:
 raise HTTPException(400, str(e))
 task = TrainingTask(
 user_id=admin.id,
 league_type=league_type.value,
 status="pending",
 message="任务已创建,等待执行",
 )
 db.session.add(task)
 db.session.commit()
 submit_training_task(
 task.id,
 league_type.value,
 body.target_column,
 body.cross_validation,
 body.cv_folds,
 )
 return {"message": "训练任务已提交", "task": task.to_dict()}


@router.get("/training/{task_ref}")
def train_status(task_ref: str, user=Depends(get_current_user)):
 task = None
 if task_ref.isdigit():
 task = db.session.get(TrainingTask, int(task_ref))
 if task is None:
 task = TrainingTask.query.filter_by(public_id=task_ref).first()
 if task is None:
 raise HTTPException(404, "任务不存在")
 if task.user_id != user.id:
 raise HTTPException(403, "无权查看该任务")
 return task.to_dict()


@router.get("/training")
def train_list(user=Depends(get_current_user)):
 tasks = (
 TrainingTask.query.filter_by(user_id=user.id)
 .order_by(TrainingTask.id.desc())
 .limit(50)
 .all()
 )
 return {"items": [t.to_dict() for t in tasks], "total": len(tasks)}
