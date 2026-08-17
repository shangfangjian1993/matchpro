"""训练 worker:独立子进程执行异步训练任务(由 api/blueprints/training.py 启动)。

用途:训练任务从 gunicorn worker 线程中剥离——
- 训练进程不占用 web worker,API 吞吐不受影响;
- 子进程 start_new_session 脱离进程组,worker 重启/滚动发布不中断训练;
- 任务状态/指标/通知直接写入数据库,前端轮询契约不变。

用法(由 submit_training_task 调用,一般无需手动执行):
    python scripts/train_worker.py --task-id 1 --league-type premier_league \
        --target-column goals --cv true --cv-folds 5

环境变量:DATABASE_URL(同 API)、MODELS_DIR(默认 <项目根>/models)。
"""
import argparse
import logging

_logger = logging.getLogger(__name__)
import json
import os
import sys

# ---- sys.path 修复:从任意 cwd 启动都可导入包 ----
_ROOT = str(__import__('app.core.paths', fromlist=['PROJECT_ROOT']).PROJECT_ROOT)  # 项目根
_PKG_DIR = os.path.join(_ROOT, "src")  # src 布局:包在 src/ 下
for _p in (_PKG_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _default_database_url() -> str:
    """与 api/app.py 的默认值保持一致(本地 sqlite;生产由环境变量注入)"""
    return "sqlite:///" + os.path.join(_ROOT, "data", "football.db")


def main() -> int:
    ap = argparse.ArgumentParser(description="模型训练 worker")
    ap.add_argument("--task-id", required=True, type=int)
    ap.add_argument("--league-type", required=True)
    ap.add_argument("--target-column", default="goals")
    ap.add_argument("--cv", default="true", help="是否交叉验证(true/false)")
    ap.add_argument("--cv-folds", default=5, type=int)
    args = ap.parse_args()

    from app.api.db import (
        ModelRecord,
        Notification,
        TrainingTask,
        db,
        init_db,
        session_scope,
    )
    from app.core.timeutil import utcnow
    from app.data.adapters import _resolve_league_type
    from app.services.training.trainer import train_model

    init_db(os.environ.get("DATABASE_URL", None))

    from app.core.paths import MODELS_DIR as _MD
    models_dir = os.environ.get("MODELS_DIR") or str(_MD)

    with session_scope():
        task = db.session.get(TrainingTask, args.task_id)
        if task is None:
            print(f"任务 {args.task_id} 不存在", file=sys.stderr)
            return 2
        task.status = "running"
        task.started_at = utcnow()
        task.message = "模型训练中..."
        db.session.commit()

        try:
            league_type = _resolve_league_type(args.league_type)
            metrics = train_model(
                league_type,
                args.target_column,
                cross_validation=args.cv.lower() in ("1", "true", "yes"),
                cv_folds=args.cv_folds,
                models_dir=models_dir,
            )
        except Exception as e:
            _logger.exception("训练任务 %s 失败: %s", args.task_id, e)
            task.status = "failed"
            task.message = f"训练失败: {e}"
            task.finished_at = utcnow()
            db.session.add(Notification(
                user_id=task.user_id,
                title="模型训练失败",
                content=f"{args.league_type} 模型训练失败: {e}",
            ))
            db.session.commit()
            return 1

        # 更新(或新建)模型记录
        record = ModelRecord.query.filter_by(league_type=league_type.value).first()
        if record is None:
            record = ModelRecord(
                name=f"{league_type.value}模型",
                league_type=league_type.value,
                model_type=metrics.get("model_type", "HGBR"),
                version=str(metrics.get("model_version", "1.0.0")),
            )
            db.session.add(record)
        for key in ("mse", "mae", "rmse", "poisson_loss", "exact_accuracy"):
            v = metrics.get(key, 0.0)
            setattr(record, key, float(v) if v is not None else 0.0)
        record.accuracy = float(metrics.get("exact_accuracy", 0.0) or 0.0)
        record.feature_count = int(metrics.get("feature_count", 0) or 0)
        record.model_path = metrics.get("model_path", "")
        record.training_date = utcnow()

        task.status = "succeeded"
        task.message = "训练完成"
        task.metrics_json = json.dumps(metrics, ensure_ascii=False, default=str)
        task.finished_at = utcnow()
        db.session.add(Notification(
            user_id=task.user_id,
            title="模型训练完成",
            content=(f"{args.league_type} 模型训练完成,"
                     f"版本 v{metrics.get('model_version', '?')},"
                     f"MSE {float(metrics.get('mse', 0.0) or 0.0):.4f}"),
        ))
        db.session.commit()
        print(f"训练完成: {args.league_type} v{metrics.get('model_version')}")
        return 0


if __name__ == "__main__":
    from app.services.cli import run
    raise SystemExit(run(main))
