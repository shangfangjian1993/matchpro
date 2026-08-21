"""全联赛真实模型训练:对 matches 表中的真实历史数据训练五大联赛模型。

用法(数据灌入后):
    DATABASE_URL=sqlite:///<项目根>/data/football.db MODELS_DIR=artifacts/models \
        .mlvenv/bin/python -m multi_league_model_system.app.services.train_all_leagues [--leagues E0,SP1,...]

与 API 训练流程一致(版本化保存 + latest 指针 + ModelRecord 落库),训练指标写入 models 表,
前端"模型性能"页可见。
"""

import os
import sys

_ROOT = str(__import__("app.core.paths", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT)
# 五层结构:核心包在 src/,顶层包 api/models/data 在项目根 —— 两者都要注入
for _p in (_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.api.db import ModelRecord
from app.core.config import LeagueType
from app.core.timeutil import utcnow
from app.services.training.trainer import train_model

DEFAULT_LEAGUES = ["PREMIER_LEAGUE", "LA_LIGA", "BUNDESLIGA", "SERIE_A", "LIGUE_1"]


def main():
    from app.services.cli import add_log_level_arg, make_parser, parse_leagues

    ap = make_parser("训练五大联赛真实模型(全量历史数据,版本化保存 + latest 指针)")
    ap.add_argument(
        "--leagues",
        default=",".join(DEFAULT_LEAGUES),
        help="联赛枚举名,逗号分隔(如 PREMIER_LEAGUE,LA_LIGA)",
    )
    add_log_level_arg(ap)
    args = ap.parse_args()
    from app.services.cli import setup_logging

    setup_logging(args.log_level)
    league_names = parse_leagues(args.leagues)

    from app.api.db import init_db, session_scope

    init_db(os.environ.get("DATABASE_URL", None))
    from app.core.paths import MODELS_DIR as _MD

    models_dir = os.environ.get("MODELS_DIR", str(_MD))

    results = []
    with session_scope():
        from app.api.db import db

        db.create_all()
        for name in league_names:
            try:
                lt = LeagueType[name]
            except KeyError:
                print(f"未知联赛 {name},跳过")
                continue
            print(f"=== 训练 {lt.value} ===", flush=True)
            t0 = utcnow()
            try:
                metrics = train_model(lt, "goals", True, 5, models_dir=models_dir)
                record = ModelRecord.query.filter_by(league_type=lt.value).first()
                if record is None:
                    record = ModelRecord(
                        name=f"{lt.value}模型",
                        league_type=lt.value,
                        model_type=metrics.get("model_type", "HGBR"),
                        version=str(metrics.get("model_version", "1.0.0")),
                    )
                    db.session.add(record)
                for key in ("mse", "mae", "rmse", "poisson_loss", "exact_accuracy"):
                    v = metrics.get(key, 0.0)
                    setattr(record, key, float(v) if v is not None else 0.0)
                record.version = str(
                    metrics.get("model_version", record.version or "1.0.0")
                )
                record.accuracy = float(metrics.get("exact_accuracy", 0.0) or 0.0)
                record.feature_count = int(metrics.get("feature_count", 0) or 0)
                record.model_path = metrics.get("model_path", "")
                record.training_date = db.func.now()
                db.session.commit()
                secs = (utcnow() - t0).total_seconds()
                print(
                    f"  OK v{metrics.get('model_version')} acc={metrics.get('exact_accuracy'):.4f} "
                    f"poisson={metrics.get('poisson_loss'):.4f} shape={metrics.get('training_data_shape')} ({secs:.0f}s)",
                    flush=True,
                )
                results.append((lt.value, "OK", str(metrics.get("model_version"))))
            except Exception as e:
                secs = (utcnow() - t0).total_seconds()
                print(f"  FAIL: {e} ({secs:.0f}s)", flush=True)
                results.append((lt.value, "FAIL", str(e)[:120]))

    print("\n==== 汇总 ====")
    for r in results:
        print("  ", r[0], r[1], r[2])
    bad = [r for r in results if r[1] != "OK"]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    from app.services.cli import run

    raise SystemExit(run(main))
