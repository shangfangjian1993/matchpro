"""Pipeline 5:replay —— 快照回放 + 评估(§6 全量指标)→ 自动学习。

幂等:已回填的快照跳过(--force 强制重算)。
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    from app.services.cli import add_log_level_arg, make_parser, setup_logging

    ap = make_parser("Pipeline replay:快照回放评估(§6)+ 触发权重/校准学习")
    ap.add_argument("--force", action="store_true", help="强制重算已回填快照")
    add_log_level_arg(ap)
    args = ap.parse_args()
    setup_logging(args.log_level)

    from app.services.data.backfill_snapshots import main as backfill_main

    backfill_main()
    # §7.2 回放评估 → 汇入 experiment(service 内完成)
    from app.api.db import PredictionSnapshot, init_db, session_scope
    from app.replay.service import summarize

    init_db()
    with session_scope():
        summary = summarize()
    print(
        f"✅ [pipeline] replay 完成: 快照 {summary.get('count', 0)} 条 | "
        f"Log-Loss {summary.get('log_loss', 'N/A')} | Brier {summary.get('brier', 'N/A')}"
    )

    # ---- §0.3 AUTO LEARN:快照 ≥150/联赛 → 自动重学 Ensemble 权重 + 自动模型选择 ----
    with session_scope():
        from app.api.db import League

        per_league = {
            l.league_type: PredictionSnapshot.query.filter(
                PredictionSnapshot.is_correct.isnot(None),
                PredictionSnapshot.league == l.name,
            ).count()
            for l in League.query.all()
        }
    ready = [lt for lt, cnt in per_league.items() if cnt >= 150]
    if ready:
        print(f"↻ AUTO LEARN: {ready} 快照充足(≥150),重学权重 + 自动模型选择")
        from app.services.training.learn_ensemble_weights import main as learn_main

        learn_main()
        from app.services.model.auto_select_model import main as select_main

        select_main()
    else:
        print(f"ℹ 快照积累中(各联赛 {per_league});≥150/联赛后自动重学权重")


if __name__ == "__main__":
    from app.services.cli import run

    raise SystemExit(run(main))
