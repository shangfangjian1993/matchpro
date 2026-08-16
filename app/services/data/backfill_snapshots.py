"""快照回填 + Replay 评估(§7.2):赛后实际比分 → 快照 §6 全量指标。

核心逻辑在 app/replay/service.py(replay_all/backfill_snapshot),本模块仅为 CLI 入口。
"""
import sys

_ROOT = str(__import__('app.core.paths', fromlist=['PROJECT_ROOT']).PROJECT_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> int:
    from app.services.cli import add_log_level_arg, make_parser, setup_logging
    ap = make_parser("快照回填 + Replay 评估(§6 全量指标)")
    ap.add_argument("--force", action="store_true", help="强制重算已回填快照")
    add_log_level_arg(ap)
    args = ap.parse_args()
    setup_logging(args.log_level)

    from app.api.db import init_db, session_scope
    from app.replay.service import replay_all
    init_db()
    with session_scope():
        summary = replay_all(force=args.force)
    print(f"✅ 回填完成: 快照 {summary.get('count', 0)} 条 | "
          f"Log-Loss {summary.get('log_loss', 'N/A')} | Brier {summary.get('brier', 'N/A')}")
    return 0


if __name__ == "__main__":
    from app.services.cli import run
    raise SystemExit(run(main))
