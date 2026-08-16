"""Pipeline 1:ingest —— Raw→Canonical(数据引擎)。

幂等:重复执行不产生重复数据(入库去重 + 白名单校验)。
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    from app.services.cli import add_log_level_arg, make_parser, setup_logging
    ap = make_parser("Pipeline ingest:同步赛果/xG/赛程/伤停(幂等)")
    add_log_level_arg(ap)
    args = ap.parse_args()
    setup_logging(args.log_level)
    from app.services.data.auto_sync import run_sync_job
    run_sync_job("all")
    print("✅ [pipeline] ingest 完成")


if __name__ == "__main__":
    from app.services.cli import run
    raise SystemExit(run(main))
