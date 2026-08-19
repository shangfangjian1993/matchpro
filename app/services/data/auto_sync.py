"""自动定时同步任务:赛果/xG/赛程/模型重训。

由 scripts/scheduler.py 按时间窗口调用(--job),也可手动执行:
    python scripts/auto_sync.py --job daily       # 同步当前赛季赛果 + xG
    python scripts/auto_sync.py --job fixtures    # 同步赛程(需 FOOTBALL_DATA_ORG_KEY)
    python scripts/auto_sync.py --job weekly      # 重训五大联赛模型
    python scripts/auto_sync.py --job all         # 以上全部

- 幂等(upsert),重复运行安全
- 文件锁防并发(同步与训练互斥)
- 日志:stderr(由调度器重定向到 /var/log/auto_sync.log)
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.cli import add_log_level_arg, make_parser, run, setup_logging

_LEAGUE_CODES = ["E0", "SP1", "D1", "I1", "F1"]
_FDO_CODES = ["PL", "PD", "BL1", "SA", "FL1"]
_LOCK = "/tmp/auto_sync.lock"


def _job_collector(freq: str):
    """数据源集采编排(重构:按实效性频次,主源→降级)。"""
    from app.services.data.collector import run_frequency

    print(f"[collector] 执行 {freq} ...", flush=True)
    report = run_frequency(freq)
    ok = all(v.get("ok", False) for v in report.values())
    for dtype, v in report.items():
        status = "✅" if v.get("ok") else "❌"
        print(f"  {status} {dtype}: {v.get('source')} | {v.get('detail')}", flush=True)
    print(f"[collector] {freq} 完成 rc={'0' if ok else '1'}", flush=True)
    return 0 if ok else 1


def _current_season_start() -> int:
    """当前足球赛季起始年:8 月~次年 5 月为一季(2026-08 → 2026)"""
    from datetime import datetime

    now = datetime.now(tz=datetime.timezone.utc)
    return now.year if now.month >= 8 else now.year - 1


def _with_lock(fn, *args, **kwargs):
    """flock 互斥:同步/训练不并发(防止同写模型目录/数据库)"""
    import fcntl

    with open(_LOCK, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            return fn(*args, **kwargs)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _job_daily():
    from app.data.pipeline import run_history, run_xg

    season = _current_season_start()
    print(f"[daily] 同步 {season}/{season + 1} 赛季赛果 + xG ...", flush=True)
    r_h = run_history([season], _LEAGUE_CODES)
    r_x = run_xg([season], _LEAGUE_CODES)
    print(
        f"[daily] 赛果: 新增 {r_h['inserted']} 更新 {r_h['updated']} 错误 {len(r_h['errors'])}",
        flush=True,
    )
    print(f"[daily] xG:   更新 {r_x['updated']} 错误 {len(r_x['errors'])}", flush=True)
    if r_h["errors"] or r_x["errors"]:
        print(f"[daily] 警告: {r_h['errors'][:3] + r_x['errors'][:3]}", flush=True)
        return 1
    return 0


def _job_fixtures():
    key = os.environ.get("FOOTBALL_DATA_ORG_KEY", "").strip()
    if not key:
        print("[fixtures] 未设置 FOOTBALL_DATA_ORG_KEY,跳过", flush=True)
        return 0
    from app.data.pipeline import run_fixtures

    season = _current_season_start()
    print(f"[fixtures] 同步 {season}/{season + 1} 赛程 ...", flush=True)
    r = run_fixtures(season, _FDO_CODES)
    print(
        f"[fixtures] 新增 {r['inserted']} 更新 {r['updated']} 错误 {len(r['errors'])}",
        flush=True,
    )
    return 1 if r["errors"] else 0


def _job_injury():
    """拉取当日全部伤停(api-football,100 次/天额度内;按日期 1 次覆盖当天所有场次)"""
    key = os.environ.get("API_FOOTBALL_KEY", "").strip()
    if not key:
        print("[injury] 未设置 API_FOOTBALL_KEY,跳过", flush=True)
        return 0
    from datetime import datetime

    from app.data.sources.injuries.collector import InjuriesCollector

    day = datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d")
    print(f"[injury] 拉取 {day} 伤停 ...", flush=True)
    c = InjuriesCollector()
    recs = c.fetch_by_date(day, use_cache=False)  # 强制刷新当日缓存
    print(f"[injury] 获取 {len(recs)} 条伤停记录(已缓存)", flush=True)
    return 0


def _job_weekly():
    from app.api.db import init_db, session_scope
    from app.core.config import LeagueType
    from app.services.training.trainer import train_model

    init_db()
    results = []
    with session_scope():
        for code in _LEAGUE_CODES:
            lt = LeagueType[
                {
                    "E0": "PREMIER_LEAGUE",
                    "SP1": "LA_LIGA",
                    "D1": "BUNDESLIGA",
                    "I1": "SERIE_A",
                    "F1": "LIGUE_1",
                }[code]
            ]
            print(f"[weekly] 重训 {lt.value} ...", flush=True)
            m = train_model(lt, "goals", True, 5)
            results.append(f"{lt.value}: v{m.get('model_version')}")
            print(
                f"[weekly]   OK v{m.get('model_version')} poisson={m.get('poisson_loss'):.4f}",
                flush=True,
            )
    print(f"[weekly] 完成: {', '.join(results)}", flush=True)
    return 0


def run_sync_job(job: str = "all") -> dict:
    """同步任务分发(daily/fixtures/injury/weekly/all);返回汇总 dict。"""
    jobs = (
        ["daily", "fixtures", "injury", "weekly", "monthly"] if job == "all" else [job]
    )
    result = {}
    for j in jobs:
        fn = {
            "daily": _job_daily,
            "fixtures": _job_fixtures,
            "injury": _job_injury,
            "weekly": _job_weekly,
            "monthly": lambda: _job_collector("monthly"),
        }[j]
        try:
            rc = _with_lock(fn)
            result[j] = rc
        except Exception as e:
            print(f"[{j}] 异常: {e}", file=sys.stderr)
            result[j] = 1
        import time

        time.sleep(2)
    return result


def main() -> int:
    ap = make_parser("自动定时同步任务:赛果/xG/赛程/模型重训")
    ap.add_argument(
        "--job", required=True, choices=["daily", "fixtures", "injury", "weekly", "all"]
    )
    add_log_level_arg(ap)
    args = ap.parse_args()
    setup_logging(args.log_level)

    jobs = (
        ["daily", "fixtures", "injury", "weekly"] if args.job == "all" else [args.job]
    )
    code = 0
    for job in jobs:
        fn = {
            "daily": _job_daily,
            "fixtures": _job_fixtures,
            "injury": _job_injury,
            "weekly": _job_weekly,
        }[job]
        try:
            rc = _with_lock(fn)
            code = code or rc
        except Exception as e:
            print(f"[{job}] 异常: {e}", file=sys.stderr)
            code = 1
        time.sleep(2)
    return code


if __name__ == "__main__":
    raise SystemExit(run(main))
