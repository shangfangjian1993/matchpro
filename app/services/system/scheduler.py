"""容器内定时调度器:按时间窗口触发 auto_sync 任务。

由 entrypoint.sh 后台启动(独立进程),不依赖系统 cron:
- 每天 08:00 / 22:00 → daily(同步当前赛季赛果 + xG)
- 每天 09:30      → fixtures(同步赛程,需 FOOTBALL_DATA_ORG_KEY)
- 每周一 03:00    → weekly(重训五大联赛模型)

窗口机制:在目标时间 ±5 分钟内触发一次,然后睡眠到下一窗口,避免重复执行。
日志:由 entrypoint 重定向到 /var/log/auto_sync.log。
"""

import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto_sync.py")
# (任务, 星期几(0=周一..6=周日, None=每天), 时, 分)
_SCHEDULE = [
    ("daily", None, 8, 0),
    ("daily", None, 22, 0),
    ("fixtures", None, 9, 30),
    ("injury", None, 14, 0),  # 每日赛前拉当天伤停
    ("weekly", 0, 3, 0),
    ("monthly", 0, 4, 0),  # 每月首日 04:00   # 周一凌晨
]
_WINDOW_MIN = 5  # 触发窗口 ±5 分钟


def _due(task, weekday, hour, minute, now: datetime) -> bool:
    if weekday is not None and now.weekday() != weekday:
        return False
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return abs((now - target).total_seconds()) <= _WINDOW_MIN * 60


def _run(task: str) -> None:
    """执行任务(子进程,避免本进程异常影响调度循环)"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            os.path.join(os.path.dirname(_SCRIPT), "..", "src"),
            os.path.dirname(os.path.dirname(_SCRIPT)),
        ]
    )
    print(
        f"[scheduler] {datetime.now(tz=timezone.utc):%H:%M} 触发 {task}",
        flush=True,
    )
    try:
        r = subprocess.run(
            [sys.executable, _SCRIPT, "--job", task],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        print(r.stdout, flush=True)
        if r.stderr:
            print(r.stderr, file=sys.stderr, flush=True)
        print(f"[scheduler] {task} 完成 rc={r.returncode}", flush=True)
    except Exception as e:
        print(f"[scheduler] {task} 执行异常: {e}", file=sys.stderr, flush=True)


def main() -> int:
    print(
        "[scheduler] 启动,计划: "
        + ", ".join(
            f"{t} {w if w is not None else '每日'} {h:02d}:{m:02d}"
            for t, w, h, m in _SCHEDULE
        ),
        flush=True,
    )
    last_run = {}  # (task, weekday) -> 日期,防止窗口内重复
    while True:
        now = datetime.now(tz=timezone.utc)
        for task, weekday, hour, minute in _SCHEDULE:
            key = (task, now.date())
            if _due(task, weekday, hour, minute, now) and key not in last_run:
                _run(task)
                last_run[key] = True
                # 清理超过 3 天的记录,防内存增长
                if len(last_run) > 60:
                    cutoff = now.date() - timedelta(days=3)
                    last_run = {k: v for k, v in last_run.items() if k[1] >= cutoff}
        time.sleep(30)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
