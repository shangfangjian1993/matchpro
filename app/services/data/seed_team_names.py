"""种子导入:把内置队名映射写入 team_names 表(幂等,可重复执行)。

用法:
    python scripts/seed_team_names.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.cli import run, setup_logging


def main() -> int:
    setup_logging("INFO")

    from app.api.db import init_db, session_scope
    from app.data.canonical.team_names_zh import seed_from_builtin

    init_db()
    with session_scope():
        n = seed_from_builtin()
    print(f"✅ 种子导入完成(新增/更新 {n} 条;共 269 条映射)")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
