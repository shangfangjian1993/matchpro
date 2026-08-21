"""Pipeline 2:feature —— Feature Factory → Feature Store(版本化)。

幂等:特征版本由公式哈希决定,重跑不重复注册。
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    from app.services.cli import add_log_level_arg, make_parser, setup_logging

    ap = make_parser("Pipeline feature:重算 ELO + 校验特征注册表(版本化)")
    add_log_level_arg(ap)
    args = ap.parse_args()
    setup_logging(args.log_level)

    from app.api.db import FeatureStore, init_db, session_scope

    init_db()
    with session_scope():
        n = FeatureStore.query.count()
        print(f"✅ [pipeline] feature store 现状: {n} 条注册(特征由训练时自动注册)")
    # ELO 重算(防泄漏时间重放,写 teams 三维)
    from app.services.data.compute_elo import main as compute_main

    compute_main()
    print("✅ [pipeline] feature 完成(ELO 三维已重算)")


if __name__ == "__main__":
    from app.services.cli import run

    raise SystemExit(run(main))
