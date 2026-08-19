"""Pipeline 3:train —— Ensemble 训练 + Experiment 记录 + 自动模型选择。

幂等:每次训练产生新版本,active 指针自动指向最优(自动模型选择)。
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    from app.services.cli import add_log_level_arg, make_parser, setup_logging

    ap = make_parser("Pipeline train:全联赛训练 + Ensemble 权重学习 + 自动选择")
    add_log_level_arg(ap)
    args = ap.parse_args()
    setup_logging(args.log_level)

    # 1) 全联赛训练(Experiment 自动记录)
    from app.services.training.train_all_leagues import main as train_main

    train_main()
    # 2) Ensemble 权重滚动学习(四成员)
    from app.services.training.learn_ensemble_weights import main as learn_main

    learn_main()
    # 3) 自动模型选择(active 指针)
    from app.services.model.auto_select_model import main as select_main

    select_main()
    print("✅ [pipeline] train 完成")


if __name__ == "__main__":
    from app.services.cli import run

    raise SystemExit(run(main))
