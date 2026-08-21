"""从快照库拟合概率校准器,保存 artifacts/calibration/<league>.cal。

支持两种模式:
1. 默认(全历史):使用全部历史快照训练校准器(当前生产模式)
2. Temporal OOF(--temporal-oof):按赛季起始 cutoff 分段,只用 cutoff 之前数据 fit calibration,
   确保每个 fold 的 calibration artifact 无未来信息。

用法:
    python -m app.services.training.fit_calibration [--league premier_league] [--temporal-oof]
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from app.services.cli import make_parser, run, setup_logging

MIN_SAMPLES = 250  # 拟合最少样本(60/20/20 三段:train≥150/val≥50/test≥50)


def fit_calibration_for_window(snaps, lg, suffix=""):
    """为指定快照窗口拟合 calibration。"""
    from app.calibration.calibrator import Calibrator
    from app.core.paths import ARTIFACTS_DIR

    probs, labels = [], []
    for s in snaps:
        p = json.loads(s.probabilities_json)
        gh, ga = s.actual_home_goals or 0, s.actual_away_goals or 0
        label = 0 if gh > ga else (1 if gh == ga else 2)
        _pc = p.get("pre_calibration") or [
            p.get("home_win", 0),
            p.get("draw", 0),
            p.get("away_win", 0),
        ]
        probs.append([float(_pc[0]), float(_pc[1]), float(_pc[2])])
        labels.append(label)

    cal = Calibrator.fit_best(
        np.array(probs), np.array(labels), val_fraction=0.25, test_fraction=0.2
    )

    _cal_dir = os.path.join(ARTIFACTS_DIR, "calibration")
    os.makedirs(_cal_dir, exist_ok=True)
    path = os.path.join(_cal_dir, f"{lg.league_type}{suffix}.cal")
    cal.save(path)
    return path, len(snaps), getattr(cal, "_test_ece", None), cal._test_n


def main() -> int:
    ap = make_parser("概率校准器拟合:快照库 → artifacts/calibration/<league>.cal")
    ap.add_argument("--league", default=None, help="联赛枚举名;默认全部")
    ap.add_argument(
        "--temporal-oof",
        action="store_true",
        help="启用 temporal OOF:按赛季起始 cutoff 分段,只用 cutoff 之前数据 fit calibration",
    )
    setup_logging("INFO")
    args = ap.parse_args()

    from app.api.db import League, PredictionSnapshot, init_db, session_scope

    init_db()
    with session_scope():
        leagues = League.query.all()
        if args.league:
            leagues = [l for l in leagues if l.league_type == args.league]
        fitted = 0
        for lg in leagues:
            snaps = (
                PredictionSnapshot.query.filter_by(league_id=lg.id)
                .filter(PredictionSnapshot.actual_home_goals.isnot(None))
                .order_by(PredictionSnapshot.kickoff.asc())
                .all()
            )
            if len(snaps) < MIN_SAMPLES:
                print(f"{lg.name}: 样本 {len(snaps)} < {MIN_SAMPLES},跳过")
                continue

            if args.temporal_oof:
                # Temporal OOF:按赛季起始(8/1)分段
                # 2022-08-01 ~ 2023-07-31 为一个 fold,以此类推
                import pandas as pd

                season_boundaries = {}
                for s in snaps:
                    kickoff = pd.Timestamp(s.kickoff)
                    season_start = kickoff.year if kickoff.month >= 8 else kickoff.year - 1
                    if season_start not in season_boundaries:
                        season_boundaries[season_start] = []
                    season_boundaries[season_start].append(s)

                sorted_seasons = sorted(season_boundaries.keys())
                cumulative_snaps = []
                for season_start in sorted_seasons:
                    fold_snaps = season_boundaries[season_start]
                    if len(cumulative_snaps) < MIN_SAMPLES:
                        cumulative_snaps.extend(fold_snaps)
                        continue

                    # 用累积数据 fit calibration
                    path, n, test_ece, test_n = fit_calibration_for_window(
                        cumulative_snaps, lg, suffix=f"_temporal_{season_start}"
                    )
                    print(
                        f"✅ {lg.name} {season_start}+: {n} 样本 → {path} "
                        f"(test ECE {test_ece:.4f}, n={test_n})"
                    )
                    fitted += 1
                    cumulative_snaps.extend(fold_snaps)

                if fitted == 0:
                    print(f"{lg.name}: 赛季数不足,跳过 temporal OOF")
            else:
                # 默认:全历史模式
                path, n, test_ece, test_n = fit_calibration_for_window(snaps, lg)
                print(
                    f"✅ {lg.name}: {n} 样本 → {path} "
                    f"(test ECE {test_ece:.4f}, n={test_n})"
                )
                fitted += 1

        print(f"完成: 拟合 {fitted} 个校准器")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
