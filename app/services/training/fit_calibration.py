"""从快照库拟合概率校准器,保存 models/<league>_model.cal。

- 样本:prediction_snapshots 中已回填(有 actual)的快照
- 方法:每类样本 ≥50 → Beta 校准;否则跳过(样本不足)
- 输出:models/<league>_model.cal(与模型版本同目录)

用法:
    python scripts/fit_calibration.py [--league premier_league]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from app.services.cli import make_parser, run, setup_logging

MIN_SAMPLES = 150  # 每类最少样本(总量,三分类共需)


def main() -> int:
    ap = make_parser("概率校准器拟合:快照库 → models/<league>_model.cal")
    ap.add_argument("--league", default=None, help="联赛枚举名;默认全部")
    setup_logging("INFO")
    args = ap.parse_args()

    from app.api.db import League, PredictionSnapshot, init_db, session_scope
    from app.calibration.calibrator import Calibrator

    init_db()
    with session_scope():
        leagues = League.query.all()
        if args.league:
            leagues = [l for l in leagues if l.league_type == args.league]
        fitted = 0
        for lg in leagues:
            snaps = PredictionSnapshot.query.filter_by(league_id=lg.id).filter(
                PredictionSnapshot.actual_home_goals.isnot(None)).all()
            if len(snaps) < MIN_SAMPLES:
                print(f"{lg.name}: 样本 {len(snaps)} < {MIN_SAMPLES},跳过")
                continue
            probs, labels = [], []
            for s in snaps:
                p = json.loads(s.probabilities_json)
                gh, ga = s.actual_home_goals or 0, s.actual_away_goals or 0
                label = 0 if gh > ga else (1 if gh == ga else 2)
                probs.append([p.get("home_win", 0), p.get("draw", 0), p.get("away_win", 0)])
                labels.append(label)
            # 联赛择优(评审 P1):β/Platt/Isotonic 三方法,评估段选 ECE 最低
            cal = Calibrator.fit_best(np.array(probs), np.array(labels))
            path = f"models/{lg.league_type}_model.cal"
            cal.save(path)
            fitted += 1
            print(f"✅ {lg.name}: {len(snaps)} 样本 → {path} "
                  f"(方法 {cal.method}, ECE {getattr(cal, '_ece', 'N/A'):.4f})")
        print(f"完成: 拟合 {fitted} 个校准器")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
