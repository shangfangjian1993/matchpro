"""Pipeline 4:predict —— 预测 + Snapshot(可复现)。

用法:python pipelines/predict.py --league premier_league --home 阿森纳 --away 切尔西
输出:胜平负概率 + λ + 快照落库(幂等:同对阵同日不重复)。
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    import json

    from app.services.cli import add_log_level_arg, make_parser, setup_logging
    ap = make_parser("Pipeline predict:预测 + Snapshot(幂等)")
    ap.add_argument("--league", required=True, help="联赛枚举值,如 premier_league")
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--date", default=None, help="ISO 日期(复盘用;默认现在)")
    add_log_level_arg(ap)
    args = ap.parse_args()
    setup_logging(args.log_level)

    from app.api.db import init_db, session_scope
    from app.core.config import LeagueType
    from app.prediction.service import predict_match
    init_db()
    with session_scope():
        lt = LeagueType(args.league)
        r = predict_match(lt, args.home, args.away, args.date)
    print(json.dumps({
        "home": r["home_team_zh"], "away": r["away_team_zh"],
        "home_win": r["home_win_probability"], "draw": r["draw_probability"],
        "away_win": r["away_win_probability"],
        "lambda": [r["predicted_home_goals"], r["predicted_away_goals"]],
        # §4 完整输出
        "top_scores": r.get("top_scores", []),
        "over_2_5": r.get("over_2_5"), "under_2_5": r.get("under_2_5"),
        "btts": r.get("btts"), "expected_xg": r.get("expected_xg"),
        # 审查二十四:不确定性(置信度/熵/模型分歧/数据质量)
        "confidence": r.get("confidence"),
        "confidence_score": r.get("confidence_score"),
        "prediction_entropy": r.get("prediction_entropy"),
        "model_disagreement": r.get("model_disagreement"),
        "data_quality_score": r.get("data_quality_score"),
    }, ensure_ascii=False, indent=2))
    print("✅ [pipeline] predict 完成(快照已落库)")


if __name__ == "__main__":
    from app.services.cli import run
    raise SystemExit(run(main))
