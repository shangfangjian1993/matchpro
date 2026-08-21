"""生产环境一键初始化:爬取数据 → 训练模型 → 验证预测。

部署后执行一次即完成上线准备(幂等:sync upsert 可重复跑,训练版本递增)。

用法:
 python scripts/init_production.py # 全流程(数据 + 训练 + 验证)
 python scripts/init_production.py --skip-data # 只训练(数据已就位)
 python scripts/init_production.py --skip-train # 只爬数据
 python scripts/init_production.py --with-fixtures # 含未来赛程(需 FOOTBALL_DATA_ORG_KEY)
 python scripts/init_production.py --leagues E0,SP1 # 只处理指定联赛

环境变量(与 API 相同):DATABASE_URL / JWT_SECRET_KEY / ADMIN_PASSWORD(容器内已注入)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.cli import (
 add_json_arg,
 add_log_level_arg,
 make_parser,
 parse_leagues,
 run,
 setup_logging,
)

# 联赛 fdco 代码 → LeagueType 枚举
FDCO_TO_LEAGUE = {
 "E0": "PREMIER_LEAGUE",
 "SP1": "LA_LIGA",
 "D1": "BUNDESLIGA",
 "I1": "SERIE_A",
 "F1": "LIGUE_1",
}


def main() -> int:
 ap = make_parser("生产环境一键初始化:爬取数据 → 训练模型 → 验证预测(幂等)")
 ap.add_argument("--seasons", default="2016-2025", help="历史赛季范围(起始年)")
 ap.add_argument(
 "--leagues", default="E0,SP1,D1,I1,F1", help="fdco 联赛代码,逗号分隔"
 )
 ap.add_argument("--skip-data", action="store_true", help="跳过数据爬取")
 ap.add_argument("--skip-train", action="store_true", help="跳过模型训练")
 ap.add_argument(
 "--with-fixtures",
 action="store_true",
 help="额外爬取未来赛程(需 FOOTBALL_DATA_ORG_KEY)",
 )
 add_json_arg(ap)
 add_log_level_arg(ap)
 args = ap.parse_args()
 setup_logging(args.log_level)

 import json

 from app.api.db import init_db, session_scope
 from app.core.config import LeagueType
 from app.prediction.service import predict_match
 from app.services.training.trainer import train_model

 init_db()
 report = {"data": None, "training": None, "verification": None}
 league_codes = parse_leagues(args.leagues)
 invalid = [c for c in league_codes if c not in FDCO_TO_LEAGUE]
 if invalid:
 print(f"错误: 未知联赛代码 {invalid}(可选: E0,SP1,D1,I1,F1)", file=sys.stderr)
 return 2

 # ---------- 阶段 1:数据 ----------
 if not args.skip_data:
 print("\n===== 阶段 1/3:爬取数据 =====\n", flush=True)
 from app.data.pipeline import run_fixtures, run_history, run_xg

 seasons = [int(s) for s in str(args.seasons).split(",")]
 if len(seasons) == 2 and seasons[1] > seasons[0]:
 seasons = list(range(seasons[0], seasons[1] + 1))
 r_h = run_history(seasons, league_codes)
 r_x = run_xg(seasons, league_codes)
 report["data"] = {"history": r_h, "xg": r_x}
 print(
 f"历史: 新增 {r_h['inserted']} 更新 {r_h['updated']} 错误 {len(r_h['errors'])}"
 )
 print(f"xG: 更新 {r_x['updated']} 错误 {len(r_x['errors'])}")
 if args.with_fixtures:
 key = os.environ.get("FOOTBALL_DATA_ORG_KEY", "").strip()
 if not key:
 print("警告: 未设置 FOOTBALL_DATA_ORG_KEY,跳过赛程", file=sys.stderr)
 else:
 r_f = run_fixtures(max(seasons), ["PL", "PD", "BL1", "SA", "FL1"])
 report["data"]["fixtures"] = r_f
 print(f"赛程: 新增 {r_f['inserted']} 错误 {len(r_f['errors'])}")
 if r_h["errors"] or r_x["errors"]:
 print(
 f"警告: 数据爬取存在错误: {r_h['errors'][:3] + r_x['errors'][:3]}",
 file=sys.stderr,
 )
 else:
 print("\n===== 阶段 1/3:跳过数据爬取 =====\n")

 # ---------- 阶段 2:训练 ----------
 if not args.skip_train:
 print("\n===== 阶段 2/3:训练模型 =====\n", flush=True)
 results = {}
 with session_scope():
 for code in league_codes:
 lt = LeagueType[FDCO_TO_LEAGUE[code]]
 print(f"--- 训练 {lt.value} ---", flush=True)
 metrics = train_model(lt, "goals", True, 5)
 results[lt.value] = {
 "version": metrics.get("model_version"),
 "poisson_loss": round(float(metrics.get("poisson_loss", 0)), 4),
 "accuracy": round(float(metrics.get("exact_accuracy", 0)), 4),
 "features": metrics.get("feature_count"),
 }
 print(
 f" OK v{metrics.get('model_version')} "
 f"poisson={metrics.get('poisson_loss'):.4f} "
 f"acc={metrics.get('exact_accuracy'):.4f}"
 )
 report["training"] = results
 else:
 print("\n===== 阶段 2/3:跳过训练 =====\n")

 # ---------- 阶段 3:验证 ----------
 print("\n===== 阶段 3/3:验证预测 =====\n", flush=True)
 verified = []
 with session_scope():
 for code in league_codes:
 lt = LeagueType[FDCO_TO_LEAGUE[code]]
 try:
 r = predict_match(lt, "Arsenal FC", "Chelsea FC")
 verified.append(
 {
 "league": lt.value,
 "ok": True,
 "home_win": r["home_win_probability"],
 }
 )
 print(
 f" ✅ {lt.value}: 主胜 {r['home_win_probability']:.3f} "
 f"λ {r['raw_lambda_home']}/{r['raw_lambda_away']}"
 )
 except Exception as e:
 verified.append({"league": lt.value, "ok": False, "error": str(e)[:80]})
 print(f" ⚠️ {lt.value}: {str(e)[:80]}")
 report["verification"] = verified

 ok_count = sum(1 for v in verified if v["ok"])
 print("\n" + "=" * 56)
 print(
 f"初始化完成: 数据 {'✅' if report['data'] else '⏭️'} | "
 f"训练 {'✅' if report['training'] else '⏭️'} | "
 f"验证 {ok_count}/{len(verified)}"
 )
 print("=" * 56)
 if getattr(args, "json", False):
 print(json.dumps(report, ensure_ascii=False, default=str))
 return 0 if ok_count == len(verified) else 1


if __name__ == "__main__":
 raise SystemExit(run(main))
