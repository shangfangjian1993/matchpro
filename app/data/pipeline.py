"""统一数据同步管道:来源 → 清洗 → 分类 → 入库。

用法:
 DATABASE_URL=sqlite:///data/football.db python -m data.data_ingestion.sync \
 --job history --seasons 2016-2025 --leagues E0,SP1,D1,I1,F1
 ... --job fixtures --fdo-leagues PL,PD,BL1,SA,FL1 --season 2026 # 需 FOOTBALL_DATA_ORG_KEY
 ... --job xg --seasons 2016-2025
 ... --job all # history + xg(+fixtures,有 key 时)

管道: sources.fetch_* → cleanse.cleanse_* → ingest.upsert_matches
所有清洗(队名归一化/日期/字段)在入库前完成,入库后不再改名补数。
"""

import logging
import os
import sys
import time

from app.data import sources
from app.data.canonical import cleanse, ingest
from app.data.canonical.config import (
 LEAGUE_MAP_FDCO,
 LEAGUE_MAP_FDO,
 fdco_season_code,
)

logger = logging.getLogger(__name__)

REQUEST_INTERVAL = 1.2 # 源礼貌限速


def _league_known_teams(league_type: str) -> set:
 """该联赛已知队名集合:历史完赛 + 本赛季赛程队名。

 用于数据源白名单校验(fdco 新赛季 CSV 偶发内容错位,如 2627/SP1 曾返回葡超)。
 升班马队名在本赛季赛程(scheduled)中存在,天然包含。
 """
 from app.api.db import League, Match
 from app.core.config import LeagueType as _LT
 from app.data.canonical.store import _app_ctx

 # 入库用枚举小写值(premier_league);外部参数可能是大写枚举名,统一转换
 try:
 league_type_val = _LT[league_type].value
 except KeyError:
 league_type_val = league_type.lower()
 _, db = _app_ctx()
 from app.api.db import session_scope

 with session_scope():
 league = db.session.query(League).filter_by(league_type=league_type_val).first()
 if league is None:
 return set()
 known = {
 r[0]
 for r in db.session.query(Match.home_team)
 .filter(Match.league_id == league.id, Match.match_status == "finished")
 .distinct()
 }
 known |= {
 r[0]
 for r in db.session.query(Match.away_team)
 .filter(Match.league_id == league.id, Match.match_status == "finished")
 .distinct()
 }
 known |= {
 r[0]
 for r in db.session.query(Match.home_team)
 .filter(Match.league_id == league.id, Match.match_status == "scheduled")
 .distinct()
 }
 return known


def run_history(seasons: list[int], league_codes: list[str]) -> dict:
 """fdco 历史:下载 → 清洗 → upsert"""
 total = {"inserted": 0, "updated": 0, "skipped": 0, "errors": []}
 all_unmatched = set()
 for year in seasons:
 season_code = fdco_season_code(year)
 for league in league_codes:
 if league not in LEAGUE_MAP_FDCO:
 logger.error("未知 fdco 联赛 %s", league)
 continue
 lt = LEAGUE_MAP_FDCO[league]
 try:
 rows = sources.fetch_fdco(season_code, league)
 except Exception as e:
 logger.error("下载失败 %s/%s: %s", season_code, league, e)
 total["errors"].append(f"{season_code}/{league}: {e}")
 continue
 # 白名单校验:该联赛已知队名(历史+赛程),防止数据源内容错位混入
 known_teams = _league_known_teams(lt)
 unmatched = []
 cleaned = []
 for row in rows:
 m = cleanse.cleanse_fdco_row(row, lt, unmatched)
 if m is None:
 continue
 if known_teams and (
 m.home_team not in known_teams or m.away_team not in known_teams
 ):
 total["errors"].append(
 f"{season_code}/{league}: 队名不在 {lt} 已知集合 "
 f"({m.home_team} vs {m.away_team})"
 )
 continue
 cleaned.append(m)
 all_unmatched.update(unmatched)
 r = ingest.upsert_matches(cleaned)
 for k in ("inserted", "updated", "skipped"):
 total[k] += r[k]
 total["errors"] += r["errors"]
 logger.info(
 "%s/%s(%s): 新增 %d 更新 %d 跳过 %d%s",
 season_code,
 league,
 lt,
 r["inserted"],
 r["updated"],
 r["skipped"],
 f",未匹配 {len(unmatched)}" if unmatched else "",
 )
 time.sleep(REQUEST_INTERVAL)
 if all_unmatched:
 logger.warning(
 "未命中队名映射 %d 个: %s", len(all_unmatched), sorted(all_unmatched)[:20]
 )
 return total


def run_fixtures(season: int, fdo_leagues: list[str]) -> dict:
 """fdo 赛程/赛果(含未来赛程):下载 → 清洗 → upsert"""
 api_key = os.environ.get("FOOTBALL_DATA_ORG_KEY", "").strip()
 if not api_key:
 print(
 "缺少 FOOTBALL_DATA_ORG_KEY。免费注册: https://www.football-data.org/register"
 )
 sys.exit(2)
 total = {"inserted": 0, "updated": 0, "skipped": 0, "errors": []}
 for code in fdo_leagues:
 if code not in LEAGUE_MAP_FDO:
 logger.error("未知 fdo 竞赛 %s", code)
 continue
 lt = LEAGUE_MAP_FDO[code]
 try:
 rows = sources.fetch_fdo(code, season, api_key)
 except Exception as e:
 logger.error("fdo %s/%d 失败: %s", code, season, e)
 total["errors"].append(f"{code}/{season}: {e}")
 continue
 cleaned = [
 m for m in (cleanse.cleanse_fdo_row(r, lt) for r in rows) if m is not None
 ]
 r = ingest.upsert_matches(cleaned)
 for k in ("inserted", "updated", "skipped"):
 total[k] += r[k]
 logger.info(
 "fdo %s/%d: 新增 %d 更新 %d 跳过 %d",
 code,
 season,
 r["inserted"],
 r["updated"],
 r["skipped"],
 )
 time.sleep(REQUEST_INTERVAL)
 return total


# fdco 联赛代码 → understat 联赛代码(--leagues 参数统一用 fdco 代码驱动两个源)
FDCO_TO_UNDERSTAT = {
 "E0": "EPL",
 "SP1": "La_liga",
 "D1": "Bundesliga",
 "I1": "Serie_A",
 "F1": "Ligue_1",
}


def run_xg(seasons: list[int], league_codes: list[str]) -> dict:
 """understat xG:下载 → 清洗(只带 xG)→ upsert 补字段"""
 from app.data.canonical.config import UNDERSTAT_LEAGUES

 total = {"inserted": 0, "updated": 0, "skipped": 0, "errors": []}
 all_unmatched = set()
 for year in seasons:
 for code in league_codes:
 code = FDCO_TO_UNDERSTAT.get(code, code)
 if code not in UNDERSTAT_LEAGUES:
 logger.error("未知 understat 联赛 %s", code)
 continue
 lt = UNDERSTAT_LEAGUES[code]
 try:
 rows = sources.fetch_understat(code, year)
 except Exception as e:
 logger.error("understat %s/%d 失败: %s", code, year, e)
 total["errors"].append(f"{code}/{year}: {e}")
 continue
 unmatched = []
 cleaned = []
 for row in rows:
 m = cleanse.cleanse_understat_row(row, lt, unmatched)
 if m is not None:
 cleaned.append(m)
 all_unmatched.update(unmatched)
 r = ingest.upsert_matches(cleaned)
 for k in ("inserted", "updated", "skipped"):
 total[k] += r[k]
 logger.info(
 "xg %s/%d: 更新 %d 跳过 %d%s",
 code,
 year,
 r["updated"],
 r["skipped"],
 f",未匹配 {len(unmatched)}" if unmatched else "",
 )
 time.sleep(REQUEST_INTERVAL)
 if all_unmatched:
 logger.warning(
 "未命中队名映射 %d 个: %s", len(all_unmatched), sorted(all_unmatched)[:20]
 )
 return total


def _parse_seasons(s: str) -> list[int]:
 if "-" in s and "," not in s:
 a, b = s.split("-")
 return list(range(int(a), int(b) + 1))
 return [int(x) for x in s.split(",")]


def main():
 from app.services.cli import add_json_arg, add_log_level_arg, make_parser

 ap = make_parser("统一数据同步管道:来源 → 清洗 → 分类 → 入库")
 add_json_arg(ap)
 add_log_level_arg(ap)
 ap.add_argument(
 "--job",
 required=True,
 choices=["history", "fixtures", "xg", "all"],
 help="history=fdco历史回填; fixtures=fdo赛程(需key); xg=understat回填; all=全部",
 )
 ap.add_argument(
 "--seasons", default="2016-2025", help="赛季范围(起始年),如 2016-2025 或 2026"
 )
 ap.add_argument(
 "--leagues", default=",".join(LEAGUE_MAP_FDCO), help="fdco/understat 联赛代码"
 )
 ap.add_argument(
 "--fdo-leagues", default=",".join(LEAGUE_MAP_FDO), help="fdo 竞赛代码"
 )
 ap.add_argument(
 "--season", type=int, default=None, help="fdo 单赛季(默认取 --seasons 的最大值)"
 )
 args = ap.parse_args()

 from app.services.cli import setup_logging

 setup_logging(getattr(args, "log_level", "INFO"))

 seasons = _parse_seasons(args.seasons)
 leagues = [l.strip().upper() for l in args.leagues.split(",") if l.strip()]
 fdo_leagues = [l.strip().upper() for l in args.fdo_leagues.split(",") if l.strip()]

 total = {"inserted": 0, "updated": 0, "skipped": 0, "errors": []}
 if args.job in ("history", "all"):
 r = run_history(seasons, leagues)
 for k in total:
 total[k] += r[k]
 if args.job in ("fixtures", "all"):
 r = run_fixtures(args.season or max(seasons), fdo_leagues)
 for k in total:
 total[k] += r[k]
 if args.job in ("xg", "all"):
 r = run_xg(seasons, leagues)
 for k in total:
 total[k] += r[k]

 if getattr(args, "json", False):
 import json as _json

 print(
 _json.dumps(
 {
 "job": args.job,
 "inserted": total["inserted"],
 "updated": total["updated"],
 "skipped": total["skipped"],
 "errors": total["errors"][:5],
 },
 ensure_ascii=False,
 )
 )
 else:
 print(
 f"\n==== 管道完成: 新增 {total['inserted']},更新 {total['updated']},"
 f"跳过 {total['skipped']},错误 {len(total['errors'])} ===="
 )
 if total["errors"]:
 if not getattr(args, "json", False):
 print("错误明细:", total["errors"][:5])
 sys.exit(1)


if __name__ == "__main__":
 from app.services.cli import run

 raise SystemExit(run(main))
