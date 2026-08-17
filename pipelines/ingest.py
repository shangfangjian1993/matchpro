"""Pipeline 1:ingest —— Raw→Canonical(数据引擎)。

幂等:重复执行不产生重复数据(入库去重 + 白名单校验)。
可选:SoccerData(FBref)enrich 回填扩展统计列:
  python pipelines/ingest.py --enrich-soccerdata --league premier_league [--season 2024]
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    from app.services.cli import add_log_level_arg, make_parser, setup_logging
    ap = make_parser("Pipeline ingest:同步赛果/xG/赛程/伤停 + 可选 FBref enrich")
    ap.add_argument("--enrich-soccerdata", action="store_true",
                    help="用 SoccerData(FBref)回填统计(FBref 在数据中心 IP 有验证码,建议 NAS) ")
    ap.add_argument("--enrich-apifootball", action="store_true",
                    help="用 api-football(正式 API,无验证码)回填统计;推荐替代 FBref")
    ap.add_argument("--league", default=None,
                    help="enrich 联赛(默认全部 5 大联赛)")
    ap.add_argument("--season", default="2024",
                    help="enrich 赛季起始年(默认 2024)")
    add_log_level_arg(ap)
    args = ap.parse_args()
    setup_logging(args.log_level)

    from app.services.data.auto_sync import run_sync_job
    run_sync_job("all")
    print("✅ [pipeline] ingest 基础数据完成")

    if args.enrich_soccerdata:
        _run_enrich(args.league, args.season)
    if args.enrich_apifootball:
        _run_enrich_apifootball(args.league, args.season)


def _run_enrich_apifootball(league: str | None, season: str):
    from app.core.config import LeagueType
    from app.api.db import League, Match, init_db, session_scope
    from app.data.sources.api_football_stats import available as af_available
    from app.data.sources.api_football_stats import enrich_matches as af_enrich

    if not af_available():
        print("⚠️ 未配置 API_FOOTBALL_KEY,跳过 api-football 回填")
        return
    init_db()
    leagues = [LeagueType(league)] if league else [
        LeagueType.PREMIER_LEAGUE, LeagueType.LA_LIGA, LeagueType.BUNDESLIGA,
        LeagueType.SERIE_A, LeagueType.LIGUE_1]
    with session_scope():
        for lt in leagues:
            lg = League.query.filter_by(league_type=lt.value).first()
            if lg is None:
                continue
            rows = (Match.query.filter_by(league_id=lg.id, match_status="finished")
                    .filter((Match.home_shots.is_(None)) | (Match.home_possession.is_(None)))
                    .order_by(Match.match_date.desc()).limit(500).all())
            print(f"  {lt.value}: 待回填 {len(rows)} 场(api-football)")
            if not rows:
                continue
            try:
                res = af_enrich(lt, rows, season=season)
                print(f"    → 更新 {res['updated']} | 未匹配 {res['unmatched']} | "
                      f"错误 {res['errors']} | 调用 {res['calls']}(free 100/天)")
            except Exception as e:
                print(f"    ✗ {type(e).__name__}: {e}")
    print("✅ [pipeline] api-football 统计回填完成")


def _run_enrich(league: str | None, season: str):
    from app.core.config import LeagueType
    from app.api.db import League, Match, init_db, session_scope
    from app.data.sources.soccerdata_enrich import available as sd_available
    from app.data.sources.soccerdata_enrich import enrich_matches

    if not sd_available():
        print("⚠️ soccerdata 未安装,跳过 enrich(可选数据源,不影响基础链路)")
        return
    init_db()
    leagues = [LeagueType(league)] if league else [
        LeagueType.PREMIER_LEAGUE, LeagueType.LA_LIGA, LeagueType.BUNDESLIGA,
        LeagueType.SERIE_A, LeagueType.LIGUE_1]
    with session_scope():
        for lt in leagues:
            lg = League.query.filter_by(league_type=lt.value).first()
            if lg is None:
                continue
            # 已完赛且扩展列缺失(至少 xg 缺失)的场次
            rows = (Match.query.filter_by(league_id=lg.id, match_status="finished")
                    .filter((Match.home_xg.is_(None)) | (Match.away_xg.is_(None)))
                    .order_by(Match.match_date.desc()).limit(4000).all())
            print(f"  {lt.value}: 待回填 {len(rows)} 场")
            if not rows:
                continue
            try:
                res = enrich_matches(lt, rows, season_start=season)
                print(f"    → 更新 {res['updated']} | 未匹配 {res['unmatched']} | 错误 {res['errors']} | 源 {res['fetched']} 行")
            except Exception as e:
                print(f"    ✗ {type(e).__name__}: {e}")
    print("✅ [pipeline] SoccerData enrich 完成")


if __name__ == "__main__":
    from app.services.cli import run
    raise SystemExit(run(main))
