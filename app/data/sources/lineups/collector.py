"""首发阵容采集器

数据源现状(2026-08):
- api-football /lineups:付费套餐才有。Free 套餐实测返回
  "The Lineups endpoint does not exist." → fetch_by_fixture_apifootball()
  为预留接口,升级套餐后启用(API_FOOTBALL_KEY 已就绪)。
- TheSportsDB lookupeventlineup.php:免费但覆盖有限(需先按球队+日期匹配 event,
  部分联赛/赛季无数据)→ 当前兜底实现,复用 data/news 的匹配逻辑。

用法:
    python -m data.lineups.collector --home "Manchester City" --away "Arsenal" --date 2026-08-14
    python -m data.lineups.collector --event <thesportsdb_event_id>
环境变量:API_FOOTBALL_KEY(api-football 预留接口用)。
"""
import json
import os

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "raw", "lineups")


class LineupsCollector:
    """首发阵容采集器:当前 TSDB 兜底,api-football 付费端点预留。"""

    def __init__(self, cache_dir: str | None = None):
        from app.data.sources.http import default_cache_dir
        cache_dir = cache_dir or default_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        self._news = None  # 惰性加载,避免与 news 强耦合

    def _news_collector(self):
        """复用 data/news 的 TheSportsDB 匹配/拉取逻辑,避免重复实现"""
        if self._news is None:
            from app.data.sources.news.collector import NewsCollector
            self._news = NewsCollector()
        return self._news

    # ---------------- TheSportsDB 兜底(当前可用) ----------------
    def find_event(self, home: str, away: str, match_date: str) -> str | None:
        """按球队+日期匹配 TSDB 事件 id(覆盖有限,匹配不到返回 None)"""
        return self._news_collector().find_tsdb_event(home, away, match_date)

    def fetch_tsdb_lineup(self, event_id: str, use_cache: bool = True) -> dict:
        """拉取 TSDB 首发(含缓存);返回 {lineup 原始结构} 或 {}"""
        cache_file = os.path.join(self.cache_dir, f"tsdb_{event_id}.json")
        if use_cache and os.path.exists(cache_file):
            return json.load(open(cache_file, encoding="utf-8"))
        data = self._news_collector().fetch_tsdb_lineup(event_id)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        return data

    def build_summary(self, home: str, away: str, match_date: str) -> dict:
        """赛前首发简报:先匹配事件,再拉阵容,输出文本"""
        event_id = self.find_event(home, away, match_date)
        lineup = self.fetch_tsdb_lineup(event_id) if event_id else {}
        lines = [f"[赛前首发简报] {home} vs {away} ({match_date[:10]})"]
        if lineup:
            lines.append(f"- TheSportsDB 阵容: 已获取(事件 {event_id})")
            for side in ("home", "away"):
                key = f"str{side.title()}lineup"
                start = (lineup.get(key) or "").split(";")
                start = [p.strip() for p in start if p.strip()]
                if start:
                    lines.append(f"  {side} 首发({len(start)}人): " + ", ".join(start[:11]))
        else:
            lines.append("- TheSportsDB 阵容: 未匹配到(覆盖有限)")
        return {
            "event_id": event_id,
            "lineup_available": bool(lineup),
            "brief_text": "\n".join(lines),
            "source": "thesportsdb (fallback)",
        }

    # ---------------- api-football 预留(付费套餐) ----------------
    def fetch_by_fixture_apifootball(self, fixture_id: int) -> list[dict]:
        """api-football /lineups 预留接口(Free 套餐不可用,升级后启用)。

        数据结构(付费套餐):response = [{team, formation, startXI: [{player: {id,name,number,pos,grid}}],
        substitutes: [...]}, ...] 按主客两队排列。
        """
        from app.data.sources.http import http_get_json
        api_key = os.environ.get("API_FOOTBALL_KEY", "").strip()
        url = f"https://v3.football.api-sports.io/fixtures/lineups?fixture={fixture_id}"
        data = http_get_json(url, headers={"x-apisports-key": api_key})
        return data.get("response") or []


def _main() -> int:
    from app.services.cli import make_parser
    ap = make_parser("首发阵容采集器:api-football /lineups(付费预留)+ TheSportsDB 兜底")
    ap.add_argument("--home", help="主队名(走 TSDB 匹配)")
    ap.add_argument("--away", help="客队名")
    ap.add_argument("--date", help="比赛日期 YYYY-MM-DD(默认今天)")
    ap.add_argument("--event", help="直接指定 TheSportsDB 事件 id,跳过匹配")
    args = ap.parse_args()
    from app.services.cli import setup_logging
    setup_logging(getattr(args, "log_level", "INFO"))

    c = LineupsCollector()
    if args.event:
        lineup = c.fetch_tsdb_lineup(args.event)
        print(json.dumps({"event": args.event, "count": len(lineup)}, ensure_ascii=False))
        return 0
    if not (args.home and args.away):
        print("需提供 --home/--away/--date 或 --event")
        return 1
    from datetime import date
    s = c.build_summary(args.home, args.away, args.date or date.today().isoformat())
    print(s["brief_text"])
    return 0


if __name__ == "__main__":
    from app.services.cli import run
    raise SystemExit(run(_main))
