"""伤停数据采集器:api-football(api-sports.io)/injuries 端点

- 数据:球员伤停记录(player/team/fixture/league + type: Missing Fixture / Suspended + reason)
- 配额:Free 套餐每日 100 次请求;按日期拉取 1 次覆盖当天全部联赛/场次(最省配额)
- 缓存:按日期/场次缓存到 data/injuries/cache/,当日命中不再请求
- 队名:统一走 data/data_ingestion/team_names.normalize 归一化

用法:
 python -m data.injuries.collector --date 2026-08-14
 python -m data.injuries.collector --fixture 1561360
环境变量:API_FOOTBALL_KEY(必填)、API_FOOTBALL_HOST(默认 v3.football.api-sports.io)。
"""

import datetime
import json
import os
import time

API_BASE = "https://v3.football.api-sports.io"
DEFAULT_HOST = "v3.football.api-sports.io"

# 缓存目录:data/injuries/cache(与 news 采集器同模式)
_CACHE_DIR = os.path.join(
 os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
 "raw",
 "injuries",
)


class InjuriesCollector:
 """api-football 伤停数据采集器。"""

 def __init__(self, cache_dir: str | None = None):
 from app.data.sources.http import default_cache_dir

 cache_dir = cache_dir or default_cache_dir()
 os.makedirs(cache_dir, exist_ok=True)
 self.cache_dir = cache_dir
 self.api_key = os.environ.get("API_FOOTBALL_KEY", "").strip()
 self.api_host = os.environ.get("API_FOOTBALL_HOST", DEFAULT_HOST).strip()
 if not self.api_key:
 raise RuntimeError(
 "缺少 API_FOOTBALL_KEY 环境变量(api-football 伤停数据 key,"
 "见 .env.example)"
 )

 # ---------------- HTTP ----------------
 def _headers(self) -> dict:
 return {
 "x-rapidapi-key": self.api_key,
 "x-rapidapi-host": self.api_host,
 "Accept": "application/json",
 }

 def _get_json(
 self, url: str, headers: dict | None = None, timeout: int = 20
 ) -> dict:
 from app.data.sources.http import http_get_json

 return http_get_json(url, headers=headers, timeout=timeout)

 def _check_errors(self, data: dict) -> None:
 """响应 errors 非空时抛错(统一实现见 data/_http.py)"""
 from app.data.sources.http import check_api_errors

 err = check_api_errors(data)
 if err:
 raise RuntimeError(f"api-football 返回错误: {err}")

 @staticmethod
 def dedupe(records: list[dict]) -> list[dict]:
 """API 会对同一球员同一场次返回重复记录,按 (player.id, fixture.id) 去重(保序)。

 实测 2026-08-14 /injuries?date= 返回 3 名球员各 2 条完全相同的记录。
 """
 seen = set()
 out = []
 for r in records:
 key = (
 (r.get("player") or {}).get("id"),
 (r.get("fixture") or {}).get("id"),
 )
 if key in seen:
 continue
 seen.add(key)
 out.append(r)
 return out

 # ---------------- 采集(带缓存) ----------------
 def fetch_by_date(
 self, day: str | None = None, use_cache: bool = True
 ) -> list[dict]:
 """按日期拉取全部伤停记录(1 次请求覆盖当天所有联赛)。

 返回 [{player, team, fixture, league, type, reason, ...}]。
 """
 day = day or datetime.datetime.now(datetime.timezone.utc).date().isoformat()
 cache_file = os.path.join(self.cache_dir, f"date_{day}.json")
 if use_cache and os.path.exists(cache_file):
 age = time.time() - os.path.getmtime(cache_file)
 if age < 24 * 3600:
 return self.dedupe(json.load(open(cache_file, encoding="utf-8")))
 data = self._get_json("injuries", {"date": day})
 self._check_errors(data)
 records = self.dedupe(data.get("response") or [])
 with open(cache_file, "w", encoding="utf-8") as f:
 json.dump(records, f, ensure_ascii=False, indent=1)
 return records

 def fetch_by_fixture(self, fixture_id: int, use_cache: bool = True) -> list[dict]:
 """按场次拉取伤停记录"""
 cache_file = os.path.join(self.cache_dir, f"fixture_{fixture_id}.json")
 if use_cache and os.path.exists(cache_file):
 age = time.time() - os.path.getmtime(cache_file)
 if age < 24 * 3600:
 return self.dedupe(json.load(open(cache_file, encoding="utf-8")))
 data = self._get_json("injuries", {"fixture": fixture_id})
 self._check_errors(data)
 records = self.dedupe(data.get("response") or [])
 with open(cache_file, "w", encoding="utf-8") as f:
 json.dump(records, f, ensure_ascii=False, indent=1)
 return records

 # ---------------- 队名过滤 ----------------
 @staticmethod
 def _norm(name: str) -> str:
 from app.data.canonical.team_names import normalize

 n = normalize(str(name))
 return " ".join(n.lower().split())

 def filter_by_team(self, records: list[dict], team_name: str) -> list[dict]:
 """按球队过滤伤停记录(归一化匹配:支持简写/大小写/空格差异)"""
 target = self._norm(team_name)
 out = []
 for r in records:
 t = ((r.get("team") or {}).get("name") or "").strip()
 if not t:
 continue
 if target and (
 self._norm(t) == target
 or self._norm(t) in target
 or target in self._norm(t)
 ):
 out.append(r)
 return out

 # ---------------- 简报 ----------------
 def build_brief(self, home: str, away: str, match_date: str | None = None) -> dict:
 """采集指定比赛的伤停简报(与 data/news 简报同模式)。"""
 day = (
 match_date
 or datetime.datetime.now(datetime.timezone.utc).date().isoformat()
 )[:10]
 records = self.fetch_by_date(day)
 home_inj = self.filter_by_team(records, home)
 away_inj = self.filter_by_team(records, away)

 def _fmt(rs):
 return [
 {
 "player": r["player"]["name"],
 "type": r["player"].get("type", ""),
 "reason": r["player"].get("reason", ""),
 "fixture_id": r["fixture"]["id"],
 "team": r["team"]["name"],
 }
 for r in rs
 ]

 lines = [f"[赛前伤停简报] {home} vs {away} ({day})"]
 lines.append(
 f"- api-football 伤停: 主队 {len(home_inj)} 人, 客队 {len(away_inj)} 人"
 )
 for tag, rs in (("主队", home_inj), ("客队", away_inj)):
 for it in _fmt(rs):
 lines.append(
 f" • [{it['type']}] {it['player']} ({it['reason']}) — {tag}"
 )
 return {
 "home_injuries": _fmt(home_inj),
 "away_injuries": _fmt(away_inj),
 "total_records": len(records),
 "brief_text": "\n".join(lines),
 "source": "api-football/injuries",
 }


def _main() -> int:
 from app.services.cli import make_parser

 ap = make_parser("伤停数据采集器:api-football /injuries 端点(需 API_FOOTBALL_KEY)")
 ap.add_argument("--date", help="按日期拉取(YYYY-MM-DD,默认今天)")
 ap.add_argument("--fixture", type=int, help="按场次拉取(fixture id)")
 ap.add_argument("--team", help="仅显示该队伤停(配合 --date/--fixture)")
 args = ap.parse_args()
 from app.services.cli import setup_logging

 setup_logging(getattr(args, "log_level", "INFO"))

 try:
 c = InjuriesCollector()
 except RuntimeError as e:
 print(f"错误: {e}")
 return 1

 if args.fixture:
 recs = c.fetch_by_fixture(args.fixture)
 label = f"fixture {args.fixture}"
 else:
 day = (
 args.date or datetime.datetime.now(datetime.timezone.utc).date().isoformat()
 )
 recs = c.fetch_by_date(day)
 label = day

 if args.team:
 recs = c.filter_by_team(recs, args.team)
 print(
 json.dumps(
 {
 "source": "api-football/injuries",
 "query": label,
 "team_filter": args.team,
 "count": len(recs),
 "records": recs,
 },
 ensure_ascii=False,
 indent=1,
 )
 )
 return 0


if __name__ == "__main__":
 from app.services.cli import run

 raise SystemExit(run(_main))
