"""资讯采集器:免费零门槛源(ESPN RSS 新闻流 + TheSportsDB 阵容尽力拉取)

- ESPN RSS:全球足球新闻流,按球队名关键词过滤 → 赛前简报文本
- TheSportsDB:阵容拉取(能匹配到事件就用,匹配不到跳过——覆盖有限)
- 缓存:新闻按天缓存到 data/news/cache/,避免频繁请求触发限流
"""
import json
import os
import time
from datetime import datetime

ESPN_RSS_URL = "https://www.espn.com/espn/rss/soccer/news"
TSDB_BASE = "https://www.thesportsdb.com/api/v1/json/3"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"

# 常见队名别名 → 匹配关键词(用于 RSS 过滤)
TEAM_ALIASES = {
    "Deportivo Alavés": ["alaves", "alavés", "deportivo alaves"],
    "Getafe CF": ["getafe"],
    "Sevilla FC": ["sevilla"],
    "Rayo Vallecano de Madrid": ["rayo"],
    "Real Racing Club de Santander": ["racing santander", "santander"],
    "Villarreal CF": ["villarreal"],
    "RCD Espanyol de Barcelona": ["espanyol"],
    "Levante UD": ["levante"],
    "RC Celta de Vigo": ["celta"],
    "CA Osasuna": ["osasuna"],
    "RC Deportivo La Coruña": ["deportivo la coruna", "deportivo"],
    "Elche CF": ["elche"],
}


class NewsCollector:
    def __init__(self, cache_dir: str | None = None):
        from app.data.sources.http import default_cache_dir
        self.cache_dir = cache_dir or default_cache_dir()
        os.makedirs(self.cache_dir, exist_ok=True)

    # ---------------- HTTP 抓取(curl 子进程,urllib 指纹易被拦) ----------------
    @staticmethod
    def _http_get(url: str, timeout: int = 25, retries: int = 3) -> str:
        """curl 拉取 + 重试(统一实现见 data/_http.py)"""
        from app.data.sources.http import http_get
        return http_get(url, timeout, retries)

    @staticmethod
    def _parse_rss(xml: str) -> list[dict]:
        """RSS 解析(统一实现见 data/_http.py)"""
        from app.data.sources.http import parse_rss
        return parse_rss(xml)
    def fetch_espn_rss(self, timeout: int = 25, max_age_hours: int = 6) -> list[dict]:
        """抓取 ESPN 足球 RSS(带当日缓存);返回 [{title, desc, date, link}]"""
        from datetime import datetime
        today = datetime.now().strftime("%Y%m%d")
        cache_file = os.path.join(self.cache_dir, f"espn_rss_{today}.json")
        if os.path.exists(cache_file):
            age = time.time() - os.path.getmtime(cache_file)
            if age < max_age_hours * 3600:
                return json.load(open(cache_file, encoding="utf-8"))
        xml = self._http_get(ESPN_RSS_URL, timeout)
        items = self._parse_rss(xml)
        if not items:
            # 抓取失败:尝试 Sky 备用源
            xml2 = self._http_get("https://www.skysports.com/rss/12069", timeout)  # 足球频道
            items = [n for n in self._parse_rss(xml2)
                     if any(k in n["title"].lower() for k in ("football", "soccer", "premier", "la liga"))][:15]
            if not items:
                return []
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)
        return items

    def filter_by_team(self, news: list[dict], team: str) -> list[dict]:
        kws = TEAM_ALIASES.get(team, [team.lower().split()[-1]])
        out = []
        for n in news:
            text = (n["title"] + " " + n["desc"]).lower()
            if any(k in text for k in kws):
                out.append(n)
        return out

    # ---------------- TheSportsDB(尽力而为) ----------------
    def _tsdb(self, path: str, timeout: int = 15) -> dict:
        from app.data.sources.http import http_get_json
        return http_get_json(f"{TSDB_BASE}/{path}", timeout=timeout)
    def find_tsdb_event(self, home: str, away: str, match_date: str) -> str | None:
        try:
            d = self._tsdb(f"searchteams.php?t={home.split()[-1]}")
            team = (d.get("teams") or [{}])[0]
            lid = team.get("idLeague")
            if not lid:
                return None
            season = team.get("strCurrentSeason", "")
            year = season.split("-")[0] if season else (match_date[:4] if match_date else str(datetime.now().year))
            evs = self._tsdb(f"eventsseason.php?id={lid}&s={year}")
            want_date = match_date[:10]
            for e in (evs.get("events") or []):
                if e.get("dateEvent") == want_date and home.split()[-1].lower() in (e.get("strHomeTeam") or "").lower():
                    return e.get("idEvent")
        except Exception:
            return None
        return None

    def fetch_tsdb_lineup(self, event_id: str) -> dict:
        return self._tsdb(f"lookupeventlineup.php?id={event_id}")

    # ---------------- 简报 ----------------
    def build_brief(self, home: str, away: str, match_date: str) -> dict:
        """采集指定比赛的资讯简报:{news, lineup, brief_text, sources}"""
        news = self.fetch_espn_rss()
        home_news = self.filter_by_team(news, home)
        away_news = self.filter_by_team(news, away)
        event_id = self.find_tsdb_event(home, away, match_date)
        lineup = self.fetch_tsdb_lineup(event_id) if event_id else {}
        lines = [f"[赛前资讯简报] {home} vs {away} ({match_date[:10]})"]
        lines.append(f"- ESPN 新闻命中: 主队 {len(home_news)} 条, 客队 {len(away_news)} 条")
        for n in home_news + away_news:
            lines.append(f"  • {n['title']} ({n['date']})")
        if lineup:
            lines.append(f"- TheSportsDB 阵容: 已获取(事件 {event_id})")
        else:
            lines.append("- TheSportsDB 阵容: 未匹配到(覆盖有限)")
        return {
            "home_news": home_news, "away_news": away_news,
            "lineup_available": bool(lineup), "lineup": lineup,
            "brief_text": "\n".join(lines),
            "sources": ["espn_rss", "thesportsdb"],
        }
