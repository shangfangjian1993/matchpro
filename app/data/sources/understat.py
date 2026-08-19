"""understat 源适配器:xG 数据。"""

import json

from app.data.sources.http import http_get

UNDERSTAT_BASE = "https://understat.com/getLeagueData/{league}/{season}"


def fetch_understat(league_code: str, season: int) -> list[dict]:
    """单赛季比赛数组(dates),含 xG。"""
    data = json.loads(
        http_get(
            UNDERSTAT_BASE.format(league=league_code, season=season),
            headers={
                "X-Requested-With": "XMLHttpRequest",
                "Referer": f"https://understat.com/league/{league_code}/{season}",
            },
        )
    )
    return data.get("dates", [])
