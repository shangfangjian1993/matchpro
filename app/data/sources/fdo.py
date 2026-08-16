"""fdo 源适配器:football-data.org 赛程/赛果 API。"""
import json

from app.data.sources.http import http_get

FDO_BASE = "https://api.football-data.org/v4"


def fetch_fdo(league_code: str, season: int, api_key: str) -> list[dict]:
    """单赛季全部比赛(含未来赛程)。"""
    url = f"{FDO_BASE}/competitions/{league_code}/matches?season={season}"
    data = json.loads(http_get(url, headers={"X-Auth-Token": api_key}))
    return data.get("matches", [])
