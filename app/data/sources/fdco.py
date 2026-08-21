"""fdco 源适配器:football-data.co.uk 历史 CSV。"""

import csv
import io

from app.data.sources.http import http_get

FDCO_BASE = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"


def fetch_fdco(season_code: str, league_code: str) -> list[dict]:
 """单赛季单联赛 CSV → 原始行 dict 列表(不做清洗)。"""
 url = FDCO_BASE.format(season=season_code, league=league_code)
 text = http_get(url).decode("utf-8-sig", errors="replace")
 return list(csv.DictReader(io.StringIO(text)))
