from __future__ import annotations

from app.data.sources.http import SeasonNotAvailable  # noqa: F401

"""管线配置:赛季窗口、API endpoints、数据源参数。"""

from dataclasses import dataclass, field

# 采集窗口(赛季数)—— stats/odds 仅近 N 季(用户策略)
STATS_SEASONS = 20
# 训练窗口(训练只用最近 N 季;数据全量保留)
TRAIN_SEASONS = 10


# ---- Bzzoiro ----
BZZOIRO_BASE = "https://sports.bzzoiro.com/api/v2/_"

# bzzoiro league_id 映射(LeagueType.value → bzzoiro league_id)
BZZOIRO_LEAGUE_IDS = {
    "premier_league": 1,
    "la_liga": 3,
    "bundesliga": 5,
    "serie_a": 4,
    "ligue_1": 6,
    "champions_league": 7,
    "europa_league": 8,
}

# bzzoiro stats 字段映射(API key → DB column)
BZZOIRO_STATS_MAP = {
    "expected_goals": "xg",
    "total_shots": "shots",
    "shots_on_target": "shots_on_target",
    "ball_possession": "possession",
    "corner_kicks": "corners",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "fouls": "fouls",
    "offsides": "offsides",
    "tackles": "tackles",
    "interceptions": "interceptions",
    "clearances": "clearances",
    "blocked_shots": "blocked_shots",
    "big_chances": "big_chances",
    "total_saves": "total_saves",
    "shots_inside_box": "shots_inside_box",
    "shots_outside_box": "shots_outside_box",
}


# ---- FDCO (football-data.co.uk CSV) ----
FDCO_BASE = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

# fdco 联赛代码 → LeagueType 枚举值(大写)
LEAGUE_MAP_FDCO = {
    "E0": "PREMIER_LEAGUE",
    "SP1": "LA_LIGA",
    "D1": "BUNDESLIGA",
    "I1": "SERIE_A",
    "F1": "LIGUE_1",
}

# fdco 命名:2526 = 2025/26 赛季(格式 "YYNN")
def fdco_season_code(year: int) -> str:
    """赛季起始年 → fdco 代码,如 2025 -> '2526'"""
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


# fdco CSV 列 → matches 表字段(赔率列一律忽略——统计路线不读赔率)
FDCO_COLUMN_MAP = {
    "FTHG": "home_goals",  # 全场主队进球
    "FTAG": "away_goals",  # 全场客队进球
    "HTHG": "home_ht_goals",  # 半场主队进球
    "HTAG": "away_ht_goals",  # 半场客队进球
    "HS": "home_shots",  # 主队射门
    "AS": "away_shots",  # 客队射门
    "HST": "home_shots_on_target",  # 主队射正
    "AST": "away_shots_on_target",  # 客队射正
    "HC": "home_corners",  # 主队角球
    "AC": "away_corners",  # 客队角球
    "HY": "home_yellow_cards",  # 主队黄牌
    "AY": "away_yellow_cards",  # 客队黄牌
    "HR": "home_red_cards",  # 主队红牌
    "AR": "away_red_cards",  # 客队红牌
}


# ---- Understat ----
UNDERSTAT_BASE = "https://understat.com/getLeagueData/{league}/{season}"

# understat 联赛代码 → LeagueType 枚举值(大写)
UNDERSTAT_LEAGUES = {
    "EPL": "PREMIER_LEAGUE",
    "La_liga": "LA_LIGA",
    "Bundesliga": "BUNDESLIGA",
    "Serie_A": "SERIE_A",
    "Ligue_1": "LIGUE_1",
}

# fdco 联赛代码 → understat 联赛代码(--leagues 参数统一用 fdco 代码驱动两个源)
FDCO_TO_UNDERSTAT = {
    "E0": "EPL",
    "SP1": "La_liga",
    "D1": "Bundesliga",
    "I1": "Serie_A",
    "F1": "Ligue_1",
}


# ---- api-football (api-sports.io) ----
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

# api-football league id(5 大联赛)
API_FOOTBALL_LEAGUE_IDS = {
    "premier_league": 39,
    "la_liga": 140,
    "bundesliga": 78,
    "serie_a": 135,
    "ligue_1": 61,
}

# statistic type → (matches 列 主, 客)
API_FOOTBALL_STAT_MAP = {
    "Total Shots": ("home_shots", "away_shots"),
    "Shots on Goal": ("home_shots_on_target", "away_shots_on_target"),
    "Ball Possession": ("home_possession", None),  # 客队 = 100 - 主
    "Corner Kicks": ("home_corners", "away_corners"),
    "Yellow Cards": ("home_yellow_cards", "away_yellow_cards"),
    "Red Cards": ("home_red_cards", "away_red_cards"),
    "Passes %": ("home_passing_accuracy", "away_passing_accuracy"),
}

API_FOOTBALL_TEAM_ALIAS = {
    "manchester united": "man united",
    "manchester city": "man city",
    "wolverhampton": "wolves",
    "nottingham": "nottingham",
    "brighton": "brighton",
    "west ham": "west ham",
    "paris saint germain": "psg",
    "paris saintgermain": "psg",
    "internazionale milano": "inter milan",
    "inter milan": "inter milan",
    "borussia monchengladbach": "gladbach",
    "borussia dortmund": "dortmund",
    "germain": "psg",
}


# ---- football-data.org ----
# fdo 竞赛代码 → LeagueType 枚举值(大写)
LEAGUE_MAP_FDO = {
    "PL": "PREMIER_LEAGUE",
    "PD": "LA_LIGA",
    "BL1": "BUNDESLIGA",
    "SA": "SERIE_A",
    "FL1": "LIGUE_1",
    "CL": "CHAMPIONS_LEAGUE",
    "EL": "EUROPA_LEAGUE",
    "WC": "WORLD_CUP",
    "EC": "EUROPEAN_CHAMPIONSHIP",
}

# fdo 命名:2026 = 2026/27 赛季(用起始年)
def fdo_season_code(year: int) -> int:
    return year


# ---- StatsBomb ----
# statsbomb 开放数据(只到 2020/21)—— 历史 xG 补充源
STATSBOMB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# ---- zafronix ----
# zafronix(欧战/国家队数据采集,具体 endpoint 由采集器配置)
ZAFRONIX_BASE = "https://api.zafronix.com/v1"


# ---- HTTP 客户端 ----
HTTP_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (football-prediction-pipeline)"
)
HTTP_DEFAULT_TIMEOUT = 40


# ---- 限速 ----
REQUEST_INTERVAL = 1.2  # 源礼貌限速(秒)
API_FOOTBALL_INTERVAL = 0.4  # api-free 限速更严

