"""采集器配置:源联赛代码 ↔ 系统 LeagueType ↔ 赛季命名规则"""

# football-data.co.uk CSV 文件代码 → LeagueType 枚举值
LEAGUE_MAP_FDCO = {
 "E0": "PREMIER_LEAGUE", # 英格兰超级联赛
 "SP1": "LA_LIGA", # 西班牙甲级联赛
 "D1": "BUNDESLIGA", # 德国甲级联赛
 "I1": "SERIE_A", # 意大利甲级联赛
 "F1": "LIGUE_1", # 法国甲级联赛
}

# football-data.org 竞赛代码 → LeagueType 枚举值
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


# fdco 命名:2526 = 2025/26 赛季(格式 "YYNN")
def fdco_season_code(year: int) -> str:
 """赛季起始年 → fdco 代码,如 2025 -> '2526'"""
 return f"{year % 100:02d}{(year + 1) % 100:02d}"


# fdo 命名:2026 = 2026/27 赛季(用起始年)
def fdo_season_code(year: int) -> int:
 return year


# fdco CSV 列 → matches 表字段(赔率列一律忽略——统计路线不读赔率)
FDCO_COLUMN_MAP = {
 "FTHG": "home_goals", # 全场主队进球
 "FTAG": "away_goals", # 全场客队进球
 "HTHG": "home_ht_goals", # 半场主队进球
 "HTAG": "away_ht_goals", # 半场客队进球
 "HS": "home_shots", # 主队射门
 "AS": "away_shots", # 客队射门
 "HST": "home_shots_on_target", # 主队射正
 "AST": "away_shots_on_target", # 客队射正
 "HC": "home_corners", # 主队角球
 "AC": "away_corners", # 客队角球
 "HY": "home_yellow_cards", # 主队黄牌
 "AY": "away_yellow_cards", # 客队黄牌
 "HR": "home_red_cards", # 主队红牌
 "AR": "away_red_cards", # 客队红牌
}

# understat 联赛代码 → LeagueType 枚举值
UNDERSTAT_LEAGUES = {
 "EPL": "PREMIER_LEAGUE",
 "La_liga": "LA_LIGA",
 "Bundesliga": "BUNDESLIGA",
 "Serie_A": "SERIE_A",
 "Ligue_1": "LIGUE_1",
}
