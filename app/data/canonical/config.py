"""统一数据清洗入库配置:指标注册表、联赛配置、源优先级。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ============================================================
# 指标注册表:所有数据源的字段名 → 统一字段名
# ============================================================

MATCH_FIELDS = [
    "league_type", "date", "home_team", "away_team",
    "season_id", "match_status", "match_stage",
]

GOAL_FIELDS = [
    "home_goals", "away_goals",
    "home_ht_goals", "away_ht_goals",
]

STAT_FIELDS = [
    "home_shots", "away_shots",
    "home_shots_on_target", "away_shots_on_target",
    "home_shots_inside_box", "away_shots_inside_box",
    "home_shots_outside_box", "away_shots_outside_box",
    "home_blocked_shots", "away_blocked_shots",
    "home_possession", "away_possession",
    "home_passing_accuracy", "away_passing_accuracy",
    "home_fouls", "away_fouls",
    "home_yellow_cards", "away_yellow_cards",
    "home_red_cards", "away_red_cards",
    "home_offsides", "away_offsides",
    "home_tackles", "away_tackles",
    "home_interceptions", "away_interceptions",
    "home_clearances", "away_clearances",
    "home_total_saves", "away_total_saves",
    "home_corners", "away_corners",
    "home_big_chances", "away_big_chances",
    "home_counter_attacks", "away_counter_attacks",
    "home_xg", "away_xg",
    "home_xg_chain", "away_xg_chain",
    "home_efficiency", "away_efficiency",
    "home_transition_speed", "away_transition_speed",
    "home_defensive_actions", "away_defensive_actions",
    "home_tactical_rating", "away_tactical_rating",
    "home_experience", "away_experience",
]

# 源字段映射: source_name → {源字段: 统一字段}
SOURCE_FIELD_MAPS: Dict[str, Dict[str, str]] = {
    "fdco": {
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals",
        "FTAG": "away_goals",
        "HTHG": "home_ht_goals",
        "HTAG": "away_ht_goals",
        "HS": "home_shots",
        "AS": "away_shots",
        "HST": "home_shots_on_target",
        "AST": "away_shots_on_target",
        "HC": "home_corners",
        "AC": "away_corners",
        "HY": "home_yellow_cards",
        "AY": "away_yellow_cards",
        "HR": "home_red_cards",
        "AR": "away_red_cards",
    },
    "bzzoiro": {
        "event_date": "date",
        "home_team": "home_team",
        "away_team": "away_team",
        "home_score": "home_goals",
        "away_score": "away_goals",
        "home_score_ht": "home_ht_goals",
        "away_score_ht": "away_ht_goals",
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
    },
    "understat": {
        "datetime": "date",
        "h": "home_team",
        "a": "away_team",
        "goals.h": "home_goals",
        "goals.a": "away_goals",
        "xG.h": "home_xg",
        "xG.a": "away_xg",
    },
}

@dataclass
class LeagueConfig:
    league_type: str
    fdco_code: str
    bzzoiro_id: int
    understat_code: str
    api_football_id: int
    fdo_code: str

LEAGUES: Dict[str, LeagueConfig] = {
    "premier_league": LeagueConfig("premier_league", "E0", 1, "EPL", 39, "PL"),
    "la_liga": LeagueConfig("la_liga", "SP1", 3, "La_liga", 140, "PD"),
    "bundesliga": LeagueConfig("bundesliga", "D1", 5, "Bundesliga", 78, "BL1"),
    "serie_a": LeagueConfig("serie_a", "I1", 4, "Serie_A", 135, "SA"),
    "ligue_1": LeagueConfig("ligue_1", "F1", 6, "Ligue_1", 61, "FL1"),
}

DATA_TYPE_PRIORITY: Dict[str, List[str]] = {
    "results": ["bzzoiro", "fdco"],
    "xg": ["understat", "bzzoiro"],
    "stats": ["bzzoiro", "api_football"],
    "odds": ["bzzoiro"],
    "tournaments": ["zafronix", "fdo"],
    "injuries": ["api_football"],
}

STATS_SEASONS = 20
TRAIN_SEASONS = 10
