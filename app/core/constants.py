"""核心常量(§1.1 app/core/constants):与 config.py 分离的纯常量。"""

from __future__ import annotations

from app.core.config import LeagueType

# 赛事型联赛(非联赛循环赛;训练/加载走赛事模型工厂)
TOURNAMENT_LEAGUE_TYPES = frozenset(
    {
        LeagueType.WORLD_CUP,
        LeagueType.EUROPEAN_CHAMPIONSHIP,
        LeagueType.CHAMPIONS_LEAGUE,
        LeagueType.EUROPA_LEAGUE,
    }
)

# Match 指标列权威白名单(单一定义源;新增指标列只需改这一处 + db.Match)
MATCH_METRIC_COLUMNS = (
    "home_xg",
    "away_xg",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_corners",
    "away_corners",
    "home_possession",
    "home_yellow_cards",
    "away_yellow_cards",
    "home_red_cards",
    "away_red_cards",
    "home_ht_goals",
    "away_ht_goals",
    "home_passing_accuracy",
    "away_passing_accuracy",
    "home_xg_chain",
    "away_xg_chain",
    "home_efficiency",
    "away_efficiency",
    "home_transition_speed",
    "away_transition_speed",
    "home_defensive_actions",
    "away_defensive_actions",
    "home_counter_attacks",
    "away_counter_attacks",
    "home_tactical_rating",
    "away_tactical_rating",
    "home_experience",
    "away_experience",
    "match_stage",
)
