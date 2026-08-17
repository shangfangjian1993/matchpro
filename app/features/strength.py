"""Feature Factory · 01_team_strength —— Team Strength(ELO)(§1.1 app/features/strength)。

规格声明(审查 P1-2:formula_hash 必须描述"公式"而非仅列名):
  FEATURES : 规格列表(name/window/agg/source/policy)——特征规格单一来源
  version() : 规格 JSON 哈希(特征 A/B 归因;增删/改公式即变化)
  compute() : 特征计算入口(滚动族由 factory 统一调度)
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd

FAMILY = "01_team_strength"
DESCRIPTION = "Team Strength(ELO)"
FEATURES: list[dict] = [
    {
        "name": "home_elo",
        "window": "all",
        "agg": "rating",
        "source": "teams.elo_rating",
        "policy": "time_replay_prematch"
    },
    {
        "name": "away_elo",
        "window": "all",
        "agg": "rating",
        "source": "teams.elo_rating",
        "policy": "time_replay_prematch"
    },
    {
        "name": "elo_diff",
        "window": "all",
        "agg": "diff",
        "source": "teams.elo_rating",
        "policy": "time_replay_prematch"
    },
    {
        "name": "home_attack_elo",
        "window": "all",
        "agg": "rating",
        "source": "teams.attack_elo",
        "policy": "time_replay_prematch"
    },
    {
        "name": "away_attack_elo",
        "window": "all",
        "agg": "rating",
        "source": "teams.attack_elo",
        "policy": "time_replay_prematch"
    },
    {
        "name": "attack_elo_diff",
        "window": "all",
        "agg": "diff",
        "source": "teams.attack_elo",
        "policy": "time_replay_prematch"
    },
    {
        "name": "home_defense_elo",
        "window": "all",
        "agg": "rating",
        "source": "teams.defense_elo",
        "policy": "time_replay_prematch"
    },
    {
        "name": "away_defense_elo",
        "window": "all",
        "agg": "rating",
        "source": "teams.defense_elo",
        "policy": "time_replay_prematch"
    },
    {
        "name": "defense_elo_diff",
        "window": "all",
        "agg": "diff",
        "source": "teams.defense_elo",
        "policy": "time_replay_prematch"
    }
]




def compute(df, league_type: str | None = None) -> pd.DataFrame:
    """附加三维 ELO 特征(赛前值,时间重放防泄漏)——真实现。"""
    from app.models.elo_goal.rating import with_elo_features
    return with_elo_features(df, is_national=(league_type in ("world_cup", "european_championship")))



def version() -> str:
    """公式哈希(审查 §16):规格 JSON + 实现代码哈希(implementation version)。

    规格变化或 compute 实现变化 → 版本变化(特征 A/B 归因)。
    审查七 V7-2 补漏:ELO 实现在 app/models/elo_goal/rating.py(Dynamic K 等),
    必须纳入哈希 —— 否则"实现变了但特征版本不变",模型不会触发重训。
    """
    import inspect
    spec = json.dumps(sorted(FEATURES, key=lambda f: f["name"]),
                      ensure_ascii=False, sort_keys=True)
    impl = inspect.getsource(compute)
    try:
        from app.models.elo_goal.rating import EloSystem, with_elo_features
        impl += "|" + inspect.getsource(with_elo_features)
        impl += "|" + inspect.getsource(EloSystem.update)
        impl += "|" + inspect.getsource(EloSystem._k_factor)
        impl += "|" + inspect.getsource(EloSystem.__init__)
    except Exception:
        pass  # 实现不可得时仅规格哈希
    raw = spec + "|" + impl
    return hashlib.sha256(raw.encode()).hexdigest()[:12]
