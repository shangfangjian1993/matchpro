"""特征注册表:特征列 → 6 大族分类 + 版本管理。

六大特征族(V2 方案):
01 Team Strength       ELO 相关
02 Attack/Defense      进失球/xG/射门攻防
03 Form & Momentum     近期状态/主场客场
04 Squad Availability 伤停/阵容(当前无特征,预留)
05 Match Environment   赛程/环境(预留)
06 Opponent Interaction 交手记录(H2H,已降权)

版本管理:
- 特征版本 = 特征列集合的稳定哈希(任何特征增删 → 版本变化)
- 训练后自动注册到 feature_store,支持特征维度归因/回滚
"""
from __future__ import annotations

import hashlib
import re

# 特征列名 → 族:命名模式分类(与 6 家族 FEATURES 声明互补,§1.1)
FAMILY_PATTERNS = [
    (r"(^|_)elo($|_|diff)", "01_team_strength"),
    ((r"(goals_avg|conceded_avg|_xg_|shots|sot|corners|passing|efficiency|transition|"
      r"defensive_actions|counter|tactical|experience|xg_chain|possession|ht_goals)"), "02_attack_defense"),
    ((r"(form|win_rate|recent|home_form|away_form|home_goals_avg|away_goals_avg|"
      r"home_conceded|away_conceded|home_win|away_win)"), "03_form_momentum"),
    (r"(availability|injured)", "04_squad_availability"),
    (r"(stage|season|league_)", "05_match_environment"),
    (r"h2h", "06_opponent_interaction"),
]
DEFAULT_FAMILY = "02_attack_defense"


def family_of(feature: str) -> str:
    """特征列 → 族名。"""
    for pattern, family in FAMILY_PATTERNS:
        if re.search(pattern, feature):
            return family
    return DEFAULT_FAMILY


def feature_version(feature_columns: list[str]) -> str:
    """特征集版本:列集合稳定哈希前 12 位(增删特征即变化)。"""
    if not feature_columns:
        return "empty"
    joined = "|".join(sorted(feature_columns))
    return hashlib.sha256(joined.encode()).hexdigest()[:12]


def summarize(feature_columns: list[str]) -> dict:
    """特征构成摘要:各族特征数 + 版本。"""
    families: dict[str, list[str]] = {}
    for f in feature_columns:
        families.setdefault(family_of(f), []).append(f)
    return {
        "version": feature_version(feature_columns),
        "total": len(feature_columns),
        "families": {k: {"count": len(v), "features": v} for k, v in sorted(families.items())},
    }


def compute_all(df, league_type: str | None = None):
    """统一特征计算入口(评审 P0-1.2):按家族顺序调用 compute()。

    当前:strength(ELO)真实现,其余家族滚动特征由模型层生成(compute 为声明);
    未来:各家族逐步接管,支持增量计算。
    """
    from app.features.strength import compute as _strength
    # ELO 需在滚动特征前注入(特征依赖 ELO 列)
    return _strength(df, league_type)


def logical_version() -> str:
    """逻辑哈希:聚合 6 家族版本(评审:特征 A/B 归因用)。"""
    import hashlib as _hl

    from app.features import attack_defense, form, h2h, strength
    parts = [m.version() for m in (strength, attack_defense, form, h2h)]
    return _hl.sha256("|".join(parts).encode()).hexdigest()[:12]


def register(feature_columns: list[str], league_type: str) -> str:
    """训练后注册特征集到 feature_store;返回特征版本。"""
    from app.api.db import FeatureStore, db
    version = feature_version(feature_columns)
    # 审查 §19:formula_hash = 公式规格哈希(规格+实现),与 version() 同源,
    # 不再是 feature name hash(两个概念并存已消除)
    formula_hash = logical_version()
    for f in feature_columns:
        exists = db.session.query(FeatureStore).filter_by(
            league_type=league_type, feature_name=f, version=version).first()
        if exists is None:
            db.session.add(FeatureStore(
                league_type=league_type, feature_name=f,
                family=family_of(f), version=version,
                formula_hash=formula_hash,
                status="active",
            ))
    db.session.commit()
    return version


if __name__ == "__main__":
    cols = ["home_elo", "away_elo", "elo_diff", "home_team_goals_avg",
            "home_team_form", "home_team_h2h_goals", "home_team_xg_avg"]
    s = summarize(cols)
    print(f"特征数: {s['total']} | 版本: {s['version']}")
    for fam, info in s["families"].items():
        print(f"  {fam}: {info['count']} 个 {info['features']}")
