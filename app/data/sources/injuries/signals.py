"""伤停数据 → info_fusion 信号映射(规则驱动,无需 LLM)

链路: InjuriesCollector.fetch_by_date → filter_by_team → injuries_to_signals → fuse()
协议与 models/prediction/info_fusion.py 的 SIGNAL_KEYS 对齐。

规则(V2_ARCHITECTURE.md §5.2 Phase A 严格实现——按位置加权):
  前锋缺阵:进攻 -6% | 中场: -3% | 后卫: -2% | 门将: -3%
  核心识别:缺阵时长 ≥ 60 天(受伤日期 → 比赛日,现有 injuries 数据可推)→ 该球员 ×1.5
  封顶 ±20%(MAX_IMPACT 保持,与 info_fusion MAX_ADJ=0.20 对齐,只降不升)
  rotation_risk 恒 0;confidence 固定 0.8

已知问题(留待后改):免费 api-football injuries 无 position 字段(文档 §5.1 假设
"injuries 含 position 已有"不成立)——位置无法解析时按中场基准 -3% 计,并在
summary 标注 position_missing;付费 lineups 数据到位后自动启用真实位置加权。
"""

from datetime import datetime, timezone

# 位置 → 权重(文档 §5.2 Phase A;position 缺失时按中场基准)
POSITION_WEIGHTS = {
    "forward": 0.06,
    "attacker": 0.06,
    "striker": 0.06,
    "cf": 0.06,
    "winger": 0.06,
    "lw": 0.06,
    "rw": 0.06,
    "midfielder": 0.03,
    "cm": 0.03,
    "dm": 0.03,
    "am": 0.03,
    "defender": 0.02,
    "cb": 0.02,
    "lb": 0.02,
    "rb": 0.02,
    "goalkeeper": 0.03,
    "gk": 0.03,
}
DEFAULT_POSITION_WEIGHT = 0.03  # 位置未知 → 中场基准

# 攻击端位置(影响进攻);防守端位置(影响防守);中场攻防皆影响
ATTACK_POSITIONS = {
    "forward",
    "attacker",
    "striker",
    "cf",
    "winger",
    "lw",
    "rw",
    "midfielder",
    "cm",
    "dm",
    "am",
}
DEFENSE_POSITIONS = {
    "defender",
    "cb",
    "lb",
    "rb",
    "goalkeeper",
    "gk",
    "midfielder",
    "cm",
    "dm",
    "am",
}

# 核心识别:缺阵 ≥ 60 天视为核心球员(长期缺阵),影响 ×1.5
CORE_DAYS = 60
CORE_MULT = 1.5

# 单侧最大影响(对齐 info_fusion MAX_ADJ)
MAX_IMPACT = 0.20


def _position_key(raw: str) -> str:
    return (raw or "").strip().lower().replace(" ", "")


def _position_weight(player: dict) -> float:
    pos = _position_key((player or {}).get("position") or "")
    if pos in POSITION_WEIGHTS:
        return POSITION_WEIGHTS[pos]
    return DEFAULT_POSITION_WEIGHT


def _is_attack_position(player: dict) -> bool:
    pos = _position_key((player or {}).get("position") or "")
    # 位置未知 → 按中场处理(攻防皆影响)
    return pos in ATTACK_POSITIONS or pos == ""


def _is_defense_position(player: dict) -> bool:
    pos = _position_key((player or {}).get("position") or "")
    return pos in DEFENSE_POSITIONS or pos == ""


def _core_mult(player: dict, match_day: str) -> float:
    """核心识别:缺阵时长(受伤日期 → 比赛日)≥ 60 天 → ×1.5。"""
    d = (player or {}).get("date") or ""
    if not d:
        return 1.0
    try:
        inj_dt = datetime.fromisoformat(str(d).replace("Z", ""))
        day_dt = datetime.fromisoformat(
            str(match_day or datetime.now(tz=timezone.utc).date())
        )
        days = (day_dt - inj_dt).days
        return CORE_MULT if days >= CORE_DAYS else 1.0
    except (ValueError, TypeError):
        return 1.0


def _side_impacts(records: list[dict], match_day: str) -> tuple[float, float, int, int]:
    """(进攻影响累计, 防守影响累计, 人数, 位置未知人数)

    每缺 1 人:前锋 -6% 进攻;中场 -3% 攻防;后卫 -2% 防守;门将 -3% 防守。
    """
    atk, dfn = 0.0, 0.0
    n = 0
    pos_missing = 0
    for r in records:
        player = r.get("player") or {}
        if not _position_key(player.get("position") or ""):
            pos_missing += 1
        w = _position_weight(player) * _core_mult(player, match_day)
        if _is_attack_position(player):
            atk += w
        if _is_defense_position(player):
            dfn += w
        n += 1
    return atk, dfn, n, pos_missing


def injuries_to_signals(
    home_injuries: list[dict], away_injuries: list[dict], match_day: str | None = None
) -> dict:
    """伤停记录 → info_fusion 兼容信号 dict(位置加权,§5.2 Phase A)。

    home_injuries/away_injuries: filter_by_team 后的伤停记录列表。
    无伤停数据的一侧输出 0(不猜测);双方都无数据时 sources 为空。
    """
    h_atk, h_dfn, h_n, h_pm = _side_impacts(home_injuries, match_day or "")
    a_atk, a_dfn, a_n, a_pm = _side_impacts(away_injuries, match_day or "")

    def _cap(v: float) -> float:
        return -min(MAX_IMPACT, v)

    signals = {
        # 伤停削弱本方进攻(前锋/中场)
        "home_attack_impact": round(_cap(h_atk), 4),
        "away_attack_impact": round(_cap(a_atk), 4),
        # 伤停削弱本方防守(后卫/门将/中场)
        "home_defense_impact": round(_cap(h_dfn), 4),
        "away_defense_impact": round(_cap(a_dfn), 4),
        # 轮换风险恒 0(缺阵已计入;rotation_risk 语义为密集赛程轮换不确定性)
        "home_rotation_risk": 0.0,
        "away_rotation_risk": 0.0,
        # 动机类信号伤停数据不提供,取 0
        "home_motivation": 0.0,
        "away_motivation": 0.0,
        "confidence": 0.8,
        "sources": ["api-football/injuries"] if (h_n or a_n) else [],
        "summary": {
            "home_injured": h_n,
            "home_position_missing": h_pm,
            "away_injured": a_n,
            "away_position_missing": a_pm,
        },
    }
    return signals


def signal_brief(signals: dict) -> str:
    """信号的人类可读摘要(供简报/调试)"""
    s = signals["summary"]
    lines = [
        (
            f"伤停信号: 主队缺 {s['home_injured']} 人"
            f"(位置未知 {s['home_position_missing']}),"
            f"客队缺 {s['away_injured']} 人(位置未知 {s['away_position_missing']})"
        ),
        (
            f"  λ 修正: 主队攻 {signals['home_attack_impact']:+.2f}/防 {signals['home_defense_impact']:+.2f},"
            f" 客队攻 {signals['away_attack_impact']:+.2f}/防 {signals['away_defense_impact']:+.2f}"
        ),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # 自测:位置加权方向
    rec = [
        {"player": {"name": "A", "position": "Forward"}},
        {"player": {"name": "B", "position": "Defender"}},
        {"player": {"name": "C", "position": ""}},
    ]
    sig = injuries_to_signals(rec, [], "2026-08-15")
    print(signal_brief(sig))
    assert sig["home_attack_impact"] == -0.09, sig[
        "home_attack_impact"
    ]  # 前锋6%+未知中场3%
    assert sig["home_defense_impact"] == -0.05, sig[
        "home_defense_impact"
    ]  # 后卫2%+未知中场3%
    print("✅ 位置加权自测通过")
