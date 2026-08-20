"""阵容强度向量(审查九 P1-11)。

数据受限(free 数据源无首发/球员级出场分钟),当前实现为"伤停驱动的
阵容强度结构":从 injuries 信号按位置聚合缺失强度,输出 XI 维度向量;
expected_minutes 用核心球员缺阵时长(≥60d ×1.5)估计。

未来升级路径:player → expected_minutes → position value →
starting XI strength → attack/defense latent state。
"""

from __future__ import annotations

# 位置缺失权重(审查 §5.2):前锋 -6% / 中场 -3% / 后卫 -2% / 门将 -3%
POSITION_WEIGHT = {"前锋": 0.06, "中场": 0.03, "后卫": 0.02, "门将": 0.03}


def build_strength_vector(sig: dict) -> dict:
    """从伤停信号构建主/客阵容强度向量。

    sig: injuries_to_signals 输出 {home: [...], away: [...], sources: [...]}
    返回 {"home": {...}, "away": {...}, "available": bool}。
    每侧:{"position_loss": {...}, "total_loss": float, "expected_minutes_adj": float}
    """
    out = {"home": None, "away": None, "available": bool(sig.get("sources"))}
    for side in ("home", "away"):
        recs = sig.get(side) or []
        pos_loss = {}
        total = 0.0
        long_core = 0  # 长期缺阵核心(≥60 天)数量
        for r in recs:
            pos = r.get("position") or "中场"
            w = POSITION_WEIGHT.get(pos, 0.03)
            days = r.get("days_out") or 0
            mult = 1.5 if days >= 60 else 1.0
            if r.get("core"):
                long_core += 1 if days >= 60 else 0
            loss = w * mult
            pos_loss[pos] = round(pos_loss.get(pos, 0.0) + loss, 4)
            total += loss
        out[side] = {
            "position_loss": pos_loss,
            "expected_minutes_adj": round(1.0 - min(total, 0.20), 4),
            "long_core_out": long_core,
        }
    return out


def strength_to_lambda_adjust(vector: dict, side: str) -> float:
    """阵容强度 → λ 乘数(与 signals_to_adjust 同语义,统一入口)。"""
    v = (vector or {}).get(side)
    if not v:
        return 1.0
    return round(1.0 - (v.get("total_loss") or 0.0), 4)
