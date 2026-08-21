"""Regime Detector + Score Matrix 级调整(审查九 P0-3/P1-12)。

P0-3:Prior Blend 只改 1X2 而 score matrix 不变 → 两套输出统计不一致。
修复:先算目标 1X2(α·P_model + (1-α)·近期频率),再用 IPF(迭代比例拟合)
调整 score matrix 使其边缘 = 目标 1X2,所有输出(Top5/O/U/BTTS/xG/1X2)
从同一最终矩阵导出 —— 彻底消除"1X2 与比分分布不一致"。

P1-12:Regime Detector —— 检测平局率/进球率/主场优势漂移,输出 regime
标签;α 动态化:稳定赛季 α≈0.85,剧烈 shift α≈0.55(审查十三)。
"""

from __future__ import annotations

import numpy as np


# ── IPF:按 1X2 类别边缘调整 score matrix ──────────────────────────────────
def ipf_to_target(
    matrix: np.ndarray,
    target: tuple[float, float, float],
    max_iter: int = 200,
    tol: float = 1e-10,
) -> np.ndarray:
    """迭代调整矩阵使 1X2 边缘 = target,保持类内(胜/平/负格点)相对结构。

    matrix: 10x10 概率矩阵;target = (home_win, draw, away_win)。
    每轮对下三角/对角/上三角格点施加同一乘数(类内比例不变),归一化后
    重复,直到边缘收敛 —— 标准 IPF 的类别边缘变体。
    """
    m = np.asarray(matrix, dtype=float).copy()
    s = m.sum()
    if s <= 0:
        return m
    m = m / s
    n = len(m)
    tril = np.tril(np.ones((n, n), dtype=bool), -1)
    diag = np.eye(n, dtype=bool)
    triu = np.triu(np.ones((n, n), dtype=bool), 1)
    th, td, ta = (float(x) for x in target)
    for _ in range(max_iter):
        ph = m[tril].sum()
        pd_ = np.trace(m)
        pa = m[triu].sum()
        if abs(ph - th) < tol and abs(pd_ - td) < tol and abs(pa - ta) < tol:
            break
        mult = np.ones_like(m)
        mult[tril] = th / max(ph, 1e-12)
        mult[diag] = td / max(pd_, 1e-12)
        mult[triu] = ta / max(pa, 1e-12)
        m = m * mult
        m = m / m.sum()
    return m


# ── Regime Detector(审查 P1-12)────────────────────────────────────────────
REGIMES = (
    "NORMAL",
    "HIGH_DRAW",
    "LOW_SCORING",
    "HIGH_SCORING",
    "HOME_ADV_SHIFT",
    "STRENGTH_COMPRESSED",
)


def detect(
    league_id,
    cutoff_dt,
    window: int = 100,
    baseline_window: int = 1000,
    min_samples: int = 50,
) -> dict:
    """检测截止该场的 regime 漂移(审查九 P1-12 + 深化)。

    维度(审查二十七):
      - Draw Rate Shift / Goal Rate Shift / Home Advantage Shift(基础)
      - Team Strength Dispersion(球队强度分散度压缩 → STRENGTH_COMPRESSED)
      - Distribution Shift(低分率 0-1 球占比,无 xG 数据下的分布代理)
      - Calibration Drift(有 production 快照时,近期 ECE vs 历史)

    返回 {regime, shift_score, 各维度值, dispersion_*, calib_*}。
    shift_score ∈ [0, 1]:0=无漂移,1=剧烈漂移。
    """
    from app.api.db import Match

    rows = (
        Match.query.filter(
            Match.league_id == league_id,
            Match.match_status == "finished",
            Match.match_date < cutoff_dt,
        )
        .order_by(Match.match_date.desc())
        .limit(window + baseline_window)
        .all()
    )
    if len(rows) < min_samples:
        return {"regime": "NORMAL", "shift_score": 0.0, "sample": len(rows)}

    def _rates(ms):
        n = len(ms)
        if n == 0:
            return 0.0, 0.0, 0.0, 0.0
        h = sum(1 for m in ms if (m.home_goals or 0) > (m.away_goals or 0))
        d = sum(1 for m in ms if (m.home_goals or 0) == (m.away_goals or 0))
        goals = sum((m.home_goals or 0) + (m.away_goals or 0) for m in ms) / n
        low = sum(1 for m in ms if (m.home_goals or 0) + (m.away_goals or 0) <= 1) / n
        return h / n, d / n, goals, low

    def _dispersion(ms):
        """球队 latent strength 分散度(审查十 P1-1 + A70A601 P1-5)。

        以**净胜球均值**作为 latent strength 近似(攻击-防守综合,
        含进失球两端信息;系统已有 ELO/xG 时可在特征层替换)。
        strength_i = mean(该队近 5 场进球) - mean(该队近 5 场失球)
        dispersion = std(全队 strength) —— 这才是"强弱分化程度"。
        A70A601 P1-5:实现与注释统一 —— 取该队**最近 5 场**(ms 已按日期
        倒序,per_team 追加顺序即时间顺序,尾部为最近 5 次出场)。
        """
        import collections

        per_team = collections.defaultdict(list)
        for m in ms:
            gh, ga = (m.home_goals or 0), (m.away_goals or 0)
            per_team[m.home_team].append((gh, ga))
            per_team[m.away_team].append((ga, gh))
        strengths = []
        for v in per_team.values():
            v5 = v[-5:]  # 最近 5 场(recent-5,与注释一致)
            if len(v5) >= 3:
                gf = sum(x[0] for x in v5) / len(v5)
                ga2 = sum(x[1] for x in v5) / len(v5)
                strengths.append(gf - ga2)
        if len(strengths) < 6:
            return None
        return float(np.std(strengths))

    recent = rows[:window]
    baseline = rows[window : window + baseline_window]
    if len(baseline) < min_samples:
        baseline = rows[min(window, len(rows) - min_samples) :]
    rh, rd, rg, rl = _rates(recent)
    bh, bd, bg, bl = _rates(baseline)
    disp_r = _dispersion(recent)
    disp_b = _dispersion(baseline)
    disp_ratio = (
        (disp_r / disp_b) if (disp_r is not None and disp_b and disp_b > 0) else None
    )
    # 审查十 P1-2:漂移评分 = 加权复合(非 max),单维贡献 cap 0.35
    # —— 防止单一维度统计噪声把整个 regime 判成剧烈漂移
    _c_draw = abs(rd - bd) / 0.10  # 平局率 ±10pp 满量程
    _c_goal = abs(rg - bg) / 0.60  # 场均进球 ±0.6 满量程
    _c_home = abs(rh - bh) / 0.12  # 主胜率 ±12pp 满量程
    _c_low = abs(rl - bl) / 0.12  # 低分率 ±12pp 满量程
    _c_str = (max(0.0, 1.0 - (disp_ratio or 1.0))) / 0.25  # 强度压缩 ±25% 满量程
    _comps = [_c_draw, _c_goal, _c_home, _c_low, _c_str]
    _capped = [min(x, 0.35) for x in _comps]
    # Calibration Drift(尽力而为:production 快照的近期 ECE)
    calib = _calibration_drift(league_id, cutoff_dt)
    # 仅作诊断字段)。drift=ECE 变化 ±5pp 满量程,converged 时与原 5 维同权
    # (原权重×0.85 + calib×0.15);drift 缺失(无快照)时退回原 5 维权重。
    _w5 = [0.25, 0.25, 0.15, 0.15, 0.20]
    if calib and calib.get("drift", 0.0) > 0:
        _c_cal = min(abs(calib["drift"]) / 0.05, 0.35)
        shift = float(
            np.clip(
                sum(w * cp for w, cp in zip(_w5, _capped)) * 0.85 + 0.15 * _c_cal,
                0.0,
                1.0,
            )
        )
    else:
        shift = float(np.clip(sum(w * cp for w, cp in zip(_w5, _capped)), 0.0, 1.0))

    if shift < 0.15 and not (disp_ratio is not None and disp_ratio < 0.8):
        regime = "NORMAL"
    elif rd - bd > 0.03:
        regime = "HIGH_DRAW"
    elif rg - bg < -0.25:
        regime = "LOW_SCORING"
    elif rg - bg > 0.25:
        regime = "HIGH_SCORING"
    elif disp_ratio is not None and disp_ratio < 0.8:
        regime = "STRENGTH_COMPRESSED"
    elif abs(rh - bh) > 0.05:
        regime = "HOME_ADV_SHIFT"
    else:
        regime = "NORMAL"
    out = {
        "regime": regime,
        "shift_score": shift,
        "draw_rate": round(rd, 4),
        "goal_rate": round(rg, 3),
        "home_rate": round(rh, 4),
        "low_score_rate": round(rl, 4),
        "base_draw": round(bd, 4),
        "base_goal": round(bg, 3),
        "base_home": round(bh, 4),
        "base_low_score": round(bl, 4),
        "dispersion_recent": (round(disp_r, 4) if disp_r is not None else None),
        "dispersion_base": (round(disp_b, 4) if disp_b is not None else None),
        "dispersion_ratio": (round(disp_ratio, 4) if disp_ratio is not None else None),
        "sample": len(recent),
    }
    if calib is not None:
        out["calibration_drift"] = calib
    return out


def _calibration_drift(league_id, cutoff_dt, window: int = 100) -> dict | None:
    """校准漂移(尽力而为):production 快照的近期 ECE vs 更早 ECE。

    快照少(production 模式)时返回 None —— 数据不足不误报。
    """
    try:
        from app.api.db import PredictionSnapshot

        snaps = (
            PredictionSnapshot.query.filter(
                PredictionSnapshot.league_id == league_id,
                PredictionSnapshot.kickoff < cutoff_dt,
            )
            .order_by(PredictionSnapshot.kickoff.desc())
            .limit(window * 2)
            .all()
        )
        if len(snaps) < 60:
            return None
        import json as _json

        from app.replay.metrics import ece as _ece

        def _ece_of(rows):
            pvecs, acts = [], []
            for s in rows:
                p = _json.loads(s.probabilities_json or "{}")
                pvecs.append(
                    [p.get("home_win", 0), p.get("draw", 0), p.get("away_win", 0)]
                )
                gh, ga = s.actual_home_goals or 0, s.actual_away_goals or 0
                acts.append(0 if gh > ga else (1 if gh == ga else 2))
            return _ece(pvecs, acts) if pvecs else None

        recent_ece = _ece_of(snaps[: min(window, len(snaps))])
        base_ece = _ece_of(snaps[min(window, len(snaps)) :])
        if recent_ece is None or base_ece is None:
            return None
        return {
            "recent_ece": round(float(recent_ece), 4),
            "base_ece": round(float(base_ece), 4),
            "drift": round(float(recent_ece - base_ece), 4),
            "n": len(snaps),
        }
    except Exception:
        return None


def dynamic_alpha(
    shift_score: float,
    alpha_stable: float = 0.85,
    alpha_shift: float = 0.55,
    regime: str = "NORMAL",
) -> float:
    """审查十三 + 深化:稳定赛季 ≈0.85,剧烈 shift ≈0.55;
    STRENGTH_COMPRESSED(强弱分化减弱 → 爆冷多)额外向近期频率倾斜。"""
    alpha = alpha_stable - (alpha_stable - alpha_shift) * shift_score
    if regime == "STRENGTH_COMPRESSED":
        alpha = min(alpha, alpha_shift + 0.05)
    return float(np.clip(alpha, alpha_shift, alpha_stable))
