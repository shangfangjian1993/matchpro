"""模型训练工具:时间衰减样本权重(评审 P1 实现方案)。

- compute_time_decay_weights:指数衰减 w = exp(-λ·Δt),λ = ln2/half_life
  含 min_weight 下限保护(防止极老比赛被完全忽略)+ normalize(均值=1)
- compute_piecewise_decay_weights:分段衰减变体(近1年全权重,1-3年线性,3年+固定)
- compute_balanced_time_weights:联赛反频率 + 时间衰减(防单联赛主导)

防泄漏约定:reference_date 必须用训练集内最大日期(绝不能用"当前真实时间")。
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd


def compute_time_decay_weights(
    dates: pd.Series | np.ndarray | list,
    *,
    half_life_days: float = 365.0,
    min_weight: float = 0.05,
    reference_date: datetime | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """指数时间衰减权重:w = exp(-λ·Δt),λ = ln(2)/half_life_days。

    half_life_days=365 → 1 年前比赛权重约为最近比赛的一半。
    reference_date 默认取传入日期的最大值(训练集内,严格防泄漏)。
    """
    dts = pd.to_datetime(pd.Series(dates), utc=True)
    if reference_date is None:
        reference_date = dts.max()
    if hasattr(reference_date, "tzinfo") and getattr(reference_date, "tzinfo", None) is None:
        reference_date = reference_date.replace(tzinfo=timezone.utc)

    delta_days = (pd.Timestamp(reference_date) - dts).dt.total_seconds() / 86400.0
    delta_days = np.clip(delta_days.to_numpy(), 0.0, None)  # 未来比赛按 0 天(防泄漏时通常已过滤)

    decay_rate = np.log(2) / max(half_life_days, 1e-6)
    weights = np.exp(-decay_rate * delta_days)
    weights = np.maximum(weights, min_weight)  # 下限保护

    if normalize:
        weights = weights / weights.mean()
    return weights.astype(np.float64)


def compute_piecewise_decay_weights(dates, reference_date: datetime | None = None) -> np.ndarray:
    """分段衰减:近 1 年全权重,1~3 年线性 1.0→0.3,3 年以上固定 0.3。"""
    dts = pd.to_datetime(pd.Series(dates), utc=True)
    if reference_date is None:
        reference_date = dts.max()
    delta = (pd.Timestamp(reference_date) - dts).dt.days.clip(lower=0).to_numpy(dtype=float)

    w = np.ones_like(delta, dtype=float)
    mask = (delta > 365) & (delta <= 1095)
    w[mask] = 1.0 - 0.7 * (delta[mask] - 365) / (1095 - 365)
    w[delta > 1095] = 0.3
    return (w / w.mean()).astype(np.float64)


def compute_balanced_time_weights(df: pd.DataFrame, half_life_days: float = 365.0) -> np.ndarray:
    """联赛反频率 + 时间衰减(防止某联赛样本过多主导)。"""
    time_w = compute_time_decay_weights(df["date"], half_life_days=half_life_days,
                                        normalize=False)
    league_counts = df["league"].value_counts()
    league_w = df["league"].map(lambda x: 1.0 / league_counts.get(x, 1)).to_numpy()
    league_w = league_w / league_w.mean()
    combined = time_w * league_w
    return (combined / combined.mean()).astype(np.float64)


if __name__ == "__main__":
    # 自测
    import pandas as pd
    dates = pd.date_range("2020-01-01", "2023-01-01", periods=5)
    w = compute_time_decay_weights(dates, half_life_days=365.0)
    assert w[0] < w[-1], "更老的比赛权重应更低"
    assert abs(w.mean() - 1.0) < 1e-9, "normalize 均值=1"
    assert w.min() >= 0.05 - 1e-12, "下限保护"
    print("✅ compute_time_decay_weights:", [round(x, 3) for x in w])
    w2 = compute_piecewise_decay_weights(dates)
    assert abs(w2.mean() - 1.0) < 1e-9
    print("✅ compute_piecewise_decay_weights OK")
    df = pd.DataFrame({"date": dates, "league": ["A", "A", "B", "B", "B"]})
    w3 = compute_balanced_time_weights(df)
    assert abs(w3.mean() - 1.0) < 1e-9
    print("✅ compute_balanced_time_weights OK")
"""赛前 ELO 提取(审查 P0-1):当前对阵两队的赛前 rating,而非历史最后一场的任意值。"""


def prematch_elo(hist_df, home_team: str, away_team: str, dim: str = "attack") -> tuple:
    """返回 (home_elo, away_elo, diff)。

    dim: "overall"|"attack"|"defense";取两队各自最后一次出场(主/客任意角色)
    后的赛前值(hist_df 已含时间重放 ELO 列)。
    """
    col_h = f"home_{dim}_elo" if dim != "overall" else "home_elo"
    col_a = f"away_{dim}_elo" if dim != "overall" else "away_elo"
    if col_h not in hist_df.columns:
        return None, None, None

    def _team_last(team):
        h = hist_df.index[hist_df["home_team"] == team]
        a = hist_df.index[hist_df["away_team"] == team]
        candidates = []
        if len(h):
            candidates.append((h[-1], hist_df.loc[h[-1], col_h]))
        if len(a):
            candidates.append((a[-1], hist_df.loc[a[-1], col_a]))
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0])  # 行序=时间序(已按 date 排序)
        return candidates[-1][1]

    home_elo = _team_last(home_team)
    away_elo = _team_last(away_team)
    if home_elo is None or away_elo is None:
        return None, None, None
    return float(home_elo), float(away_elo), float(home_elo - away_elo)
