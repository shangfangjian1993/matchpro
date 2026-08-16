"""评价指标体系(V2_ARCHITECTURE.md §6 全量)。

| 指标 | 用途 |
|---|---|
| Log-Loss / Brier / RPS | 概率质量(主指标) |
| Calibration Error(ECE) | 校准度 |
| Goal MAE / RMSE | 进球精度 |
| 胜负准确率 | 辅助参考 |
| Top-3/Top-5 覆盖率 | 参考(不作主指标) |

所有函数输入统一:
  probs: [home_win, draw, away_win](预测概率,和=1)
  actual: 0=主胜 1=平 2=客胜(或实际进球数)
"""
from __future__ import annotations

import math

import numpy as np


def log_loss(probs, actual: int, eps: float = 1e-12) -> float:
    """Log-Loss(三分类)。"""
    p = max(eps, min(1.0 - eps, probs[actual]))
    return -math.log(p)


def brier_score(probs, actual: int) -> float:
    """多分类 Brier Score:mean_k (p_k - y_k)²,y 为 one-hot。"""
    y = np.zeros(3)
    y[actual] = 1.0
    return float(np.sum((np.asarray(probs) - y) ** 2))


def rps(probs, actual: int) -> float:
    """Ranked Probability Score:概率 CDF 与真实 CDF 的平方差和。"""
    p = np.asarray(probs)
    y = np.zeros(3)
    y[actual] = 1.0
    return float(np.sum((np.cumsum(p) - np.cumsum(y)) ** 2))


def ece(probs_list: list, actuals: list, n_bins: int = 10) -> float:
    """Expected Calibration Error:按预测最大概率分箱,|准确率 - 平均概率| 加权平均。"""
    conf = [max(p) for p in probs_list]
    acc = [1.0 if np.argmax(p) == a else 0.0 for p, a in zip(probs_list, actuals)]
    conf, acc = np.asarray(conf), np.asarray(acc)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1])
        if mask.sum() == 0:
            continue
        total += mask.sum() * abs(acc[mask].mean() - conf[mask].mean())
    return float(total / max(1, len(conf)))


def goal_mae_rmse(pred_goals, actual_goals) -> tuple[float, float]:
    """进球精度:MAE / RMSE(可传每侧进球或总进球)。"""
    p = np.asarray(pred_goals, dtype=float)
    a = np.asarray(actual_goals, dtype=float)
    err = p - a
    return float(np.abs(err).mean()), float(np.sqrt((err ** 2).mean()))


def accuracy(probs_list: list, actuals: list) -> float:
    """胜负准确率(辅助参考)。"""
    if not probs_list:
        return 0.0
    hits = sum(1 for p, a in zip(probs_list, actuals) if np.argmax(p) == a)
    return hits / len(probs_list)


def topk_coverage(score_matrix, actual_goals: tuple[int, int], k: int) -> bool:
    """实际比分是否在概率 Top-k 内。score_matrix: (home_goals × away_goals) 概率矩阵。"""
    m = np.asarray(score_matrix, dtype=float)
    flat = [(i, j, m[i, j]) for i in range(m.shape[0]) for j in range(m.shape[1])]
    flat.sort(key=lambda t: -t[2])
    return (actual_goals[0], actual_goals[1]) in {(i, j) for i, j, _ in flat[:k]}


if __name__ == "__main__":
    # 自测:完美预测 vs 随机预测
    good = [([0.9, 0.05, 0.05], 0), ([0.05, 0.9, 0.05], 1), ([0.05, 0.05, 0.9], 2)]
    bad = [([0.4, 0.3, 0.3], 0), ([0.3, 0.4, 0.3], 1), ([0.3, 0.3, 0.4], 2)]
    ll_g = sum(log_loss(p, a) for p, a in good) / 3
    ll_b = sum(log_loss(p, a) for p, a in bad) / 3
    assert ll_g < ll_b, "Log-Loss 应更低"
    assert brier_score([1.0, 0.0, 0.0], 0) == 0.0
    assert abs(rps([1.0, 0.0, 0.0], 0)) < 1e-12
    assert ece([p for p, _ in good], [a for _, a in good]) < 0.3
    assert topk_coverage(np.array([[0.5, 0.1], [0.1, 0.3]]), (0, 0), 1) is True
    print(f"✅ 评估指标自测通过 (Log-Loss 好/差: {ll_g:.3f} / {ll_b:.3f})")
