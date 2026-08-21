"""评价指标体系(V2_ARCHITECTURE.md 6 全量)。

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
 return float(np.abs(err).mean()), float(np.sqrt((err**2).mean()))


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


# ── 


def calibration_slope_intercept(
 probs_list: list, actuals: list, n_bins: int = 10
) -> dict:
 """校准回归:conf = slope·acc + intercept(

 将预测置信度分桶,对 (桶平均置信度, 桶实际频率) 做线性拟合;
 slope≈1 & intercept≈0 表示理想校准。slope<1 → 过度自信。
 """
 import numpy as np

 if not probs_list:
 return {"slope": None, "intercept": None, "n_bins": 0}
 conf = np.array([max(p) for p in probs_list])
 acc = (np.array([np.argmax(p) for p in probs_list]) == np.array(actuals)).astype(
 float
 )
 edges = np.linspace(0.0, 1.0, n_bins + 1)
 xs, ys = [], []
 for i in range(n_bins):
 mask = (conf > edges[i]) & (conf <= edges[i + 1])
 if mask.sum() >= 5:
 xs.append(float(conf[mask].mean()))
 ys.append(float(acc[mask].mean()))
 if len(xs) < 2:
 return {"slope": None, "intercept": None, "n_bins": len(xs)}
 slope, intercept = np.polyfit(xs, ys, 1)
 return {
 "slope": round(float(slope), 4),
 "intercept": round(float(intercept), 4),
 "n_bins": len(xs),
 }


def sharpness(probs_list: list) -> float:
 """锐度:预测概率分布的集中度(
 import numpy as np

 if not probs_list:
 return 0.0
 p = np.array(probs_list)
 return round(float(np.mean(p.max(axis=1))), 4)


def brier_decomposition(probs_list: list, actuals: list, n_bins: int = 10) -> dict:
 """Brier 分解(

 - Reliability:校准误差(越小越好)
 - Resolution:区分度(越大越好)
 - Uncertainty:内在不确定性(常数)
 """
 import numpy as np

 if not probs_list:
 return {
 "reliability": None,
 "resolution": None,
 "uncertainty": None,
 "brier": None,
 }
 probs = np.array(probs_list)
 k = probs.shape[1]
 actuals = np.array(actuals)
 # 三分类 one-hot
 obs = np.zeros((len(actuals), k))
 obs[np.arange(len(actuals)), actuals] = 1.0
 brier = float(np.mean(np.sum((probs - obs) ** 2, axis=1)))
 base = obs.mean(axis=0) # 基率
 uncertainty = float(np.sum(base * (1 - base)))
 # Reliability + Resolution(按类别平均)
 conf = probs.max(axis=1)
 edges = np.linspace(0.0, 1.0, n_bins + 1)
 rel = res = 0.0
 for i in range(n_bins):
 mask = (conf > edges[i]) & (conf <= edges[i + 1])
 if mask.sum() == 0:
 continue
 n = mask.sum()
 p_bar = probs[mask].mean(axis=0)
 o_bar = obs[mask].mean(axis=0)
 rel += n * float(np.sum((o_bar - p_bar) ** 2))
 res += n * float(np.sum((o_bar - base) ** 2))
 n = len(actuals)
 return {
 "reliability": round(float(rel / n), 5),
 "resolution": round(float(res / n), 5),
 "uncertainty": round(float(uncertainty), 5),
 "brier": round(float(brier), 5),
 }


def logloss_by_bucket(
 probs_list: list, actuals: list, edges=(0.5, 0.6, 0.7, 0.8, 0.9)
) -> dict:
 """LogLoss 按概率桶(
 if not probs_list:
 return {}
 out = {}
 buckets = [0.0] + list(edges) + [1.01]
 for lo, hi in __import__("itertools").pairwise(buckets):
 idx = [i for i, p in enumerate(probs_list) if lo <= max(p) < hi]
 if len(idx) >= 10:
 ll = sum(log_loss(probs_list[i], actuals[i]) for i in idx) / len(idx)
 acc = sum(1 for i in idx if np.argmax(probs_list[i]) == actuals[i]) / len(
 idx
 )
 out[f"{lo:.2f}-{hi:.2f}"] = {
 "n": len(idx),
 "log_loss": round(float(ll), 4),
 "accuracy": round(float(acc), 4),
 }
 return out


def score_log_likelihood(
 score_matrix, actual_goals: tuple[int, int], eps: float = 1e-12
) -> float:
 """比分分布对数似然(

 score_matrix: 10x10 概率矩阵;实际比分越界(≥10)时用边缘尾部近似。
 """
 import numpy as np

 m = np.asarray(score_matrix, dtype=float)
 if m.size == 0:
 return 0.0
 h, a = int(actual_goals[0]), int(actual_goals[1])
 if h < m.shape[0] and a < m.shape[1]:
 p = float(m[h, a])
 else:
 # 越界:用行/列边缘尾部(第 10 行/列之和)近似
 ph = float(m[min(h, m.shape[0] - 1), :].sum())
 pa = float(m[:, min(a, m.shape[1] - 1)].sum())
 p = ph * pa
 return float(-np.log(max(p, eps)))


# 便捷聚合:一次算全(供 summarize/backfill 使用)
def extended_metrics(
 probs_list: list,
 actuals: list,
 score_matrices: list | None = None,
 actual_scores: list | None = None,
) -> dict:
 """全套扩展指标(
 out = {
 "brier_decomp": brier_decomposition(probs_list, actuals),
 "calibration_slope": calibration_slope_intercept(probs_list, actuals)["slope"],
 "calibration_intercept": calibration_slope_intercept(probs_list, actuals)[
 "intercept"
 ],
 "sharpness": sharpness(probs_list),
 "logloss_by_bucket": logloss_by_bucket(probs_list, actuals),
 }
 if score_matrices and actual_scores and len(score_matrices) == len(actual_scores):
 sll = [
 score_log_likelihood(m, s) for m, s in zip(score_matrices, actual_scores)
 ]
 out["score_log_likelihood_mean"] = round(float(np.mean(sll)), 5)
 out["score_log_likelihood_n"] = len(sll)
 return out
