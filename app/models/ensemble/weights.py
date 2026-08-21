"""权重配置与学习(Typed Schema + Split Optimizers)。

Layer-1: Goal λ Ensemble (HGBR/ELO/Bayes)
Layer-2: Score Distribution (Poisson/DC/NB,基于 fused λ)
Layer-3: Outcome (Shape + GBM)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
import os

import numpy as np


@dataclass(frozen=True)
class GoalLambdaWeights:
 """Layer-1: Goal λ 权重。"""
 hgbr: float
 elo: float
 bayes: float
 
 def to_dict(self) -> dict:
 return asdict(self)


@dataclass(frozen=True)
class ScoreDistributionWeights:
 """Layer-2: Score Distribution 权重。"""
 poisson: float
 dc: float
 nb: float
 
 def to_dict(self) -> dict:
 return asdict(self)


@dataclass(frozen=True)
class OutcomeWeights:
 """Layer-3: Outcome 权重。"""
 shape: float
 gbm: float
 
 def to_dict(self) -> dict:
 return asdict(self)


@dataclass(frozen=True)
class EnsembleWeights:
 """完整 Ensemble 权重(Typed Schema)。"""
 goal_lambda: GoalLambdaWeights
 score_distribution: ScoreDistributionWeights
 outcome: OutcomeWeights
 
 def to_dict(self) -> dict:
 return {
 "goal_lambda": self.goal_lambda.to_dict(),
 "score_distribution": self.score_distribution.to_dict(),
 "outcome": self.outcome.to_dict(),
 }
 
 @classmethod
 def from_dict(cls, data: dict) -> EnsembleWeights:
 gl = data.get("goal_lambda", {})
 sd = data.get("score_distribution", {})
 oc = data.get("outcome", {})
 return cls(
 goal_lambda=GoalLambdaWeights(
 hgbr=gl.get("hgbr", 0.5),
 elo=gl.get("elo", 0.3),
 bayes=gl.get("bayes", 0.2),
 ),
 score_distribution=ScoreDistributionWeights(
 poisson=sd.get("poisson", 0.5),
 dc=sd.get("dc", 0.3),
 nb=sd.get("nb", 0.2),
 ),
 outcome=OutcomeWeights(
 shape=oc.get("shape", 1.0),
 gbm=oc.get("gbm", 0.0),
 ),
 )
 
 def to_flat(self) -> dict:
 """转为 flat 格式(兼容 engine.py)。"""
 return {
 "hgbr": self.goal_lambda.hgbr,
 "elo": self.goal_lambda.elo,
 "bayes": self.goal_lambda.bayes,
 "poisson": self.score_distribution.poisson,
 "dc": self.score_distribution.dc,
 "nb": self.score_distribution.nb,
 "shape_weight": self.outcome.shape,
 "gbm_weight": self.outcome.gbm,
 }


# 默认权重
DEFAULT_WEIGHTS = EnsembleWeights(
 goal_lambda=GoalLambdaWeights(hgbr=0.5, elo=0.3, bayes=0.2),
 score_distribution=ScoreDistributionWeights(poisson=0.5, dc=0.3, nb=0.2),
 outcome=OutcomeWeights(shape=1.0, gbm=0.0),
)


def to_layered(data: dict) -> dict:
 """转换为分层格式(幂等)。"""
 if isinstance(data, EnsembleWeights):
 return data.to_dict()
 if "goal_lambda" in data and "score_distribution" in data and "outcome" in data:
 return _normalize_layered(data)
 return _flat_to_layered(data)


def _flat_to_layered(flat: dict) -> dict:
 """flat → layered(: 全精度,无 round/renormalize)。"""
 goal = {
 "hgbr": float(flat.get("hgbr", 0.0)),
 "elo": float(flat.get("elo", 0.0)),
 "bayes": float(flat.get("bayes", 0.0)),
 }
 gsum = sum(goal.values()) or 1.0
 
 sd = {
 "poisson": float(flat.get("poisson", gsum)),
 "dc": float(flat.get("dc", 0.0)),
 "nb": float(flat.get("nb", 0.0)),
 }
 ssum = sum(sd.values()) or 1.0
 
 outcome = {
 "shape": float(flat.get("shape_weight", 1.0)),
 "gbm": float(flat.get("gbm_weight", flat.get("gbm", 0.0))),
 }
 
 return {
 "goal_lambda": {k: v / gsum for k, v in goal.items()},
 "score_distribution": {k: v / ssum for k, v in sd.items()},
 "outcome": outcome,
 }


def _normalize_layered(layered: dict) -> dict:
 """归一化 layered 格式。"""
 goal = layered.get("goal_lambda", {})
 gsum = sum(float(goal.get(k, 0.0)) for k in ["hgbr", "elo", "bayes"]) or 1.0
 sd = layered.get("score_distribution", {})
 ssum = sum(float(sd.get(k, 0.0)) for k in ["poisson", "dc", "nb"]) or 1.0
 outcome = layered.get("outcome", {})
 return {
 "goal_lambda": {k: float(goal.get(k, 0.0)) / gsum for k in ["hgbr", "elo", "bayes"]},
 "score_distribution": {k: float(sd.get(k, 0.0)) / ssum for k in ["poisson", "dc", "nb"]},
 "outcome": {"shape": float(outcome.get("shape", 1.0)), "gbm": float(outcome.get("gbm", 0.0))},
 }


def from_layered(layered: dict) -> dict:
 """layered → flat(兼容 engine.py)。"""
 goal = layered.get("goal_lambda", {})
 sd = layered.get("score_distribution", {})
 outcome = layered.get("outcome", {})
 return {
 "hgbr": float(goal.get("hgbr", 0.0)),
 "elo": float(goal.get("elo", 0.0)),
 "bayes": float(goal.get("bayes", 0.0)),
 "poisson": float(sd.get("poisson", 0.0)),
 "dc": float(sd.get("dc", 0.0)),
 "nb": float(sd.get("nb", 0.0)),
 "shape_weight": float(outcome.get("shape", 1.0)),
 "gbm_weight": float(outcome.get("gbm", 0.0)),
 }


def _is_layered(data) -> bool:
 return isinstance(data, dict) and "goal_lambda" in data and "score_distribution" in data


_WEIGHTS_PATH = None


def set_weights_path(path: str | None):
 global _WEIGHTS_PATH
 _WEIGHTS_PATH = path


class OptimizationError(Exception):
 """SLSQP 优化失败。"""
 pass


# ── : 拆分 Layer-1 / Layer-2 optimizer ──

def optimize_goal_lambda_weights(
 samples: list[dict],
 shrinkage: float = 0.15,
 prior: dict | None = None,
 max_weight: float = 0.7,
) -> dict:
 """Layer-1: 使用 Poisson Goal NLL 优化 λ 权重。
 
 输入: samples 中包含 hgbr_lam_h/a, elo_lam_h/a, bayes_lam_h/a
 输出: {hgbr, elo, bayes, poisson, dc, nb} (后三个为0)
 """
 candidates = ["hgbr", "elo", "bayes"]
 present = [n for n in candidates if any(f"{n}_lam_h" in s for s in samples)]
 
 if not present:
 return {name: 1.0 if name == "hgbr" else 0.0 for name in ["hgbr", "elo", "bayes", "poisson", "dc", "nb"]}
 
 n = max(1, len(samples))
 
 if prior is None:
 _prior = _compute_baseline_prior_poisson(samples, present, n)
 else:
 _prior = np.array([prior.get(name, 0.0) for name in present], dtype=float)
 _prior = _prior / _prior.sum() if _prior.sum() > 0 else np.ones(len(present)) / len(present)
 
 if len(present) < 2:
 return {name: (1.0 if name == present[0] else 0.0) for name in ["hgbr", "elo", "bayes", "poisson", "dc", "nb"]}
 
 def _nll(w):
 ll = 0.0
 for s in samples:
 lam_h = sum(w[i] * s.get(f"{name}_lam_h", 1.5) for i, name in enumerate(present))
 lam_a = sum(w[i] * s.get(f"{name}_lam_a", 1.2) for i, name in enumerate(present))
 from math import lgamma, log
 gh = int(s.get("home_goals", 0))
 ga = int(s.get("away_goals", 0))
 ll -= (gh * log(max(lam_h, 1e-12)) - lam_h - lgamma(gh + 1))
 ll -= (ga * log(max(lam_a, 1e-12)) - lam_a - lgamma(ga + 1))
 ll = ll / n
 if shrinkage > 0 and len(present) > 1:
 ll += shrinkage * float(np.mean((w - _prior) ** 2))
 return ll
 
 return _run_slsqp(_nll, _prior, present, max_weight, 
 ["hgbr", "elo", "bayes", "poisson", "dc", "nb"], n, "Layer-1")


def optimize_score_distribution_weights(
 samples: list[dict],
 shrinkage: float = 0.15,
 prior: dict | None = None,
 max_weight: float = 0.7,
) -> dict:
 """Layer-2: 使用 1X2 LogLoss 优化 score distribution 权重。
 
 输入: samples 中包含 poisson, dc, nb (基于 fused λ 的 1X2 概率)
 输出: {hgbr, elo, bayes, poisson, dc, nb} (前三个为0)
 """
 candidates = ["poisson", "dc", "nb"]
 present = [n for n in candidates if any(n in s for s in samples)]
 
 if not present:
 return {name: 1.0 if name == "poisson" else 0.0 for name in ["hgbr", "elo", "bayes", "poisson", "dc", "nb"]}
 
 n = max(1, len(samples))
 
 if prior is None:
 _prior = _compute_baseline_prior_1x2(samples, present, n)
 else:
 _prior = np.array([prior.get(name, 0.0) for name in present], dtype=float)
 _prior = _prior / _prior.sum() if _prior.sum() > 0 else np.ones(len(present)) / len(present)
 
 if len(present) < 2:
 return {name: (1.0 if name == present[0] else 0.0) for name in ["hgbr", "elo", "bayes", "poisson", "dc", "nb"]}
 
 def _nll(w):
 ll = 0.0
 for s in samples:
 p = np.zeros(3)
 for i, name in enumerate(present):
 if name not in s:
 continue
 p += w[i] * np.asarray(s[name])
 p = np.clip(p, 1e-12, None)
 p = p / p.sum()
 ll -= math.log(p[s["actual"]])
 ll = ll / n
 if shrinkage > 0 and len(present) > 1:
 ll += shrinkage * float(np.mean((w - _prior) ** 2))
 return ll
 
 return _run_slsqp(_nll, _prior, present, max_weight,
 ["hgbr", "elo", "bayes", "poisson", "dc", "nb"], n, "Layer-2")


def _run_slsqp(objective, prior, candidates, max_weight, all_names, n, layer_name):
 """运行 SLSQP 优化(: bounded-simplex 初始化)。"""
 from scipy.optimize import minimize
 
 
 w0 = _project_to_bounded_simplex(prior, max_weight)
 
 res = minimize(
 objective, w0, method="SLSQP",
 bounds=[(0.0, max_weight)] * len(candidates),
 constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
 options={"maxiter": 200, "ftol": 1e-9},
 )
 
 if not res.success:
 raise OptimizationError(f"{layer_name} SLSQP failed: {res.message}")
 
 w = np.clip(res.x, 0.0, 1.0)
 w = w / w.sum()
 
 out = {name: 0.0 for name in all_names}
 for i, name in enumerate(candidates):
 out[name] = float(w[i])
 out["log_loss"] = float(objective(w))
 out["slsqp_success"] = res.success
 out["slsqp_message"] = res.message
 return out


def _project_to_bounded_simplex(prior, max_weight):
 """投影到 bounded simplex: Σw=1, 0≤w≤max_weight。
 
 FIX: 确保初始点合法。
 """
 w = np.asarray(prior, dtype=float).copy()
 w = np.clip(w, 0.0, max_weight)
 wsum = w.sum()
 if wsum > 0:
 w = w / wsum
 else:
 w = np.full_like(w, 1.0 / len(w))
 # 再次 clip 确保 ≤ max_weight
 w = np.clip(w, 0.0, max_weight)
 wsum = w.sum()
 if wsum > 0:
 w = w / wsum
 return w


def learn_weights(
 samples: list[dict],
 tau: float = 0.0,
 phi: float = 1e9,
 shrinkage: float = 0.15,
 prior: dict | None = None,
 max_weight: float = 0.7,
) -> dict:
 """兼容旧接口:调用拆分后的 optimizer。
 
 .. deprecated:: 33cd2d2
 使用 optimize_goal_lambda_weights() 和 optimize_score_distribution_weights() 代替。
 此函数仅用于向后兼容。
 
 自动检测输入格式:
 - OOF samples (含 att_diff, hgbr_lam_h/a): 调用 build_member_samples
 - Member samples (含 hgbr, poisson, dc, nb 等 1X2 概率): 直接使用
 
 Layer-1: Poisson Goal NLL (λ fusion)
 Layer-2: 1X2 LogLoss (score distribution)
 """
 import warnings
 warnings.warn(
 "learn_weights() is deprecated. Use optimize_goal_lambda_weights() and optimize_score_distribution_weights() instead.",
 DeprecationWarning,
 stacklevel=2,
 )
 # 检测输入格式
 is_oof_samples = len(samples) > 0 and "att_diff" in samples[0]
 
 if is_oof_samples:
 # OOF samples → 需要 build_member_samples
 try:
 from app.services.training.ensemble.member_builder import build_member_samples
 except ImportError:
 def build_member_samples(samps, tau, phi, weights=None):
 return samps
 
 samples_stage1 = build_member_samples(samples, tau=0.0, phi=1e9)
 w_layer1 = optimize_goal_lambda_weights(samples_stage1, shrinkage=shrinkage, prior=prior, max_weight=max_weight)
 
 samples_stage2 = build_member_samples(samples, tau, phi, weights=w_layer1)
 w_layer2 = optimize_score_distribution_weights(samples_stage2, shrinkage=shrinkage, prior=prior, max_weight=max_weight)
 else:
 # 已经是 member samples (含 hgbr, poisson, dc, nb 等)
 # 直接用于 Layer-2 优化
 w_layer2 = optimize_score_distribution_weights(samples, shrinkage=shrinkage, prior=prior, max_weight=max_weight)
 
 # Layer-1 使用默认权重(因为没有 λ 值)
 w_layer1 = {"hgbr": 1.0, "elo": 0.0, "bayes": 0.0, "poisson": 0.0, "dc": 0.0, "nb": 0.0}
 
 return {
 "hgbr": w_layer1.get("hgbr", 0.0),
 "elo": w_layer1.get("elo", 0.0),
 "bayes": w_layer1.get("bayes", 0.0),
 "poisson": w_layer2.get("poisson", 0.0),
 "dc": w_layer2.get("dc", 0.0),
 "nb": w_layer2.get("nb", 0.0),
 "gbm": 0.0,
 "log_loss": w_layer2.get("log_loss", 0.0),
 "n": len(samples),
 "shrinkage": shrinkage,
 }


def _compute_baseline_prior_poisson(samples, candidates, n):
 """基于 Poisson NLL 计算 Layer-1 先验。"""
 if len(samples) < 30:
 return np.array(
 [0.6 if name == "hgbr" else 0.4 / max(1, len(candidates) - 1)
 for name in candidates],
 dtype=float,
 )
 _nll = {}
 for name in candidates:
 _v = []
 for s in samples:
 try:
 lam_h = s.get(f"{name}_lam_h", 1.5)
 lam_a = s.get(f"{name}_lam_a", 1.2)
 gh = int(s.get("home_goals", 0))
 ga = int(s.get("away_goals", 0))
 from math import lgamma, log
 ll = -(gh * log(max(lam_h, 1e-12)) - lam_h - lgamma(gh + 1))
 ll -= (ga * log(max(lam_a, 1e-12)) - lam_a - lgamma(ga + 1))
 _v.append(-ll)
 except Exception:
 pass
 _nll[name] = float(np.mean(_v)) if _v else math.inf
 return _prior_from_nll(_nll, candidates)


def _compute_baseline_prior_1x2(samples, candidates, n):
 """基于 1X2 NLL 计算 Layer-2 先验。"""
 if len(samples) < 30:
 return np.array(
 [0.6 if name == "poisson" else 0.4 / max(1, len(candidates) - 1)
 for name in candidates],
 dtype=float,
 )
 _nll = {}
 for name in candidates:
 _v = []
 for s in samples:
 try:
 _p = np.clip(np.asarray(s[name], dtype=float), 1e-12, None)
 _p = _p / _p.sum()
 _v.append(-math.log(_p[int(s["actual"])]))
 except Exception:
 pass
 _nll[name] = float(np.mean(_v)) if _v else math.inf
 return _prior_from_nll(_nll, candidates)


def _prior_from_nll(_nll, candidates):
 """从 NLL 计算 prior(softmax)。"""
 _T = 0.15
 _logits = np.array([-_nll.get(n, math.inf) / _T for n in candidates])
 _inf_mask = np.array([not math.isfinite(_nll.get(n, math.inf)) for n in candidates])
 if np.all(_inf_mask):
 return np.ones(len(candidates)) / len(candidates)
 _m = _logits[~_inf_mask].max()
 _e = np.zeros(len(candidates))
 _e[~_inf_mask] = np.exp(_logits[~_inf_mask] - _m)
 return _e / _e.sum()


def load_weights(league_key: str, default: EnsembleWeights | None = None) -> dict:
 """加载该联赛学习到的权重(返回 flat 格式)。"""
 if default is None:
 default = DEFAULT_WEIGHTS
 
 path = _WEIGHTS_PATH
 if path is None:
 try:
 from app.core.paths import ARTIFACTS_DIR as _AD
 path = os.path.join(str(_AD), "ensemble", "ensemble_weights.json")
 except Exception:
 path = None
 
 if not path or not os.path.exists(path):
 return default.to_flat()
 
 try:
 with open(path, encoding="utf-8") as _wf:
 data = json.load(_wf)
 except (json.JSONDecodeError, UnicodeDecodeError, OSError) as _e:
 raise ValueError(f"ensemble_weights.json 损坏: {_e}") from _e
 
 if not isinstance(data, dict):
 raise TypeError(f"ensemble_weights.json 顶层应为 dict,实际 {type(data).__name__}")
 
 if league_key in data:
 _entry = data[league_key]
 if not isinstance(_entry, dict):
 raise TypeError(f"ensemble_weights[{league_key}] 应为 dict,实际 {type(_entry).__name__}")
 if _is_layered(_entry):
 return from_layered(_entry)
 return _entry
 
 return default.to_flat()
