"""概率校准引擎:raw 概率 → calibrated 概率(评审 多方法并存 + 联赛择优)。

方法(统一接口):
- Beta(默认):三分类 β 校准,log-loss 最优,单调性天然保证
- Platt:每类 logistic 回归(基于 logit(p))
- Isotonic:每类等渗回归(非参数,样本充足时最灵活)

拟合:fit_best() 在训练段拟合三种,评估段选 ECE 最低(联赛择优)。
持久化:.cal 文件 {"method", "params"}(与模型版本同目录绑定)。
"""

from __future__ import annotations

import os
import pickle

import numpy as np

_KEYS = ["home_win", "draw", "away_win"]


def _ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
 """Expected Calibration Error(评估用,越小越好)。"""
 conf = probs.max(axis=1)
 acc = (probs.argmax(axis=1) == labels).astype(float)
 edges = np.linspace(0.0, 1.0, n_bins + 1)
 total = 0.0
 for i in range(n_bins):
 mask = (conf > edges[i]) & (conf <= edges[i + 1])
 if mask.sum() == 0:
 continue
 total += mask.sum() * abs(acc[mask].mean() - conf[mask].mean())
 return total / max(1, len(conf))


def _softmax(scores: np.ndarray) -> np.ndarray:
 scores = scores - scores.max(axis=1, keepdims=True)
 e = np.exp(scores)
 return e / e.sum(axis=1, keepdims=True)


class Calibrator:
 """三分类概率校准器基类(Beta 为默认实现,保持旧接口兼容)。"""

 method = "beta"

 def __init__(self, alpha=None, beta=None, fitted_n: int = 0):
 self.alpha = alpha if alpha is not None else np.ones(3)
 self.beta = beta if beta is not None else np.ones(3)
 self.fitted_n = fitted_n

 # ---------------- 拟合 ----------------
 @classmethod
 def fit_beta(
 cls, probs: np.ndarray, labels: np.ndarray, max_iter: int = 1000
 ) -> Calibrator:
 """Beta 校准(三分类):P_cal[i] ∝ exp(alpha_i + beta_i * log(p_i))。"""
 n, k = probs.shape
 alpha = np.zeros(k)
 beta = np.ones(k)

 def _nll(params):
 a = params[:k]
 b = params[k:]
 logp = np.log(np.clip(probs, 1e-9, 1.0))
 soft = _softmax(a[None, :] + b[None, :] * logp)
 return -np.mean(np.log(np.clip(soft[np.arange(n), labels], 1e-9, 1.0)))

 params = np.concatenate([alpha, beta])
 m = np.zeros_like(params)
 v = np.zeros_like(params)
 lr, beta1, beta2, eps = 0.1, 0.9, 0.999, 1e-8
 best, best_params = float("inf"), params.copy()
 for step in range(max_iter):
 grad = np.zeros_like(params)
 h = 1e-5
 for i in range(len(params)):
 pp = params.copy()
 pp[i] += h
 pm = params.copy()
 pm[i] -= h
 grad[i] = (_nll(pp) - _nll(pm)) / (2 * h)
 m = beta1 * m + (1 - beta1) * grad
 v = beta2 * v + (1 - beta2) * grad * grad
 mhat = m / (1 - beta1 ** (step + 1))
 vhat = v / (1 - beta2 ** (step + 1))
 params -= lr * mhat / (np.sqrt(vhat) + eps)
 val = _nll(params)
 if val < best:
 best, best_params = val, params.copy()
 if step > 50 and step % 100 == 0 and abs(best - val) < 1e-6:
 break
 return cls(alpha=best_params[:k], beta=best_params[k:], fitted_n=n)

 @classmethod
 def fit_platt(cls, probs: np.ndarray, labels: np.ndarray) -> Calibrator:
 """Platt:每类 logistic 回归(特征 = logit(p_i),评审 P1)。"""
 from sklearn.linear_model import LogisticRegression

 n, k = probs.shape
 coef, intercept = [], []
 for c in range(k):
 p = np.clip(probs[:, c], 1e-9, 1 - 1e-9)
 x = np.log(p / (1 - p)).reshape(-1, 1)
 y = (labels == c).astype(int)
 lr = LogisticRegression(C=1.0, max_iter=500)
 lr.fit(x, y)
 coef.append(float(lr.coef_[0][0]))
 intercept.append(float(lr.intercept_[0]))
 cal = cls(alpha=np.array(coef), beta=np.array(intercept), fitted_n=n)
 cal.method = "platt"
 return cal

 @classmethod
 def fit_isotonic(cls, probs: np.ndarray, labels: np.ndarray) -> Calibrator:
 """Isotonic:每类等渗回归(非参数,评审 P1)。"""
 from sklearn.isotonic import IsotonicRegression

 n, k = probs.shape
 regs = []
 for c in range(k):
 reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
 reg.fit(probs[:, c], (labels == c).astype(float))
 regs.append(reg)
 cal = cls(fitted_n=n)
 cal._regs = regs
 cal.method = "isotonic"
 return cal

 @classmethod
 def fit_best(
 cls,
 probs: np.ndarray,
 labels: np.ndarray,
 val_fraction: float = 0.2,
 test_fraction: float = 0.0,
 ) -> Calibrator:
 """联赛择优(评审 P1 + 第五轮

 三段切分(时间序):
 Train(60%) → 拟合三种方法
 Validation(20%) → 选 ECE 最低
 Test(20%,test_fraction>0 时) → 独立最终报告(选择偏差后)
 选中方法在 Train+Validation 上重训(更多数据),Test 段 ECE 记录为
 _test_ece —— 最终 ECE 不再是选择偏差下的值。
 """
 n = len(probs)
 if test_fraction > 0:
 test_n = int(n * test_fraction)
 trva_p, trva_l = probs[:-test_n], labels[:-test_n]
 test_p, test_l = probs[-test_n:], labels[-test_n:]
 else:
 trva_p, trva_l, test_p, test_l = probs, labels, None, None
 split = int(len(trva_p) * (1 - val_fraction))
 tr_p, tr_l = trva_p[:split], trva_l[:split]
 va_p, va_l = trva_p[split:], trva_l[split:]
 cands = []
 for fitter, name in (
 (cls.fit_beta, "beta"),
 (cls.fit_platt, "platt"),
 (cls.fit_isotonic, "isotonic"),
 ):
 try:
 cal = fitter(tr_p, tr_l)
 cal_p = cal.apply_matrix(va_p)
 ece = _ece_score(cal_p, va_l)
 cands.append((ece, name, cal))
 except Exception as _exc:
 import logging as _lg

 _lg.getLogger(__name__).debug("拟合候选失败,跳过: %s", _exc)
 continue
 if not cands:
 return cls.identity()
 cands.sort(key=lambda t: t[0])
 best = cands[0][2]
 # 全量(train+val)重训选中方法 —— 生产校准器用更多数据
 try:
 best = (
 cands[0][1] == "beta"
 and cls.fit_beta(trva_p, trva_l)
 or cands[0][1] == "platt"
 and cls.fit_platt(trva_p, trva_l)
 or cls.fit_isotonic(trva_p, trva_l)
 )
 except Exception:
 best = cands[0][2]
 best.fitted_n = n
 best._ece = cands[0][0]
 if test_p is not None:
 best._test_ece = _ece_score(best.apply_matrix(test_p), test_l)
 best._test_n = len(test_p)
 else:
 best._test_ece = None
 best._test_n = 0
 return best

 # ---------------- 应用 ----------------
 def apply_matrix(self, probs: np.ndarray) -> np.ndarray:
 """校准概率矩阵(批量)。"""
 logp = np.log(np.clip(probs, 1e-9, 1.0))
 return _softmax(self.alpha[None, :] + self.beta[None, :] * logp)

 def apply(self, probs: dict) -> dict:
 """校准单场概率 dict;返回新 dict(不修改原)。"""
 p = np.array([probs.get(k, 0.0) for k in _KEYS], dtype=float)
 if self.method == "platt":
 cal = self._apply_platt(p)
 elif self.method == "isotonic":
 cal = self._apply_isotonic(p)
 else:
 cal = self.apply_matrix(p.reshape(1, -1))[0]
 # 6 失败保护:∑=1 + 单调性(Beta 需 beta>0)
 if not np.isclose(cal.sum(), 1.0, atol=1e-6):
 cal = cal / cal.sum()
 if self.method == "beta" and np.any(np.asarray(self.beta) <= 0):
 out = dict(probs)
 out["calibration"] = 0
 return out
 out = dict(probs)
 for k, v in zip(_KEYS, cal):
 out[k] = round(float(v), 4)
 out["calibration"] = round(self.fitted_n, 0)
 out["cal_method"] = self.method
 return out

 def _apply_platt(self, p: np.ndarray) -> np.ndarray:
 from scipy.special import expit

 pc = np.clip(p, 1e-9, 1 - 1e-9)
 x = np.log(pc / (1 - pc))
 cal = np.array([expit(self.alpha[c] * x[c] + self.beta[c]) for c in range(3)])
 return cal / cal.sum()

 def _apply_isotonic(self, p: np.ndarray) -> np.ndarray:
 cal = np.array([self._regs[c].predict([p[c]])[0] for c in range(3)])
 cal = np.clip(cal, 1e-9, None)
 return cal / cal.sum()

 # ---------------- 持久化 ----------------
 def save(self, path: str) -> None:
 payload = {
 "method": self.method,
 "n": self.fitted_n,
 "alpha": self.alpha,
 "beta": self.beta,
 }
 if self.method == "isotonic":
 
 # 仅存阈值再手工重建会缺 X_min_/X_max_/f_ 等内部属性,
 # load 后 apply 直接 AttributeError。
 payload["regs"] = [pickle.dumps(r) for r in self._regs]
 with open(path, "wb") as f:
 pickle.dump(payload, f)

 @classmethod
 def load(cls, path: str) -> Calibrator | None:
 if not os.path.exists(path):
 return None
 with open(path, "rb") as f:
 d = pickle.load(f)
 cal = cls(alpha=d.get("alpha"), beta=d.get("beta"), fitted_n=d.get("n", 0))
 cal.method = d.get("method", "beta")
 if cal.method == "isotonic":
 from sklearn.isotonic import IsotonicRegression

 regs = []
 for item in d.get("regs", []):
 if isinstance(item, (bytes, bytearray)):
 regs.append(pickle.loads(item)) # 新格式:完整对象
 else:
 # 旧格式(仅阈值):尽力重建(缺 f_ 时退化为线性插值)
 xt, yt = item
 r = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
 r.X_thresholds_ = xt
 r.y_thresholds_ = yt
 r.X_min_ = min(xt)
 r.X_max_ = max(xt)
 try:
 from scipy.interpolate import interp1d

 r.f_ = interp1d(
 xt, yt, bounds_error=False, fill_value=(yt[0], yt[-1])
 )
 except Exception:
 r.f_ = np.interp
 regs.append(r)
 cal._regs = regs
 return cal

 @classmethod
 def identity(cls) -> Calibrator:
 return cls(fitted_n=-1)


if __name__ == "__main__":
 # 自测:三种方法拟合 + 择优
 rng = np.random.default_rng(42)
 n = 1200
 true_p = np.array([0.5, 0.3, 0.2])
 labels = rng.choice(3, size=n, p=true_p)
 raw = np.zeros((n, 3))
 for i in range(n):
 raw[i] = true_p
 raw[i, labels[i]] += 0.3
 raw[i] /= raw[i].sum()
 best = Calibrator.fit_best(raw, labels)
 print(f"✅ 择优方法: {best.method} | ECE: {getattr(best, '_ece', 'N/A'):.4f}")
 sample = {"home_win": 0.9, "draw": 0.06, "away_win": 0.04}
 print(f" 校准前: {sample}")
 print(f" 校准后: {best.apply(sample)}")
