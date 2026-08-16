"""三分类 GBM 成员(评审 P1 Ensemble 多样性)。

与 HGBR-Poisson 成员互补:直接建模胜平负(sklearn HistGradientBoostingClassifier),
输出三分类概率 P_gbm,参与 Ensemble 概率层融合(P_final = Σw·P)。

评估先行:独立训练 + holdout 指标,数据驱动决定接入权重。
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd


class GbmClassifier:
    """轻量三分类 GBM(与主 HGBR 同特征,目标 = 胜/平/负)。"""

    def __init__(self, params: dict | None = None):
        self.params = params or {
            "max_depth": 4, "learning_rate": 0.05, "max_iter": 200,
            "min_samples_leaf": 20, "random_state": 42,
        }
        self.model = None
        self.feature_columns_ = []
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """X: 特征矩阵(78 特征), y: 0/1/2(胜/平/负)。"""
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.model = HistGradientBoostingClassifier(**self.params)
        self.model.fit(X, y)
        self.feature_columns_ = list(X.columns)
        self.is_trained = True
        # 时间 holdout 评估(与主模型同口径)
        n = len(X)
        split = int(n * 0.8)
        X_eval, y_eval = X.iloc[split:], y.iloc[split:]
        p = self.model.predict_proba(X_eval)
        from app.replay.metrics import accuracy, brier_score, log_loss
        from app.replay.metrics import rps as _rps
        return {
            "log_loss": round(sum(log_loss(p[i], y_eval.iloc[i]) for i in range(len(y_eval))) / len(y_eval), 5),
            "brier": round(sum(brier_score(p[i], y_eval.iloc[i]) for i in range(len(y_eval))) / len(y_eval), 5),
            "rps": round(sum(_rps(p[i], y_eval.iloc[i]) for i in range(len(y_eval))) / len(y_eval), 5),
            "accuracy": round(accuracy([list(x) for x in p], y_eval.tolist()), 4),
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("模型未训练")
        cols = [c for c in self.feature_columns_ if c in X.columns]
        return self.model.predict_proba(X[cols])

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "features": self.feature_columns_,
                         "params": self.params}, f)

    @classmethod
    def load(cls, path: str) -> GbmClassifier | None:
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(d["params"])
        obj.model = d["model"]
        obj.feature_columns_ = d["features"]
        obj.is_trained = True
        return obj
