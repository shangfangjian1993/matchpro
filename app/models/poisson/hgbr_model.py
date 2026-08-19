"""基础模型框架 - HGBR + Poisson损失函数"""

import logging
import math
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoissonLossHGBR(BaseEstimator, RegressorMixin):
    """
    自定义HGBR模型，支持Poisson损失函数

    使用histogram-based gradient boosting进行足球比赛得分预测
    """

    def __init__(
        self,
        loss: str = "poisson",
        learning_rate: float = 0.1,
        max_depth: int = 6,
        min_samples_leaf: int = 10,
        random_state: int = 42,
        max_iter: int = 100,
        validation_fraction: float = 0.2,
        early_stopping_rounds: int = 10,
        verbose: bool = True,
    ):
        """
        初始化HGBR模型

        Args:
            loss: 损失函数，支持"poisson"
            learning_rate: 学习率
            max_depth: 最大深度
            min_samples_leaf: 叶节点最小样本数
            random_state: 随机种子
            max_iter: 最大迭代次数
            validation_fraction: 验证集比例
            early_stopping_rounds: 早停轮数(映射为 sklearn 的 n_iter_no_change)
            verbose: 是否显示训练信息
        """
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.max_iter = max_iter
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose

        # 模型属性
        self.model = None
        self.feature_importance_ = None
        self.best_iteration_ = None
        self.validation_score_ = None

        # 验证损失函数
        if loss not in ["poisson"]:
            raise ValueError(f"不支持的损失函数: {loss}，仅支持poisson")

    def _validate_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        验证和预处理数据

        Args:
            X: 特征数据
            y: 目标变量

        Returns:
            处理后的特征和目标数据
        """
        # 检查数据
        if len(X) != len(y):
            raise ValueError("特征数据和目标变量长度不匹配")

        # 统一索引，避免重复索引导致的对齐错位
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        # 只保留数值特征列（球队名等字符串列不能进树模型）
        X = X.select_dtypes(include=[np.number])

        # 移除目标变量为 NaN 的行
        finite_mask = y.notna().values
        X = X[finite_mask]
        y = y[finite_mask]

        # 确保目标变量为非负整数（符合Poisson分布）
        y = y.round().astype(int)
        y = np.maximum(y, 0)

        return X, y

    def _calculate_poisson_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算Poisson损失函数

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            Poisson损失值
        """
        # 防止预测值为0或负数
        y_pred = np.maximum(y_pred, 1e-10)

        # Poisson损失: λ - y * log(λ)
        loss = np.mean(y_pred - y_true * np.log(y_pred))
        return loss

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None
    ) -> "PoissonLossHGBR":
        """
        训练模型

        Args:
            X: 特征数据
            y: 目标变量

        Returns:
            训练好的模型
        """
        logger.info("开始训练HGBR模型...")

        # 验证数据
        X, y = self._validate_data(X, y)

        # 划分训练集和验证集（shuffle=False 按时间顺序切分;数据过少时不切分）
        if self.validation_fraction > 0 and len(X) >= 10:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                shuffle=False,
            )
            w_train = None
            if sample_weight is not None:
                _split = int(len(X) * (1 - self.validation_fraction))
                w_train = np.asarray(sample_weight, dtype=float)[:_split]
        else:
            X_train, X_val, y_train, y_val = X, None, y, None
            w_train = (
                np.asarray(sample_weight, dtype=float)
                if sample_weight is not None
                else None
            )

        # 训练梯度提升模型
        self.model = self._train_gradient_boosting(
            X_train, y_train, X_val, y_val, w_train
        )

        # 计算特征重要性
        self.feature_importance_ = self._calculate_feature_importance()

        logger.info("模型训练完成")
        return self

    def _train_gradient_boosting(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
        w_train: np.ndarray | None = None,
    ) -> Any:
        """
        训练梯度提升模型

        Args:
            X_train: 训练集特征
            y_train: 训练集目标
            X_val: 验证集特征
            y_val: 验证集目标

        Returns:
            训练好的模型
        """
        from sklearn.ensemble import HistGradientBoostingRegressor

        # 构造 HGBR 参数(early_stopping_rounds 映射为 n_iter_no_change,
        # 且必须显式开启 early_stopping 才生效)
        params = {
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            "validation_fraction": self.validation_fraction,
            "verbose": self.verbose,
        }
        if self.early_stopping_rounds is not None:
            params["early_stopping"] = True
            params["n_iter_no_change"] = self.early_stopping_rounds

        # 创建并训练模型
        model = HistGradientBoostingRegressor(**params)
        model.fit(X_train, y_train, sample_weight=w_train)

        # 保存最佳迭代次数
        if hasattr(model, "n_iter_"):
            self.best_iteration_ = model.n_iter_

        # 计算验证分数
        if X_val is not None and y_val is not None:
            val_pred = model.predict(X_val)
            self.validation_score_ = self._calculate_poisson_loss(
                y_val.values, val_pred
            )
            logger.info(f"验证集Poisson损失: {self.validation_score_:.4f}")

        return model

    def _calculate_feature_importance(self) -> pd.Series:
        """
        计算特征重要性

        Returns:
            特征重要性Series
        """
        if self.model is None:
            return pd.Series()

        try:
            if hasattr(self.model, "feature_importances_"):
                return pd.Series(
                    self.model.feature_importances_, index=self.model.feature_names_in_
                )
            else:
                # HGBR 无 feature_importances_:不返回伪造占位值,返回空并标注不可用
                return pd.Series(dtype=float, name="unavailable")
        except Exception:
            return pd.Series()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测目标变量

        Args:
            X: 特征数据

        Returns:
            预测的目标变量值
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit方法")

        # 确保特征名称一致
        if hasattr(self.model, "feature_names_in_"):
            missing_features = set(self.model.feature_names_in_) - set(X.columns)
            if missing_features:
                raise ValueError(f"缺少必要特征: {missing_features}")

            # 只使用模型训练时使用的特征
            X = X[self.model.feature_names_in_]

        # 预测
        predictions = self.model.predict(X)

        # 确保预测值为非负
        predictions = np.maximum(predictions, 0)

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测概率分布（Poisson分布）

        Args:
            X: 特征数据

        Returns:
            每个可能得分的概率（行内归一化，和为1）
        """
        predictions = self.predict(X)

        # 计算Poisson分布的概率（向量化 + 对数空间，避免溢出）
        max_goals = 10  # 预测最多10个球
        k = np.arange(max_goals + 1, dtype=float)
        # log(k!) 预计算
        log_fact_k = np.array([math.lgamma(i + 1) for i in range(max_goals + 1)])

        lambdas = np.maximum(predictions, 1e-10)[:, np.newaxis]

        # log P(k) = -λ + k·log(λ) - log(k!)
        log_probs = (
            -lambdas + k[np.newaxis, :] * np.log(lambdas) - log_fact_k[np.newaxis, :]
        )
        probabilities = np.exp(log_probs)

        # 行内归一化（截断分布，保证概率和为1）
        row_sums = probabilities.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-12)
        probabilities = probabilities / row_sums

        return probabilities

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """
        评估模型性能

        Args:
            X: 特征数据
            y: 真实目标变量

        Returns:
            评估指标字典
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit方法")

        # 预测
        y_pred = self.predict(X)

        # 计算各种指标
        metrics = {
            "mse": mean_squared_error(y, y_pred),
            "mae": mean_absolute_error(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "poisson_loss": self._calculate_poisson_loss(y.values, y_pred),
        }

        return metrics

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, cv: int = 5
    ) -> dict[str, float]:
        """
        交叉验证（时间序列切分，使用Poisson损失作为评分）

        Args:
            X: 特征数据
            y: 目标变量
            cv: 交叉验证折数

        Returns:
            交叉验证结果
        """
        from sklearn.model_selection import cross_val_score

        # 使用时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=cv)

        # 使用负Poisson损失作为评分
        def _neg_poisson_loss(y_true, y_pred):
            y_pred = np.maximum(y_pred, 1e-10)
            return -np.mean(y_pred - y_true * np.log(y_pred))

        poisson_scorer = make_scorer(_neg_poisson_loss, greater_is_better=True)
        scores = cross_val_score(self, X, y, cv=tscv, scoring=poisson_scorer)

        return {"cv_scores": scores, "cv_mean": -scores.mean(), "cv_std": scores.std()}

    def save_model(self, filepath: str) -> None:
        """
        保存模型

        Args:
            filepath: 模型保存路径
        """
        if self.model is None:
            raise ValueError("模型未训练，无法保存")

        # 创建保存目录
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        # 保存模型
        joblib.dump(
            {
                "model": self.model,
                "feature_importance": self.feature_importance_,
                "best_iteration": self.best_iteration_,
                "validation_score": self.validation_score_,
                "params": {
                    "loss": self.loss,
                    "learning_rate": self.learning_rate,
                    "max_depth": self.max_depth,
                    "min_samples_leaf": self.min_samples_leaf,
                    "random_state": self.random_state,
                    "early_stopping_rounds": self.early_stopping_rounds,
                    "max_iter": self.max_iter,
                    "validation_fraction": self.validation_fraction,
                },
            },
            filepath,
        )

        logger.info(f"模型已保存到: {filepath}")

    def load_model(self, filepath: str) -> "PoissonLossHGBR":
        """
        加载模型

        Args:
            filepath: 模型文件路径

        Returns:
            加载的模型
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")

        # 加载模型
        model_data = joblib.load(filepath)

        if not isinstance(model_data, dict) or "model" not in model_data:
            raise ValueError(f"模型文件格式无效: {filepath}")

        # 恢复模型属性
        self.model = model_data["model"]
        self.feature_importance_ = model_data.get("feature_importance")
        self.best_iteration_ = model_data.get("best_iteration")
        self.validation_score_ = model_data.get("validation_score")

        # 恢复参数
        params = model_data.get("params", {})
        self.loss = params.get("loss", self.loss)
        self.learning_rate = params.get("learning_rate", self.learning_rate)
        self.max_depth = params.get("max_depth", self.max_depth)
        self.min_samples_leaf = params.get("min_samples_leaf", self.min_samples_leaf)
        self.random_state = params.get("random_state", self.random_state)
        self.early_stopping_rounds = params.get(
            "early_stopping_rounds", self.early_stopping_rounds
        )
        self.max_iter = params.get("max_iter", self.max_iter)
        self.validation_fraction = params.get(
            "validation_fraction", self.validation_fraction
        )

        logger.info(f"模型已从 {filepath} 加载")
        return self
