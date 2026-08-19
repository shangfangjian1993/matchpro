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

        # 审查 A70A601 P1-7:移除外层 train_test_split 嵌套切分。
        # 旧实现:外层按时间切 15% 验证 + HGBR 内部再按 validation_fraction
        # 随机切验证做早停 → 有效训练样本二次折损(0.85×0.85≈0.72)。
        # 新实现:早停验证交由 HGBR 内部(validation_fraction 一次切分)承担;
        # 时间序验证由 Walk-forward/回测负责,不由单次 fit 承担。
        self.model = self._train_gradient_boosting(X, y, None, None, sample_weight)

        # 计算特征重要性(审查 A70A601 §22:HGBR 无原生 importance →
        # permutation importance 提供可解释 contribution)
        self.feature_importance_ = self._calculate_feature_importance(X, y)

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

    def _calculate_feature_importance(self, X=None, y=None) -> pd.Series:
        """
        计算特征重要性(审查 A70A601 §22)。

        审查 §22 推荐 SHAP;HGBR 无原生 feature_importances_ 时按序:
          1) shap.TreeExplainer(sklearn HistGB 支持时,快);
          2) shap.PermutationExplainer(predict 兼容兜底);
          3) sklearn permutation importance(负 Poisson 偏差)最终兜底。
        均取 |贡献| 均值(值越大 = 该列越重要)。返回 Series(index=特征名);
        极端小样本/失败时返回空(标注 unavailable)。
        """
        if self.model is None:
            return pd.Series()
        try:
            if hasattr(self.model, "feature_importances_"):
                self.importance_method_ = "native"
                return pd.Series(
                    self.model.feature_importances_, index=self.model.feature_names_in_
                )
            # 代表性样本控制计算量(全量混洗开销高;可解释性不受影响)
            _n = min(200, len(X) if X is not None else 0)
            if _n == 0:
                return pd.Series(dtype=float, name="unavailable")
            _X = X.iloc[:_n] if hasattr(X, "iloc") else X[:_n]
            _y = y.iloc[:_n] if hasattr(y, "iloc") else y[:_n]

            # 1/2) SHAP(审查 §22 推荐:TreeExplainer 优先)
            try:
                import shap

                _sv = None
                _names = getattr(self.model, "feature_names_in_", None)
                try:
                    _explainer = shap.TreeExplainer(self.model)
                    _sv = _explainer.shap_values(_X)
                    self.importance_method_ = "shap_tree"
                except Exception:
                    _sv = None
                if _sv is None:
                    try:
                        _explainer = shap.PermutationExplainer(
                            self.model.predict,
                            _X,
                            max_evals=min(80, 2 * _X.shape[1]),
                            progress_bar=False,
                        )
                        _sv = _explainer(_X).values
                        self.importance_method_ = "shap_permutation"
                    except Exception:
                        _sv = None
                if _sv is not None:
                    _abs = np.abs(np.asarray(_sv, dtype=float)).mean(axis=0)
                    return pd.Series(
                        _abs, index=_names if _names is not None else range(len(_abs))
                    )
            except Exception:
                pass

            # 3) permutation importance 兜底
            from sklearn.inspection import permutation_importance

            # 代表性样本控制计算量(全量混洗开销高;可解释性不受影响)
            _n = min(200, len(X) if X is not None else 0)
            if _n == 0:
                return pd.Series(dtype=float, name="unavailable")
            _X = X.iloc[:_n] if hasattr(X, "iloc") else X[:_n]
            _y = y.iloc[:_n] if hasattr(y, "iloc") else y[:_n]
            _pi = permutation_importance(
                self.model,
                _X,
                _y,
                n_repeats=3,
                random_state=self.random_state,
                scoring="neg_mean_poisson_deviance",
                n_jobs=-1,
            )
            self.importance_method_ = "permutation"
            _names2 = getattr(self.model, "feature_names_in_", None) or range(
                len(_pi.importances_mean)
            )
            return pd.Series(_pi.importances_mean, index=_names2)
        except Exception:
            return pd.Series(dtype=float, name="unavailable")

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
        from sklearn.model_selection import TimeSeriesSplit, cross_val_score

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
