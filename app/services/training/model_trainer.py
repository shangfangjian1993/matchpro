"""模型训练和评估系统"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.core.config import (
 TOURNAMENT_LEAGUE_TYPES,
 LeagueType,
 ModelConfig,
 MultiLeagueConfig,
)
from app.models.poisson.hgbr_model import PoissonLossHGBR
from app.models.poisson.league_factory import LeagueModelFactory
from app.models.poisson.tournament_factory import TournamentModelFactory

logger = logging.getLogger(__name__)


def _prepare_eval_split(
 model, data: pd.DataFrame, target_column: str, test_split_ratio: float = 0.2
):
 """
 准备时间外评估数据:prepare 特征 → 白名单选列 → date 排序 → 后 20% 切分。

 数据太少时退回全量。返回 (X, y, feature_columns)。
 供 ModelTrainer._evaluate_model / ModelValidator.validate_model /
 ModelValidator.monitor_model_performance 共用,保证评估口径一致。
 """
 prepared_data = (
 model.prepare_league_specific_features(data)
 if hasattr(model, "prepare_league_specific_features")
 else (
 model.prepare_tournament_specific_features(data)
 if hasattr(model, "prepare_tournament_specific_features")
 else data
 )
 )

 # 特征选择(数值列白名单)
 if hasattr(model, "_select_feature_columns"):
 feature_columns = model._select_feature_columns(prepared_data, target_column)
 else:
 feature_columns = [
 c
 for c in prepared_data.columns
 if c != target_column
 and c not in ["date", "league", "season"]
 and pd.api.types.is_numeric_dtype(prepared_data[c])
 ]

 # 按时间排序后切分:前 (1-ratio) 训练,后 ratio 时间外评估
 if "date" in prepared_data.columns and not prepared_data["date"].isna().all():
 prepared_data = prepared_data.sort_values("date", kind="mergesort").reset_index(
 drop=True
 )
 from app.models.utils import date_group_split

 _trn, _tst = date_group_split(prepared_data, ratio=1 - test_split_ratio)
 X_train = _trn[feature_columns]
 y_train = _trn[target_column]
 X_test = _tst[feature_columns]
 y_test = _tst[target_column]
 if len(X_test) == 0:
 # 数据太少时退回全量(与 train() 行为一致)
 X_test, y_test = X_train, y_train
 return X_train, y_train, X_test, y_test, feature_columns


def compute_evaluation_metrics(y_true, y_pred) -> dict[str, float]:
 """
 统一评估指标:MSE/MAE/RMSE/Poisson损失 + 精确率/±1/±2 准确率。

 ModelTrainer 与 ModelValidator 共用,避免两套重复实现。
 """
 y_true = np.asarray(y_true)
 y_pred = np.asarray(y_pred)
 mse = mean_squared_error(y_true, y_pred)
 mae = mean_absolute_error(y_true, y_pred)
 rmse = np.sqrt(mse)
 poisson_loss = float(np.mean(y_pred - y_true * np.log(np.maximum(y_pred, 1e-10))))
 y_pred_rounded = np.maximum(np.round(y_pred).astype(int), 0)
 return {
 "mse": mse,
 "mae": mae,
 "rmse": rmse,
 "poisson_loss": poisson_loss,
 "exact_accuracy": float(np.mean(y_pred_rounded == y_true)),
 "within_one_accuracy": float(np.mean(np.abs(y_pred_rounded - y_true) <= 1)),
 "within_two_accuracy": float(np.mean(np.abs(y_pred_rounded - y_true) <= 2)),
 }


class ModelTrainer:
 """
 模型训练器
 """

 def __init__(self, config: ModelConfig | None = None):
 """
 初始化模型训练器

 Args:
 config: 模型配置
 """
 self.config = config or MultiLeagueConfig().base_model_config
 self.model = None
 self.training_history = []
 self.cross_validation_results = {}
 self.feature_importance = {}
 self.model_metrics = {}

 def train_model(
 self,
 data: pd.DataFrame,
 league_type: LeagueType,
 target_column: str = "goals",
 cross_validation: bool = True,
 cv_folds: int = 5,
 ) -> dict[str, Any]:
 """
 训练模型

 Args:
 data: 训练数据
 league_type: 联赛类型
 target_column: 目标变量列名
 cross_validation: 是否进行交叉验证
 cv_folds: 交叉验证折数

 Returns:
 训练结果
 """
 logger.info(f"开始训练 {league_type.value} 模型...")

 # 创建相应的模型
 if league_type in TOURNAMENT_LEAGUE_TYPES:
 model = TournamentModelFactory.create_tournament_model(league_type)
 else:
 model = LeagueModelFactory.create_league_model(league_type)

 # 训练模型
 training_results = model.train(data, target_column)

 # 交叉验证
 if cross_validation:
 cv_results = self._perform_cross_validation(
 model, data, target_column, cv_folds, league_type
 )
 training_results["cross_validation"] = cv_results

 # 评估模型
 evaluation_results = self._evaluate_model(
 model, data, target_column, league_type
 )
 training_results["evaluation"] = evaluation_results

 # 保存训练历史
 self.training_history.append(
 {
 "timestamp": datetime.now(tz=timezone.utc),
 "league_type": league_type.value,
 "data_shape": data.shape,
 "training_results": training_results,
 "evaluation_results": evaluation_results,
 }
 )

 self.model = model

 logger.info(
 f"模型训练完成，验证损失: {evaluation_results.get('poisson_loss', 'N/A')}"
 )

 return training_results

 def _perform_cross_validation(
 self,
 model,
 data: pd.DataFrame,
 target_column: str,
 cv_folds: int,
 league_type: LeagueType,
 ) -> dict[str, Any]:
 """
 执行交叉验证

 Args:
 model: 模型
 data: 数据
 target_column: 目标变量列名
 cv_folds: 交叉验证折数
 league_type: 联赛类型
 """
 logger.info(f"开始 {cv_folds} 折交叉验证...")

 # 准备数据
 prepared_data = (
 model.prepare_league_specific_features(data)
 if hasattr(model, "prepare_league_specific_features")
 else (
 model.prepare_tournament_specific_features(data)
 if hasattr(model, "prepare_tournament_specific_features")
 else data
 )
 )

 # 统一用数值列白名单选择特征(与 _evaluate_model / train() 一致,防比分列泄漏)
 if hasattr(model, "_select_feature_columns"):
 feature_columns = model._select_feature_columns(
 prepared_data, target_column
 )
 else:
 feature_columns = [
 c
 for c in prepared_data.columns
 if c != target_column
 and c not in ["date", "league", "season"]
 and pd.api.types.is_numeric_dtype(prepared_data[c])
 ]

 # 只保留特征列 + date(时间排序用) + 目标列
 keep = [
 c
 for c in prepared_data.columns
 if c in feature_columns or c in ("date", target_column)
 ]
 prepared_data = prepared_data[keep]

 X = prepared_data[feature_columns]
 y = prepared_data[target_column]

 # 按时间排序,确保时间序列切分顺序正确(防泄漏)
 if "date" in prepared_data.columns:
 prepared_data = prepared_data.sort_values("date").reset_index(drop=True)
 X = prepared_data[feature_columns]
 y = prepared_data[target_column]

 # 使用时间序列交叉验证(
 from app.models.utils import date_group_folds

 cv_folds_list = date_group_folds(prepared_data, n_splits=cv_folds)

 cv_scores = []
 cv_mse = []
 cv_mae = []

 for _fold, (train_idx, test_idx) in enumerate(cv_folds_list):
 X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
 y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

 # 折内问题列剔除:常量列 + 非空率过低的列
 # (sklearn 分箱器对"非空率低+大量NaN"的列有 bug,如早期折内 xG 特征仅 ~13% 非空)
 keep = [
 col
 for col in X_train.columns
 if X_train[col].notna().mean() > 0.2
 and X_train[col].nunique(dropna=True) > 1
 ]
 X_train, X_test = X_train[keep], X_test[keep]

 # 训练模型(使用与最终模型相同的联赛配置)
 fold_model = PoissonLossHGBR(**model.config.parameters)
 fold_model.fit(X_train, y_train)

 # 预测和评估
 y_pred = fold_model.predict(X_test)
 mse = mean_squared_error(y_test, y_pred)
 mae = mean_absolute_error(y_test, y_pred)
 poisson_loss = fold_model._calculate_poisson_loss(y_test.values, y_pred)

 cv_scores.append(poisson_loss)
 cv_mse.append(mse)
 cv_mae.append(mae)

 cv_results = {
 "cv_scores": cv_scores,
 "cv_mean": np.mean(cv_scores),
 "cv_std": np.std(cv_scores),
 "cv_mse_mean": np.mean(cv_mse),
 "cv_mae_mean": np.mean(cv_mae),
 "cv_folds": cv_folds,
 }

 self.cross_validation_results[league_type.value] = cv_results

 logger.info(f"交叉验证完成，平均Poisson损失: {cv_results['cv_mean']:.4f}")

 return cv_results

 def _evaluate_model(
 self, model, data: pd.DataFrame, target_column: str, league_type: LeagueType
 ) -> dict[str, Any]:
 """
 评估模型性能

 Args:
 model: 模型
 data: 数据
 target_column: 目标变量列名
 league_type: 联赛类型
 """
 logger.info("开始评估模型性能...")

 # 时间外评估数据(排序 + 后 20% 切分,与 train() 的 holdout 口径一致)
 _, _, X, y, feature_columns = _prepare_eval_split(model, data, target_column)

 # 预测(直接使用底层模型,避免包装类 predict 的二次特征准备)
 if hasattr(model, "model") and model.model is not None:
 predictions = model.model.predict(X)
 else:
 predictions = model.predict(X)

 # 统一指标计算
 metrics = compute_evaluation_metrics(y, predictions)

 # 获取特征重要性
 feature_importance = {}
 if hasattr(model, "model") and model.model is not None:
 fi = getattr(model.model, "feature_importance_", None)
 if fi is not None and len(fi) > 0:
 feature_importance = (
 fi.to_dict()
 if hasattr(fi, "to_dict")
 else dict(zip(feature_columns, fi))
 )

 evaluation_results = {
 "mse": metrics["mse"],
 "mae": metrics["mae"],
 "rmse": metrics["rmse"],
 "poisson_loss": metrics["poisson_loss"],
 "accuracy_metrics": {
 k: metrics[k]
 for k in (
 "exact_accuracy",
 "within_one_accuracy",
 "within_two_accuracy",
 )
 },
 "feature_importance": feature_importance,
 "data_shape": data.shape,
 "feature_count": len(feature_columns),
 }

 self.model_metrics[league_type.value] = evaluation_results

 logger.info(f"模型评估完成，Poisson损失: {metrics['poisson_loss']:.4f}")

 return evaluation_results

 def save_model(self, filepath: str, league_type: LeagueType) -> None:
 """
 保存模型

 Args:
 filepath: 模型保存路径
 league_type: 联赛类型
 """
 if self.model is None:
 raise ValueError("没有训练好的模型可以保存")

 # 创建保存目录
 os.makedirs(os.path.dirname(filepath), exist_ok=True)

 # 保存模型
 self.model.save_model(filepath)

 # 保存训练历史(使用扩展名替换,避免覆盖模型文件本身)
 root, _ext = os.path.splitext(filepath)
 history_file = f"{root}_history.json"
 with open(history_file, "w", encoding="utf-8") as f:
 json.dump(self.training_history, f, indent=2, default=str)

 logger.info(f"模型已保存到: {filepath}")

 def load_model(self, filepath: str, league_type: LeagueType) -> None:
 """
 加载模型

 Args:
 filepath: 模型文件路径
 league_type: 联赛类型
 """
 if not os.path.exists(filepath):
 raise FileNotFoundError(f"模型文件不存在: {filepath}")

 # 创建相应的模型
 if league_type in TOURNAMENT_LEAGUE_TYPES:
 self.model = TournamentModelFactory.create_tournament_model(league_type)
 else:
 self.model = LeagueModelFactory.create_league_model(league_type)

 # 加载模型
 self.model.load_model(filepath, league_type)

 # 加载训练历史
 root, _ext = os.path.splitext(filepath)
 history_file = f"{root}_history.json"
 if os.path.exists(history_file):
 with open(history_file, encoding="utf-8") as f:
 self.training_history = json.load(f)

 logger.info(f"模型已从 {filepath} 加载")
