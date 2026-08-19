"""足球预测模型共享基类

联赛模型(league_models)与赛事模型(tournament_models)的公共实现:
滚动特征工程、特征白名单选择、训练/预测/保存/加载、数据质量校验。

设计要点:
- 特征只使用本场比赛之前的历史(expanding/rolling + shift(1)),严格防泄漏;
- 训练时把实际使用的特征列存入 pkl(feature_columns_),预测时按该白名单取列,
  避免"数值列-排除列"推断在新增比分列时静默泄漏;
- train() 内部强制按 date 排序,时间 holdout 切分不受调用方行序影响。
"""

import os
from abc import ABC, abstractmethod
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.core.config import LeagueType, ModelConfig


def _feature_flag(name: str, default: bool = True) -> bool:
    """读取 configs/leagues.yaml 的 feature_flags(§3 可选关闭)。"""
    try:
        import os as _os

        import yaml as _yaml

        # app/models/poisson/base_football_model.py → 上 4 级 = 项目根
        _root = _os.path.dirname(
            _os.path.dirname(
                _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            )
        )
        _path = _os.path.join(_root, "configs", "models.yaml")
        if _os.path.exists(_path):
            with open(_path, encoding="utf-8") as _f:
                _cfg = _yaml.safe_load(_f) or {}
            return bool((_cfg.get("features") or {}).get(name, default))
    except Exception:
        pass
    return default


# 特征选择时排除的列(非特征列 + 结果/比分列,防止数据泄漏)
EXCLUDE_COLUMNS = {
    "date",
    "match_id",
    "league",
    "season",
    "stage",
    "round",
    "id",
    "home_team",
    "away_team",
    "score",
    "result",
    "goals",
    "home_goals",
    "away_goals",
}


class BaseFootballModel(ABC):
    """所有足球预测模型的共享基类(联赛/赛事)"""

    # 训练数据量下限(联赛 100,赛事 50,由子类调整)
    min_training_rows: int = 100

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_trained = False
        self.training_history = []
        self.performance_metrics = {}
        # 训练时实际使用的特征列(持久化,预测时作为白名单)
        self.feature_columns_ = None
        # (欧战事件表接口已移除:该特征从未落地,避免误导维护者)

    @abstractmethod
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """子类实现:在核心滚动特征之上添加联赛/赛事特定特征"""

    # ------------------------------------------------------------------
    # 特征工程助手:基于历史数据的滚动统计(严格只使用本场之前的数据,防泄漏)
    # ------------------------------------------------------------------

    @staticmethod
    def _hist_matches_from_df(data: pd.DataFrame):
        """C 阶段:从 data.match_id 反查 Match ORM(行序对齐,Stats 特征族用)。

        无 match_id 列/无有效 id → None(旧行为,不引入 stats)。
        """
        if "match_id" not in data.columns:
            return None
        ids = [
            int(x)
            for x in data["match_id"].tolist()
            if x is not None and not pd.isna(x)
        ]
        if not ids:
            return None
        from app.api.db import Match

        _orm = {r.id: r for r in Match.query.filter(Match.id.in_(ids)).all()}
        return [
            _orm.get(int(x)) if (x is not None and not pd.isna(x)) else None
            for x in data["match_id"].tolist()
        ]

    @staticmethod
    def _sort_by_date(data: pd.DataFrame) -> pd.DataFrame:
        """按 date 排序(mergesort 稳定),无 date 或全 NaN 时保持原序。"""
        if "date" in data.columns and not data["date"].isna().all():
            df = data.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df.sort_values("date", kind="mergesort").reset_index(drop=True)
        return data

    def _select_feature_columns(
        self, df: pd.DataFrame, target_column: str
    ) -> list[str]:
        """数值列白名单特征选择,排除目标列与 ID/文本/比分列,防泄漏。

        全空列自动剔除:摄入层预留的指标列(如 xg/corners)在无数据时保持 NULL,
        直接选中会导致训练缺失率检查失败;有数据后自动纳入,无需改代码。
        """
        exclude = EXCLUDE_COLUMNS | {target_column}

        def _is_raw_side_col(c: str) -> bool:
            # 当场统计源列(home_xg/away_shots...)赛后可知,必须排除;
            # 滚动特征列 home_team_*/away_team_* 与 stats 滚动
            # home_tms_*/away_tms_*(赛前可得的球队历史统计)不排除。
            return (c.startswith(("home_", "away_"))) and not (
                c.startswith(("home_team_", "away_team_", "home_tms_", "away_tms_"))
            )

        return [
            c
            for c in df.columns
            if c not in exclude
            and not _is_raw_side_col(c)
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].notna().any()
            and df[c].nunique(dropna=True)
            > 1  # 剔除常量列(无信息量,且 sklearn 分箱器对常量列报错)
        ]

    # ------------------------------------------------------------------
    # 训练 / 预测 / 保存 / 加载
    # ------------------------------------------------------------------

    def train(self, data: pd.DataFrame, target_column: str = "goals") -> dict[str, Any]:
        """训练模型(内部强制按 date 排序,时间 holdout 评估,保存特征白名单)"""
        from app.models.poisson.hgbr_model import PoissonLossHGBR

        logger = self._logger()

        logger.info(f"开始训练 {self.config.league_type.value} 模型...")

        # 按时间排序,保证 holdout 切分是时间序(防泄漏)
        data = self._sort_by_date(data)

        # 准备数据(C 阶段:hist_matches 由 match_id 反查,Stats 特征族进入训练)
        prepared_data = self.prepare_features(
            data, hist_matches=self._hist_matches_from_df(data)
        )

        # 选择特征
        feature_columns = self._select_feature_columns(prepared_data, target_column)

        if not feature_columns:
            raise ValueError("没有可用的数值特征列,请检查输入数据")

        # 检查数据质量
        self._validate_data_quality(
            prepared_data[feature_columns], prepared_data[target_column]
        )

        # 初始化模型
        self.model = PoissonLossHGBR(**self.config.parameters)

        # 时间序列 holdout:前 80% 训练,后 20% 评估(防止样本内评估虚高)
        # 审查 P1-7:按日期分组切分 —— 同一比赛日不会同时出现在 train/test
        from app.models.utils import date_group_split

        _trn, _eva = date_group_split(prepared_data, ratio=0.8)
        X_train = _trn[feature_columns]
        y_train = _trn[target_column]
        X_eval = _eva[feature_columns]
        y_eval = _eva[target_column]

        if len(X_eval) == 0:
            X_eval, y_eval = X_train, y_train  # 数据太少时退回样本内评估

        # 训练子集问题列剔除:常量列 + 非空率过低的列(sklearn 分箱器对高缺失列有 bug)
        keep_cols = [
            col
            for col in feature_columns
            if X_train[col].notna().mean() > 0.2
            and X_train[col].nunique(dropna=True) > 1
        ]
        X_train = X_train[keep_cols]
        X_eval = X_eval[keep_cols]
        feature_columns = keep_cols

        # 时间衰减样本权重(评审 P1):指数衰减 + 下限保护 + 均值归一,
        # 配置在 configs/models.yaml training.sample_weight(enabled/half_life_days/min_weight)。
        sample_weight = self._sample_weights(_trn)

        # 训练模型(前 80% —— 时间安全 holdout)
        self.model.fit(X_train, y_train, sample_weight=sample_weight)

        # 记录训练时实际使用的特征列(持久化,预测白名单)
        self.feature_columns_ = list(feature_columns)

        # 评估模型(时间外样本,80% 训练模型口径)
        evaluation = self.model.evaluate(X_eval, y_eval)

        # 审查 P0-2:评估完成后用 100% 数据重训 —— 保存/上线的即生产模型,
        # 而非只吃过前 80% 历史的模型。
        full_weight = self._sample_weights(prepared_data)
        self.model.fit(
            prepared_data[feature_columns],
            prepared_data[target_column],
            sample_weight=full_weight,
        )

        # 记录训练历史
        self.training_history.append(
            {
                "timestamp": pd.Timestamp.now(),
                "data_shape": prepared_data.shape,
                "feature_count": len(feature_columns),
                "evaluation_metrics": evaluation,
            }
        )

        self.is_trained = True

        logger.info(f"模型训练完成,验证损失: {evaluation.get('poisson_loss', 'N/A')}")

        return {
            "model_type": self.config.model_type,
            "league_type": self.config.league_type.value,
            "version": self.config.version,
            "training_metrics": evaluation,
            "feature_importance": self.model.feature_importance_.to_dict()
            if self.model.feature_importance_ is not None
            else {},
            "training_data_shape": prepared_data.shape,
            "feature_count": len(feature_columns),
        }

    def predict(self, data: pd.DataFrame) -> dict[str, Any]:
        """预测比赛结果(按训练时的特征白名单取列;输入自动按 date 排序)"""
        if not self.is_trained:
            raise ValueError("模型未训练,请先调用train方法")

        # 按时间排序,保证待预测行(通常是最后一行)的滚动特征正确
        data = self._sort_by_date(data)

        # 准备数据(C 阶段:hist_matches 由 match_id 反查,Stats 特征族进入预测)
        prepared_data = self.prepare_features(
            data, hist_matches=self._hist_matches_from_df(data)
        )

        # 特征白名单:优先用训练时保存的列;训练列缺失时回退到白名单推断
        if self.feature_columns_:
            feature_columns = [
                c for c in self.feature_columns_ if c in prepared_data.columns
            ]
        else:
            feature_columns = self._select_feature_columns(prepared_data, "goals")

        if not feature_columns:
            raise ValueError("没有可用的数值特征列")

        # 预测
        predictions = self.model.predict(prepared_data[feature_columns])
        probabilities = self.model.predict_proba(prepared_data[feature_columns])

        # 计算统计信息
        pred_stats = {
            "mean_goals": np.mean(predictions),
            "std_goals": np.std(predictions),
            "min_goals": np.min(predictions),
            "max_goals": np.max(predictions),
            "median_goals": np.median(predictions),
        }

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "prediction_stats": pred_stats,
            "prepared_data_shape": prepared_data.shape,
            "feature_columns": feature_columns,
        }

    def _sample_weights(self, data: pd.DataFrame) -> np.ndarray | None:
        """样本权重(评审 P1):读 configs/models.yaml training.sample_weight 配置。

        enabled=false → 等权(None);reference_date = 训练集最大日期(严格防泄漏)。
        """
        try:
            import os as _os

            import yaml as _yaml

            _root = _os.path.dirname(
                _os.path.dirname(
                    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
                )
            )
            _cfg_path = _os.path.join(_root, "configs", "models.yaml")
            _sw = {}
            if _os.path.exists(_cfg_path):
                with open(_cfg_path, encoding="utf-8") as _f:
                    _sw = (
                        (_yaml.safe_load(_f) or {})
                        .get("training", {})
                        .get("sample_weight", {})
                    )
            if not _sw.get("enabled", False):
                return None
            if "date" not in data.columns or data["date"].isna().all():
                return None
            from app.models.utils import compute_time_decay_weights

            dates = pd.to_datetime(data["date"], errors="coerce")
            valid = dates.notna()
            w = np.full(len(dates), 1.0)
            if valid.any():
                w[valid] = compute_time_decay_weights(
                    dates[valid],
                    half_life_days=float(_sw.get("half_life_days", 365.0)),
                    min_weight=float(_sw.get("min_weight", 0.05)),
                    normalize=bool(_sw.get("normalize", True)),
                    reference_date=dates[valid].max(),  # 训练集内,防泄漏
                )
            return w
        except Exception:
            return None

    def _validate_data_quality(self, X: pd.DataFrame, y: pd.Series) -> None:
        """数据质量检查:样本量、缺失值比例、目标分布"""
        if len(X) < self.min_training_rows:
            raise ValueError(
                f"数据量不足,需要至少{self.min_training_rows}条记录,当前只有{len(X)}条"
            )

        if len(X.columns) > 0:
            missing_ratio = X.isnull().sum().sum() / (len(X) * len(X.columns))
        else:
            missing_ratio = 0.0
        if missing_ratio > 0.3:
            raise ValueError(f"缺失值比例过高: {missing_ratio:.2%}")

        if y.nunique() < 3:
            self._logger().warning("目标变量分布过于简单,可能影响模型性能")

        self._logger().info(
            f"数据质量检查通过,样本数: {len(X)}, 缺失值比例: {missing_ratio:.2%}"
        )

    def save_model(self, filepath: str, extra: dict[str, Any] | None = None) -> None:
        """保存模型与配置(含训练特征白名单;extra 供子类附加字段)"""
        if not self.is_trained:
            raise ValueError("模型未训练,无法保存")

        # 创建保存目录
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        save_data = {
            "model": self.model,
            "config": self.config,
            "training_history": self.training_history,
            "performance_metrics": self.performance_metrics,
            "is_trained": self.is_trained,
            "feature_columns": self.feature_columns_,
        }
        if extra:
            save_data.update(extra)

        joblib.dump(save_data, filepath)
        self._logger().info(f"模型已保存到: {filepath}")

    def load_model(
        self, filepath: str, league_type: LeagueType | None = None
    ) -> dict[str, Any]:
        """加载模型,返回原始 save_data 字典(供子类附加恢复)"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")

        save_data = joblib.load(filepath)

        if not isinstance(save_data, dict) or "model" not in save_data:
            raise ValueError(f"模型文件格式无效: {filepath}")

        # 恢复模型属性
        self.model = save_data["model"]
        if "config" in save_data:
            self.config = save_data["config"]
        self.training_history = save_data.get("training_history", [])
        self.performance_metrics = save_data.get("performance_metrics", {})
        self.is_trained = save_data.get("is_trained", False)
        self.feature_columns_ = save_data.get("feature_columns")

        self._logger().info(f"模型已从 {filepath} 加载")
        return save_data

    def _logger(self):
        import logging

        return logging.getLogger(__name__)
