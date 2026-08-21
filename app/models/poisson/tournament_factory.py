"""赛事模型系统

共享实现(滚动特征工程/训练/预测/保存加载)在 base_models.base_football_model.BaseFootballModel。
所有赛事(世界杯/欧洲杯/欧冠/欧联)共用同一 TournamentModel —— 差异只来自各自的
ModelConfig(参数),不再需要逐字相同的子类。
"""

import pandas as pd

from app.core.config import LeagueType, ModelConfig, MultiLeagueConfig
from app.models.poisson.base_football_model import BaseFootballModel


class TournamentModel(BaseFootballModel):
    """
    赛事模型:核心滚动统计 + 赛事阶段编码。

    世界杯/欧洲杯/欧冠/欧联等赛事通过不同 config 区分,不设子类。
    """

    min_training_rows = 50

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.tournament_stage_mapping = {
            "group": 0,
            "round_of_16": 1,
            "quarter_final": 2,
            "semi_final": 3,
            "final": 4,
        }

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """核心滚动统计 + 赛事阶段编码"""
        from app.features.factory import compute_all

        prepared = compute_all(
            data,
            league_type=self.config.league_type.value,
            metric_columns=tuple(self.metric_columns),
            side_metric_columns=tuple(self.side_metric_columns),
        )
        prepared["tournament_stage"] = self._encode_tournament_stage(data)
        return prepared

    def _encode_tournament_stage(self, data: pd.DataFrame) -> pd.Series:
        """
        编码赛事阶段(支持整列 Series 输入,未知阶段编码为 0)
        """
        if isinstance(data, pd.Series):
            return data.map(self.tournament_stage_mapping).fillna(0).astype(int)
        if "stage" in data.columns:
            return (
                data["stage"].map(self.tournament_stage_mapping).fillna(0).astype(int)
            )
        return pd.Series(0, index=data.index)

    def save_model(self, filepath: str) -> None:
        """保存模型,附加赛事阶段映射"""
        super().save_model(
            filepath, extra={"tournament_stage_mapping": self.tournament_stage_mapping}
        )

    def load_model(self, filepath: str, league_type: LeagueType | None = None) -> None:
        """加载模型,恢复赛事阶段映射"""
        save_data = super().load_model(filepath, league_type)
        if "tournament_stage_mapping" in save_data:
            self.tournament_stage_mapping = save_data["tournament_stage_mapping"]


class TournamentModelFactory:
    """
    赛事模型工厂
    """

    @staticmethod
    def create_tournament_model(
        league_type: LeagueType, config: ModelConfig | None = None
    ) -> TournamentModel:
        """
        创建赛事模型

        Args:
            league_type: 赛事类型
            config: 模型配置，如果为None则使用默认配置

        Returns:
            赛事模型实例
        """
        if config is None:
            multi_config = MultiLeagueConfig()
            config = multi_config.get_model_config(league_type)

        return TournamentModel(config)
