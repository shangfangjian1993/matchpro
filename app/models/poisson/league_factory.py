"""联赛特定模型系统

共享实现(滚动特征工程/训练/预测/保存加载)在 base_models.base_football_model.BaseFootballModel,
这里只保留联赛差异:各联赛的额外统计特征列。
"""

import pandas as pd

from app.core.config import LeagueType, ModelConfig, MultiLeagueConfig
from app.models.poisson.base_football_model import BaseFootballModel


class LeagueModel(BaseFootballModel):
    """
    联赛模型基类(共享实现见 BaseFootballModel)。

    结构统一:所有联赛使用同一套指标列(matches 表),联赛差异只在于
    metric_columns 配置——声明要"引用"的统一指标列;未引用的联赛
    无需填数据(列保持 NULL,特征选择自动跳过)。新增联赛 = 声明配置,
    无需新列。
    """

    min_training_rows = 100

    # 要引用的统一指标列(单列机制,数据中不存在时自动跳过;当前数据为主客分列,见 side_metric_columns)
    metric_columns: list[str] | None = None

    # 主客分列指标(home_xg/away_xg 形式)滚动特征:数据存在即启用
    # possession 已预留:api-football 付费套餐或未来数据源提供控球率后自动生效
    side_metric_columns: tuple[str, ...] = (
        "xg",
        "shots",
        "shots_on_target",
        "corners",
        "possession",
    )

    def prepare_league_specific_features(
        self, data: pd.DataFrame, hist_matches=None
    ) -> pd.DataFrame:
        """统一走 Feature Factory(审查 §18:ELO 注入亦在 Factory 内)。

        hist_matches:与 data 行对齐的 Match ORM 列表(C 阶段:Stats 特征族
        需要逐场 match_id/队名查 team_match_stats;None=旧行为无 stats)。
        """
        from app.features.factory import compute_all

        return compute_all(
            data,
            league_type=self.config.league_type.value,
            metric_columns=tuple(self.metric_columns or []),
            side_metric_columns=tuple(self.side_metric_columns or []),
            hist_matches=hist_matches,
        )

    def prepare_features(self, data: pd.DataFrame, hist_matches=None) -> pd.DataFrame:
        """基类入口:转发到联赛特定特征准备(逐场 Match ORM 供 stats 特征)"""
        return self.prepare_league_specific_features(data, hist_matches=hist_matches)


class PremierLeagueModel(LeagueModel):
    """英超联赛模型:引用 角球/射正"""


class LaLigaModel(LeagueModel):
    """西甲联赛模型:引用 传球成功率/进攻链"""


class BundesligaModel(LeagueModel):
    """德甲联赛模型:引用 效率/转换速度"""


class Ligue1Model(LeagueModel):
    """法甲联赛模型:引用 防守动作/反击"""


class SerieAModel(LeagueModel):
    """意甲联赛模型:引用 战术评分/经验(数据源有则填,无则跳过)"""


class GenericLeagueModel(LeagueModel):
    """
    通用联赛模型:未配置专门实现的联赛使用,只使用核心滚动统计特征
    (metric_columns 为空,不引用任何指标列)。
    """


# 全球分层第一层:合并全部联赛训练,league_code 编码捕获联赛水平差异。
# 用途:①无联赛历史球队的兜底(该队在其他联赛有记录);
#      ②跨联赛可比特征(全局 ELO)的自然载体。
_LEAGUE_CODE = {
    "premier_league": 1,
    "la_liga": 2,
    "bundesliga": 3,
    "serie_a": 4,
    "ligue_1": 5,
    "champions_league": 6,
    "world_cup": 7,
    "european_championship": 8,
}


class LeagueModelFactory:
    """
    联赛模型工厂
    """

    @staticmethod
    def create_league_model(
        league_type: LeagueType, config: ModelConfig | None = None
    ) -> LeagueModel:
        """
        创建联赛模型

        Args:
            league_type: 联赛类型
            config: 模型配置，如果为None则使用默认配置

        Returns:
            联赛模型实例
        """
        if config is None:
            multi_config = MultiLeagueConfig()
            config = multi_config.get_model_config(league_type)

        # 根据联赛类型创建相应的模型
        if league_type == LeagueType.PREMIER_LEAGUE:
            return PremierLeagueModel(config)
        elif league_type == LeagueType.LA_LIGA:
            return LaLigaModel(config)
        elif league_type == LeagueType.BUNDESLIGA:
            return BundesligaModel(config)
        elif league_type == LeagueType.LIGUE_1:
            return Ligue1Model(config)
        elif league_type == LeagueType.SERIE_A:
            return SerieAModel(config)
        else:
            # 其他联赛使用通用模型(不再实例化抽象类)
            return GenericLeagueModel(config)
