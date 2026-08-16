"""多联赛足球预测模型系统配置文件
"""

import os
from dataclasses import dataclass
from enum import Enum


class LeagueType(Enum):
    """联赛类型枚举"""
    PREMIER_LEAGUE = "premier_league"
    LA_LIGA = "la_liga"
    BUNDESLIGA = "bundesliga"
    LIGUE_1 = "ligue_1"
    SERIE_A = "serie_a"
    CHINESE_SUPER = "chinese_super"
    WORLD_CUP = "world_cup"
    EUROPEAN_CHAMPIONSHIP = "european_championship"
    CHAMPIONS_LEAGUE = "champions_league"
    EUROPA_LEAGUE = "europa_league"

@dataclass
class ModelConfig:
    """模型配置

    parameters 的键需与 sklearn HistGradientBoostingRegressor 构造参数对齐
    (early_stopping_rounds 由 hgbr_model.py 映射为 n_iter_no_change)。
    实际特征列由训练时确定并随模型持久化(feature_columns_),
    此处不再声明 features 列表,避免与实现脱节。
    """
    model_type: str
    league_type: LeagueType
    version: str
    parameters: dict

# 常量(TOURNAMENT_LEAGUE_TYPES / MATCH_METRIC_COLUMNS)已移至 constants.py
from app.core.constants import (
    MATCH_METRIC_COLUMNS,
    TOURNAMENT_LEAGUE_TYPES,
)

__all__ = ["MATCH_METRIC_COLUMNS", "TOURNAMENT_LEAGUE_TYPES"]


class MultiLeagueConfig:
    """多联赛配置管理"""

    def __init__(self):
        # 审查 §19:统一路径由 core/paths.py 提供(禁止重复推导)
        from app.core.paths import DATA_DIR, MODELS_DIR, PROJECT_ROOT
        self.base_dir = str(PROJECT_ROOT)
        self.data_dir = str(DATA_DIR)
        self.models_dir = str(MODELS_DIR)

        # 确保目录存在(只读环境下静默跳过,不阻塞导入)
        for d in (self.data_dir, self.models_dir):
            try:
                os.makedirs(d, exist_ok=True)
            except OSError:
                pass

        # 初始化配置
        self._init_configs()


# ---- Match 指标列权威白名单(单一定义源)----
# 三份历史副本(api/blueprints/ingest.METRIC_FIELDS、
# data/data_ingestion/cleanse.METRIC_FIELDS、api/model_service.MATCH_METRIC_COLUMNS)
# 曾各自维护且已失同步(model_service 缺 13 列 → 入库指标不进入训练特征)。
# 现统一引用本常量;新增指标列只需改这一处 + db.Match。
# 赛事型联赛(非联赛循环赛;训练/加载走赛事模型工厂)
    def _load_yaml_overrides(self):
        """configs/league.yaml 覆盖(§1.1 YAML 配置化;文件缺失/损坏时用 Python 默认)。"""
        import yaml as _yaml
        # 审查 P1-4:模型参数单一配置源 = configs/models.yaml(leagues 段)
        path = os.path.join(self.base_dir, "configs", "models.yaml")
        try:
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    return _yaml.safe_load(f) or {}
        except Exception:
            pass
        return {}

    def _init_configs(self):
        """初始化各联赛配置"""

        _yaml_cfg = self._load_yaml_overrides()

        # 基础模型配置(通用模型/中超兜底)
        self.base_model_config = ModelConfig(
            model_type="HGBR",
            league_type=LeagueType.PREMIER_LEAGUE,
            version="1.0.0",
                parameters={
                    "loss": "poisson",
                    "learning_rate": 0.06,
                    "max_depth": 6,
                    "min_samples_leaf": 10,
                    "max_iter": 300,
                    "early_stopping_rounds": 25,
                    "validation_fraction": 0.15,
                    "random_state": 42
                }
            )
        _yb = (_yaml_cfg.get("base_model_config") or {})
        if _yb:
            self.base_model_config.model_type = _yb.get("model_type", self.base_model_config.model_type)
            self.base_model_config.version = _yb.get("version", self.base_model_config.version)
            if _yb.get("parameters"):
                self.base_model_config.parameters.update(_yb["parameters"])

        # 联赛特定配置
        self.league_configs = {
            LeagueType.PREMIER_LEAGUE: ModelConfig(
                model_type="HGBR",
                league_type=LeagueType.PREMIER_LEAGUE,
                version="1.0.0",
                parameters={
                    "loss": "poisson",
                    "learning_rate": 0.06,
                    "max_depth": 7,
                    "min_samples_leaf": 8,
                    "max_iter": 300,
                    "early_stopping_rounds": 25,
                    "validation_fraction": 0.15,
                    "random_state": 42
                }
            ),
            LeagueType.LA_LIGA: ModelConfig(
                model_type="HGBR",
                league_type=LeagueType.LA_LIGA,
                version="1.0.0",
                parameters={
                    "loss": "poisson",
                    "learning_rate": 0.06,
                    "max_depth": 6,
                    "min_samples_leaf": 9,
                    "max_iter": 300,
                    "early_stopping_rounds": 25,
                    "validation_fraction": 0.15,
                    "random_state": 42
                }
            ),
            LeagueType.BUNDESLIGA: ModelConfig(
                model_type="HGBR",
                league_type=LeagueType.BUNDESLIGA,
                version="1.0.0",
                parameters={
                    "loss": "poisson",
                    "learning_rate": 0.06,
                    "max_depth": 8,
                    "min_samples_leaf": 6,
                    "max_iter": 300,
                    "early_stopping_rounds": 25,
                    "validation_fraction": 0.15,
                    "random_state": 42
                }
            ),
            LeagueType.LIGUE_1: ModelConfig(
                model_type="HGBR",
                league_type=LeagueType.LIGUE_1,
                version="1.0.0",
                parameters={
                    "loss": "poisson",
                    "learning_rate": 0.06,
                    "max_depth": 6,
                    "min_samples_leaf": 10,
                    "max_iter": 300,
                    "early_stopping_rounds": 25,
                    "validation_fraction": 0.15,
                    "random_state": 42
                }
            ),
            LeagueType.SERIE_A: ModelConfig(
                model_type="HGBR",
                league_type=LeagueType.SERIE_A,
                version="1.0.0",
                parameters={
                    "loss": "poisson",
                    "learning_rate": 0.06,
                    "max_depth": 7,
                    "min_samples_leaf": 8,
                    "max_iter": 300,
                    "early_stopping_rounds": 25,
                    "validation_fraction": 0.15,
                    "random_state": 42
                }
            )
        }

        # 赛事模型配置
        self.tournament_configs = {
            LeagueType.WORLD_CUP: ModelConfig(
                model_type="HGBR",
                league_type=LeagueType.WORLD_CUP,
                version="1.0.0",
                parameters={
                    "loss": "poisson",
                    "learning_rate": 0.06,
                    "max_depth": 8,
                    "min_samples_leaf": 5,
                    "max_iter": 300,
                    "early_stopping_rounds": 25,
                    "validation_fraction": 0.15,
                    "random_state": 42
                }
            ),
            LeagueType.EUROPEAN_CHAMPIONSHIP: ModelConfig(
                model_type="HGBR",
                league_type=LeagueType.EUROPEAN_CHAMPIONSHIP,
                version="1.0.0",
                parameters={
                    "loss": "poisson",
                    "learning_rate": 0.06,
                    "max_depth": 7,
                    "min_samples_leaf": 6,
                    "max_iter": 300,
                    "early_stopping_rounds": 25,
                    "validation_fraction": 0.15,
                    "random_state": 42
                }
            ),
            LeagueType.CHAMPIONS_LEAGUE: ModelConfig(
                model_type="HGBR",
                league_type=LeagueType.CHAMPIONS_LEAGUE,
                version="1.0.0",
                parameters={
                    "loss": "poisson",
                    "learning_rate": 0.06,
                    "max_depth": 7,
                    "min_samples_leaf": 7,
                    "max_iter": 300,
                    "early_stopping_rounds": 25,
                    "validation_fraction": 0.15,
                    "random_state": 42
                }
            ),
            LeagueType.EUROPA_LEAGUE: ModelConfig(
                model_type="HGBR",
                league_type=LeagueType.EUROPA_LEAGUE,
                version="1.0.0",
                parameters={
                    "loss": "poisson",
                    "learning_rate": 0.06,
                    "max_depth": 6,
                    "min_samples_leaf": 9,
                    "max_iter": 300,
                    "early_stopping_rounds": 25,
                    "validation_fraction": 0.15,
                    "random_state": 42
                }
            )
        }

        # league.yaml 覆盖各联赛参数(存在即生效;YAML 为单一事实源时删除 Python 默认)
        _yl = _yaml_cfg.get("leagues") or {}
        for _lt_value, _mc in self.league_configs.items():
            _yc = _yl.get(_lt_value.value) or {}
            if not _yc:
                continue
            _mc.model_type = _yc.get("model_type", _mc.model_type)
            _mc.version = _yc.get("version", _mc.version)
            if _yc.get("parameters"):
                _mc.parameters.update(_yc["parameters"])


    def get_model_config(self, league_type: LeagueType) -> ModelConfig:
        """获取指定联赛的模型配置"""
        if league_type in self.league_configs:
            return self.league_configs[league_type]
        elif league_type in self.tournament_configs:
            return self.tournament_configs[league_type]
        elif league_type == LeagueType.CHINESE_SUPER:
            # 中超:使用基础配置(通用模型)
            return self.base_model_config
        else:
            raise ValueError(f"不支持的联赛类型: {league_type}")

# 全局配置实例
config = MultiLeagueConfig()

def load_yaml(name: str) -> dict:
    """加载 configs/<name>.yaml(§1.1);缺失/损坏返回 {}。"""
    import yaml
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "configs", name)
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}
