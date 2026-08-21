"""模型加载器(

职责:模型/全球模型加载 + 进程内缓存(ArtifactCache);依赖 registry 解析路径。
"""

from __future__ import annotations

import logging
import os

from app.core.cache import ArtifactCache
from app.core.config import TOURNAMENT_LEAGUE_TYPES as TOURNAMENT_TYPES
from app.core.config import LeagueType
from app.models.integrity import _verify_model_integrity
from app.models.poisson.league_factory import LeagueModelFactory
from app.models.poisson.tournament_factory import TournamentModelFactory
from app.models.registry import (
 _active_version,
 _existing_versions,
 _is_pointer_file,
 _model_path,
)

logger = logging.getLogger(__name__)

_MODEL_CACHE = ArtifactCache(16)


def _load_model(league_type: LeagueType, models_dir: str, version: str = "latest"):
 """从磁盘加载已训练模型(进程内缓存,按文件 mtime 失效);
 version=latest 时优先 active_models.json 指向的最优版本,无则用指针文件"""
 if version in ("latest", ""):
 version = _active_version(league_type, models_dir) or "latest"
 path = _model_path(
 league_type, models_dir, None if version == "latest" else version
 )
 if not os.path.exists(path):
 if version not in ("latest", ""):
 logger.warning(
 "%s 版本 %s 不存在(可能被 prune),回退到文件系统最新版本",
 league_type.value,
 version,
 )
 version = "latest"
 path = _model_path(league_type, models_dir)
 if not os.path.exists(path) or _is_pointer_file(league_type, models_dir, path):
 _vs = _existing_versions(league_type, models_dir)
 if not _vs:
 raise ValueError(f"{league_type.value} 模型尚未训练,请先训练模型")
 path = _model_path(league_type, models_dir, _vs[-1])
 if not os.path.exists(path):
 raise ValueError(f"{league_type.value} 模型尚未训练,请先训练模型")
 cached = _MODEL_CACHE.get(path)
 if cached is not None:
 return cached
 _verify_model_integrity(path)
 if league_type in TOURNAMENT_TYPES:
 model = TournamentModelFactory.create_tournament_model(league_type)
 else:
 model = LeagueModelFactory.create_league_model(league_type)
 model.load_model(path)
 _MODEL_CACHE.put(path, model)
 return model
