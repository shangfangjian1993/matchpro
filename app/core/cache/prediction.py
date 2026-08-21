"""PredictionCache(

键 = 对阵 + 日期 + 数据版本(hist_max_id/updated)+ 模型 mtime +
 Ensemble 权重/DC-NB/校准器 mtime(

PostgreSQL+Redis 改造:后端抽象(CACHE_BACKEND=local|redis)。
- local: 单进程线程安全内存(默认,开发/单 worker)
- redis: 跨进程共享(生产多 worker / 多实例),键序列化 pickle + TTL
ArtifactCache(模型对象)保持本地 —— 大对象 + 绑定本地文件 mtime,
跨进程共享收益低且 Redis 承载大对象不经济。
"""

from __future__ import annotations

import os

from app.core.cache.backend import get_backend


class PredictionCache:
 """预测结果缓存:后端可切换(内存/Redis),线程安全。"""

 def __init__(self, max_size: int | None = None):
 self._max = max_size or int(os.environ.get("PREDICT_CACHE_MAX", "256"))
 self._backend = get_backend(os.environ.get("CACHE_BACKEND"), max_size=self._max)
 # 预测结果 TTL(秒);配置 prediction.cache_ttl,默认 600
 try:
 from app.core.config import load_yaml

 self._ttl = int(
 (load_yaml("models.yaml").get("prediction") or {}).get("cache_ttl", 600)
 )
 except Exception:
 self._ttl = 600

 def key(
 self,
 league_type,
 home_team,
 away_team,
 match_dt,
 hist_max_id,
 models_dir,
 hist_max_updated=None,
 ):
 from app.models.registry import _model_path

 try:
 mtime = os.path.getmtime(_model_path(league_type, models_dir))
 except OSError:
 mtime = 0.0
 _ens_mtime = 0.0
 from app.core.paths import ARTIFACTS_DIR as _AD

 _ens_dir = str(_AD / "ensemble")
 _cal_dir = str(_AD / "calibration")
 for _fn in (
 os.path.join(_ens_dir, "ensemble_weights.json"),
 os.path.join(_ens_dir, "dc_nb_params.json"),
 os.path.join(_cal_dir, f"{league_type.value}.cal"),
 ):
 try:
 _ens_mtime = max(_ens_mtime, os.path.getmtime(_fn))
 except OSError:
 pass
 return (
 league_type.value,
 home_team,
 away_team,
 str(match_dt),
 hist_max_id,
 hist_max_updated,
 mtime,
 _ens_mtime,
 )

 def get(self, key):
 v = self._backend.get(key)
 return dict(v) if v is not None else None

 def put(self, key, value) -> None:
 self._backend.put(key, value, ttl=self._ttl)

 def clear(self) -> None:
 self._backend.clear()
