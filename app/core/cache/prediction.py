"""PredictionCache(审查 §24):预测结果缓存(自 model_service/cache.py 迁移)。

键 = 对阵 + 日期 + 数据版本(hist_max_id/updated)+ 模型 mtime +
      Ensemble 权重/DC-NB/校准器 mtime(审查 P0-3:含 .cal)。
"""
from __future__ import annotations

import os
import threading


class PredictionCache:
    """预测结果缓存:线程安全 + 容量淘汰(dict 插入序)。"""

    def __init__(self, max_size: int | None = None):
        self.max_size = max_size or int(os.environ.get("PREDICT_CACHE_MAX", "256"))
        self._data: dict = {}
        self._lock = threading.Lock()

    def key(self, league_type, home_team, away_team, match_dt,
            hist_max_id, models_dir, hist_max_updated=None):
        from app.models.registry import _model_path
        try:
            mtime = os.path.getmtime(_model_path(league_type, models_dir))
        except OSError:
            mtime = 0.0
        _ens_mtime = 0.0
        from app.core.paths import PROJECT_ROOT
        _ens_dir = str(PROJECT_ROOT / "artifacts" / "ensemble")
        for _fn in (os.path.join(_ens_dir, "ensemble_weights.json"),
                    os.path.join(_ens_dir, "dc_nb_params.json"),
                    os.path.join(models_dir, f"{league_type.value}_model.cal")):
            try:
                _ens_mtime = max(_ens_mtime, os.path.getmtime(_fn))
            except OSError:
                pass
        return (league_type.value, home_team, away_team, str(match_dt),
                hist_max_id, hist_max_updated, mtime, _ens_mtime)

    def get(self, key):
        with self._lock:
            v = self._data.get(key)
            return dict(v) if v is not None else None

    def put(self, key, value) -> None:
        with self._lock:
            if len(self._data) >= self.max_size and key not in self._data:
                self._data.pop(next(iter(self._data)))
            self._data[key] = value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
