"""ArtifactCache(审查 §24):产物缓存,按 (path, mtime) 校验,容量淘汰,线程安全。"""

from __future__ import annotations

import os
import threading


class ArtifactCache:
    """产物缓存:按 (path, mtime) 校验,容量淘汰,线程安全。"""

    def __init__(self, max_size: int = 16):
        self.max_size = max_size
        self._data: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()

    def get(self, path: str):
        with self._lock:
            item = self._data.get(path)
            if item is None:
                return None
            mtime, obj = item
            try:
                if os.path.getmtime(path) != mtime:
                    self._data.pop(path, None)
                    return None
            except OSError:
                self._data.pop(path, None)
                return None
            return obj

    def put(self, path: str, obj) -> None:
        with self._lock:
            if len(self._data) >= self.max_size:
                self._data.pop(next(iter(self._data)))
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = 0.0
            self._data[path] = (mtime, obj)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
