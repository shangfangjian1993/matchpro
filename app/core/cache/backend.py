"""缓存后端抽象(

CacheBackend 统一接口:LocalBackend(默认,线程安全内存)与
RedisBackend(跨进程共享,生产)。后端由环境变量选择:

 CACHE_BACKEND=local|redis
 REDIS_URL=redis://host:6379/0(默认 redis://localhost:6379/0)

序列化统一 pickle(键/值);Redis 值带 TTL。
"""

from __future__ import annotations

import os
import pickle
import threading
import time


class CacheBackend:
 """缓存后端基类接口。"""

 def get(self, key) -> object | None:
 raise NotImplementedError

 def put(self, key, value, ttl: int | None = None) -> None:
 raise NotImplementedError

 def clear(self) -> None:
 raise NotImplementedError


class LocalBackend(CacheBackend):
 """线程安全内存后端(单个进程内共享)。"""

 def __init__(self, max_size: int = 256):
 self.max_size = max_size
 self._data: dict = {}
 self._ts: dict = {}
 self._lock = threading.Lock()

 def get(self, key):
 with self._lock:
 item = self._data.get(key)
 if item is None:
 return None
 value, expires = item
 if expires is not None and time.time() > expires:
 del self._data[key]
 del self._ts[key]
 return None
 return value

 def put(self, key, value, ttl: int | None = None) -> None:
 with self._lock:
 if len(self._data) >= self.max_size and key not in self._data:
 self._data.pop(next(iter(self._data)))
 self._data[key] = (value, (time.time() + ttl) if ttl else None)
 self._ts[key] = time.time()

 def clear(self) -> None:
 with self._lock:
 self._data.clear()
 self._ts.clear()


class RedisBackend(CacheBackend):
 """Redis 后端(跨进程/多 worker 共享;生产推荐)。"""

 def __init__(self, url: str | None = None, prefix: str = "matchpro:cache:"):
 import redis

 self._r = redis.Redis.from_url(
 url or os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
 decode_responses=False,
 )
 self._prefix = prefix
 try:
 self._r.ping()
 except Exception:
 import logging

 logging.getLogger(__name__).warning(
 "Redis 不可达(%s),缓存后端降级为本地内存", url
 )

 def _k(self, key) -> str:
 if isinstance(key, tuple):
 key = "|".join(str(x) for x in key)
 return self._prefix + str(key)

 def get(self, key):
 try:
 raw = self._r.get(self._k(key))
 return pickle.loads(raw) if raw is not None else None
 except Exception:
 return None

 def put(self, key, value, ttl: int | None = None) -> None:
 try:
 self._r.set(self._k(key), pickle.dumps(value), ex=ttl)
 except Exception:
 pass

 def clear(self) -> None:
 try:
 cursor = 0
 while True:
 cursor, keys = self._r.scan(cursor, match=self._prefix + "*")
 if keys:
 self._r.delete(*keys)
 if cursor == 0:
 break
 except Exception:
 pass


def get_backend(kind: str | None = None, max_size: int = 256) -> CacheBackend:
 """按环境变量选择后端(CACHE_BACKEND=local|redis)。"""
 kind = kind or os.environ.get("CACHE_BACKEND", "local")
 if kind == "redis":
 return RedisBackend()
 return LocalBackend(max_size=max_size)
