"""源公共 HTTP:gzip 感知的下载 + JSON 请求(源适配器共用)。"""

import gzip
import json
import os
import urllib.request

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (football-prediction-pipeline)"


def http_get(url: str, headers: dict | None = None, timeout: int = 40) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    resp = urllib.request.urlopen(req, timeout=timeout)
    raw = resp.read()
    if resp.headers.get("Content-Encoding") == "gzip" or raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw


def http_get_json(url: str, headers: dict | None = None, timeout: int = 40) -> dict:
    return json.loads(http_get(url, headers, timeout))


def default_cache_dir() -> str:
    """§2.2 数据分层 raw/:原始数据按源分目录,永不修改(JSON 缓存即 raw 层)。"""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "raw")


class JsonCache:
    """文件缓存(JSON 序列化,TTL 内命中,超时重取)——raw 层落盘实现。"""

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = cache_dir or default_cache_dir()
        os.makedirs(self.cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        import re

        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
        return os.path.join(self.cache_dir, f"{safe}.json")

    def get(self, key: str, max_age_hours: float):
        import time

        path = self._path(key)
        if not os.path.exists(path):
            return None
        if time.time() - os.path.getmtime(path) > max_age_hours * 3600:
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def set(self, key: str, value) -> None:
        with open(self._path(key), "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, default=str)
