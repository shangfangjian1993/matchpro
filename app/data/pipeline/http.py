"""HTTP 客户端(带限速、重试、429 处理)。"""
from __future__ import annotations

import gzip
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from app.data.pipeline.config import HTTP_DEFAULT_TIMEOUT, HTTP_USER_AGENT

logger = logging.getLogger(__name__)


class SeasonNotAvailable(Exception):
    """赛季数据未就绪(HTTP 300/301/302 重定向)。"""
    pass


class RateLimiter:
    """简单限速器。"""

    def __init__(self, interval: float = 1.2) -> None:
        self.interval = interval
        self.last_request = 0.0

    def wait(self) -> None:
        now = time.time()
        elapsed = now - self.last_request
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_request = time.time()


def http_get(
    url: str,
    headers: dict | None = None,
    timeout: int = HTTP_DEFAULT_TIMEOUT,
    rate_limiter: RateLimiter | None = None,
) -> bytes:
    """HTTP GET(带 gzip 解压、重试、429 处理)。"""
    if rate_limiter:
        rate_limiter.wait()

    req_headers = {"User-Agent": HTTP_USER_AGENT, **(headers or {})}
    req = urllib.request.Request(url, headers=req_headers)

    for attempt in range(3):
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip" or raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            return raw
        except urllib.error.HTTPError as e:
            if e.code in (300, 301, 302, 303, 307, 308):
                raise SeasonNotAvailable(f"HTTP {e.code}: season data not available: {url}")
            if e.code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise
        except Exception:
            if attempt < 2:
                time.sleep(1.0)
                continue
            raise
    raise RuntimeError(f"HTTP GET failed after retries: {url}")


def http_get_json(
    url: str,
    headers: dict | None = None,
    timeout: int = HTTP_DEFAULT_TIMEOUT,
    rate_limiter: RateLimiter | None = None,
) -> dict | list:
    """HTTP GET JSON。"""
    raw = http_get(url, headers, timeout, rate_limiter)
    return json.loads(raw)


def build_url(base: str, path: str, params: dict | None = None) -> str:
    """构建 URL。"""
    url = base.rstrip("/") + "/" + path.lstrip("/")
    if params:
        filtered = {k: v for k, v in params.items() if v is not None}
        if filtered:
            url += "?" + urllib.parse.urlencode(filtered)
    return url
