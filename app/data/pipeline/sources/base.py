"""数据源适配器基类(统一接口规范)。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from app.data.pipeline.canonical.normalize import NormalizedMatch


@dataclass
class SourceResult:
    """数据源抓取结果。"""
    source: str  # 来源标识(bzzoiro/fdco/understat/...)
    data_type: str  # 数据类型(results/stats/odds/xg/...)
    records: list[NormalizedMatch] = field(default_factory=list)
    raw_count: int = 0  # 原始记录数
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    errors: list[str] = field(default_factory=list)
    success: bool = True

    @property
    def record_count(self) -> int:
        return len(self.records)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.success = False


class RateLimitHandler(Protocol):
    """限速处理协议。"""
    def wait(self) -> None: ...
    def notify_429(self) -> None: ...


class BaseSource(ABC):
    """数据源适配器基类。

    所有数据源适配器必须实现:
    - fetch(): 抓取原始数据
    - normalize(): 转换为 NormalizedMatch
    - ingest(): 入库(调用 pipeline.ingest)

    子类只需实现 fetch_raw() 和 normalize_row(),基类编排流程。
    """

    SOURCE_NAME: str = ""
    DATA_TYPE: str = ""

    def __init__(
        self,
        league_type: str,
        http_client: Any | None = None,
        cache: Any | None = None,
    ) -> None:
        if not self.SOURCE_NAME:
            raise ValueError(f"{type(self).__name__} 必须定义 SOURCE_NAME")
        self.league_type = league_type
        self.http = http_client
        self.cache = cache
        self._logger = __import__("logging").get_logger(
            f"{type(self).__module__}.{type(self).__name__}"
        )

    @abstractmethod
    def fetch_raw(self, **kwargs: Any) -> list[dict]:
        """抓取原始数据(数据源特定逻辑)。

        返回原始 dict 列表(不做清洗)。
        """
        ...

    @abstractmethod
    def normalize_row(self, raw: dict) -> NormalizedMatch | None:
        """单条原始记录 → NormalizedMatch。"""
        ...

    def normalize(self, raw_records: list[dict]) -> list[NormalizedMatch]:
        """批量归一化(覆盖可自定义)。"""
        results: list[NormalizedMatch] = []
        unmatched: list[str] = []
        for raw in raw_records:
            try:
                m = self.normalize_row(raw)
                if m is not None:
                    results.append(m)
            except Exception as e:
                self._logger.debug("normalize skip: %s", e)
                unmatched.append(str(raw)[:100])
        return results

    def fetch(self, **kwargs: Any) -> SourceResult:
        """统一入口:fetch → normalize → SourceResult。

        子类可覆盖以实现自定义流程(如分页、限速)。
        """
        result = SourceResult(
            source=self.SOURCE_NAME,
            data_type=self.DATA_TYPE,
        )
        try:
            raw_records = self.fetch_raw(**kwargs)
            result.raw_count = len(raw_records)
            result.records = self.normalize(raw_records)
        except Exception as e:
            result.add_error(f"{type(e).__name__}: {e}")
            self._logger.error("[%s] fetch failed: %s", self.SOURCE_NAME, e)
        return result

    def ingest(
        self,
        records: list[NormalizedMatch],
        source: str | None = None,
    ) -> dict:
        """入库(委托给 pipeline.ingest)。"""
        from app.data.pipeline.ingest import upsert_matches

        return upsert_matches(
            records,
            source=source or self.SOURCE_NAME,
        )

    def run(self, **kwargs: Any) -> dict:
        """完整流程:fetch → normalize → ingest。

        返回 {"source": ..., "fetch": ..., "ingest": ..., "success": ...}
        """
        result = self.fetch(**kwargs)
        ingest_result = {"inserted": 0, "updated": 0, "skipped": 0, "errors": []}
        if result.records:
            ingest_result = self.ingest(result.records)
        return {
            "source": self.SOURCE_NAME,
            "data_type": self.DATA_TYPE,
            "raw_count": result.raw_count,
            "record_count": result.record_count,
            "fetch_success": result.success,
            "fetch_errors": result.errors,
            "ingest": ingest_result,
            "success": result.success and not ingest_result.get("errors"),
        }

