"""数据源适配器(统一接口: fetch → normalize → ingest)。"""

from app.data.pipeline.sources.base import BaseSource, SourceResult
from app.data.pipeline.sources.bzzoiro import BzzoiroSource
from app.data.pipeline.sources.fdco import FdcoSource
from app.data.pipeline.sources.understat import UnderstatSource
from app.data.pipeline.sources.api_football import ApiFootballSource
from app.data.pipeline.sources.statsbomb import StatsBombSource
from app.data.pipeline.sources.zafronix import ZafronixSource
from app.data.pipeline.sources.fdo import FdoSource

__all__ = [
    "BaseSource",
    "SourceResult",
    "BzzoiroSource",
    "FdcoSource",
    "UnderstatSource",
    "ApiFootballSource",
    "StatsBombSource",
    "ZafronixSource",
    "FdoSource",
]

