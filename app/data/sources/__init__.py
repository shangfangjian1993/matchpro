"""源适配器(1.1 app/data/sources):每个源一个模块。"""

from app.data.sources.fdco import fetch_fdco
from app.data.sources.fdo import fetch_fdo
from app.data.sources.understat import fetch_understat

__all__ = ["fetch_fdco", "fetch_fdo", "fetch_understat"]
