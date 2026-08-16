"""统一日志配置(审查 §35:core 提供,替代散落的 basicConfig)。"""
from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """统一日志配置:stderr 输出,含时间戳。"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_FORMAT,
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def get_logger(name: str) -> logging.Logger:
    """统一 logger 获取(格式已由 setup_logging 配置)。"""
    return logging.getLogger(name)
