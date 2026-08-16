"""统一路径(审查 §19:PROJECT_ROOT/DATA_DIR/MODELS_DIR/CONFIG_DIR 单一来源)。

所有模块经此引用项目根与关键目录,禁止硬编码绝对路径。
"""
from __future__ import annotations

from pathlib import Path

# app/core/paths.py → parents[2] = 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 数据文件(§28:根 data/);数据引擎代码在 app/data/
DATA_DIR = PROJECT_ROOT / "data"
APP_DATA_DIR = PROJECT_ROOT / "app" / "data"
MODELS_DIR = PROJECT_ROOT / "app" / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
CONFIG_DIR = PROJECT_ROOT / "configs"
DB_PATH = DATA_DIR / "football.db"


def ensure_dirs() -> None:
    for d in (DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, CONFIG_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
