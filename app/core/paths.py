"""统一路径(

所有模块经此引用项目根与关键目录,禁止硬编码绝对路径。
"""

from __future__ import annotations

from pathlib import Path

# app/core/paths.py → parents[2] = 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 数据文件(28:根 data/);数据引擎代码在 app/data/
DATA_DIR = PROJECT_ROOT / "data"
APP_DATA_DIR = PROJECT_ROOT / "app" / "data"
# 资产体系统一(
# artifacts/models/<league>/<version>.pkl 模型
# artifacts/ensemble/ 权重/τφ
# artifacts/calibration/ 校准器
# artifacts/experiments/ 实验报告
# 禁止 os.path.join("artifacts", ...) 散落拼接 —— 一律经本常量
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
CONFIG_DIR = PROJECT_ROOT / "configs"
DB_PATH = DATA_DIR / "football.db"


def ensure_dirs() -> None:
 for d in (DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, CONFIG_DIR):
 try:
 d.mkdir(parents=True, exist_ok=True)
 except OSError:
 pass
