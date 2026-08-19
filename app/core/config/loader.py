"""多联赛足球预测模型系统 — 配置加载(YAML 单一入口 + 稳定路径)。

审查 P2-2:core/config.py 拆分 —— 本模块负责"如何从 configs/ 读取配置",
不关心业务超参。load_yaml 依据 core/paths.PROJECT_ROOT 定位,保证运行
目录无关性(与旧 top-level load_yaml 行为等价)。
"""
from __future__ import annotations

import os


def _config_dir() -> str:
    from app.core.paths import PROJECT_ROOT
    return os.path.join(str(PROJECT_ROOT), "configs")


def load_yaml(name: str) -> dict:
    """加载 configs/<name>.yaml(§1.1);缺失/损坏返回 {}。"""
    import yaml

    path = os.path.join(_config_dir(), name)
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}
