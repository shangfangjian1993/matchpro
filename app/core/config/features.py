"""多联赛足球预测模型系统 — 特征开关。


load_yaml('models.yaml').get('features')"收敛为单一入口 feature_flags()。
用于:registry.logical_version(评估/模型身份)、Factory(h2h 开关)等。
"""

from __future__ import annotations

from app.core.config.loader import load_yaml


def feature_flags() -> dict:
 """特征开关(读取 configs/models.yaml 的 features 段;缺失返回 {})。
 单一读取入口 —— 若要加默认开关,在此集中声明并合并,勿再散读 yaml。
 """
 return load_yaml("models.yaml").get("features") or {}
