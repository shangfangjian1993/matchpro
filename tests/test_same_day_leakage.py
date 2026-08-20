#!/usr/bin/env python3
"""Same-day prediction-cutoff 数据泄漏测试。

验证:预测某场比赛时,不能看到同天(但时间更晚)的其他比赛。
"""
from __future__ import annotations

from unittest.mock import MagicMock


# 创建 mock 模型对象
def _mock_model(*args, **kwargs):
    model = MagicMock()
    model.model = MagicMock()
    model.model.predict = MagicMock(return_value=([1.5, 1.2],))
    model.feature_columns_ = []
    return model
