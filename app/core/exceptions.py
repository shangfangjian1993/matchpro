"""统一业务异常(审查 §35:core 提供;CLI run 统一捕获 → exit 1)。"""
from __future__ import annotations


class AppError(Exception):
    """业务异常基类(API 层可映射为 4xx;CLI 层 exit 1)。"""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ValidationError(AppError):
    """参数/业务校验失败。"""


class ModelNotReadyError(AppError):
    """模型未训练/版本缺失/加载失败。"""

    def __init__(self, message: str):
        super().__init__(message, status_code=503)


class DataError(AppError):
    """数据缺失/异常。"""
