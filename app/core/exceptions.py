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


class CorePredictionError(AppError):
    """审查十 P0-1:核心预测失败 —— 直接失败,不生成正式 Snapshot。

    code ∈ {CORE_PREDICTION_FAILURE, INVALID_LAMBDA, INVALID_PROBABILITY,
            ENSEMBLE_FAILURE, SCORE_MATRIX_FAILURE, ...}。
    与"可选成员降级"(GBM/CALIBRATION/OPTIONAL_PRIOR)严格区分:
    核心失败不允许静默退回 HGBR 后返回"看起来正常"的预测。
    """

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}", status_code=500)
        self.code = code


class DataUnavailableError(AppError):
    """数据缺失/过期(可降级为 NaN/跳过)。"""


class FeatureUnavailableError(AppError):
    """特征族数据不可用(降级:该族留 NaN,不阻断主链路)。"""


class FeatureSchemaError(AppError):
    """特征 schema 不匹配(列/类型与模型契约不一致 —— fail-fast)。"""


class FeatureComputationError(AppError):
    """特征计算失败(实现异常 —— fail-fast,不静默降级)。"""
