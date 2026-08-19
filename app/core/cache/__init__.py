"""统一缓存包(审查 §24:ArtifactCache + PredictionCache 两类)。"""

from app.core.cache.artifact import ArtifactCache
from app.core.cache.prediction import PredictionCache

__all__ = ["ArtifactCache", "PredictionCache"]
