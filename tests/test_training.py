"""训练最小测试(冒烟):单联赛训练 + 实验记录。"""
import pytest


@pytest.mark.slow
def test_train_model_smoke(db_ctx):
    """英超训练冒烟(验证 model_service 拆分后训练链完整)。"""
    from app.core.config import LeagueType
    from app.services.training.trainer import train_model
    m = train_model(LeagueType.PREMIER_LEAGUE, "goals", True, 5,
                    models_dir=str(__import__("app.core.paths", fromlist=["MODELS_DIR"]).MODELS_DIR))
    assert m.get("model_version")
    assert m.get("poisson_loss") is not None or m.get("training_metrics")
