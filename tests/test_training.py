import pytest

pytestmark = pytest.mark.db


"""训练流水线测试(冒烟):验证训练链完整。"""


@pytest.mark.slow
def test_train_model_smoke(db_ctx):
    """英超训练冒烟(验证 model_service 拆分后训练链完整)。"""
    from app.core.config import LeagueType
    from app.services.training.trainer import train_model

    m = train_model(
        LeagueType.PREMIER_LEAGUE,
        "goals",
        True,
        5,
        models_dir=str(
            __import__("app.core.paths", fromlist=["MODELS_DIR"]).MODELS_DIR
        ),
    )
    assert m.get("model_version")
    assert m.get("poisson_loss") is not None or m.get("training_metrics")


@pytest.mark.slow
def test_train_model_artifact_created(db_ctx):
    """训练后应生成 model_version。"""
    from app.core.config import LeagueType
    from app.services.training.trainer import train_model

    m = train_model(
        LeagueType.PREMIER_LEAGUE,
        "goals",
        True,
        5,
        models_dir=str(
            __import__("app.core.paths", fromlist=["MODELS_DIR"]).MODELS_DIR
        ),
    )

    version = m.get("model_version")
    assert version is not None, "model_version 缺失"
    assert isinstance(version, str)
    assert len(version) > 0
