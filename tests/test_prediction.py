import pytest

pytestmark = pytest.mark.db

"""预测链路最小测试:输出完整性 + 快照幂等。"""


@pytest.fixture(autouse=True)
def _setup_artifact(tmp_path, monkeypatch):
    """Create production_artifact.json in isolated temp dir."""
    import json
    
    fixture = {
        "league": "premier_league",
        "goal_lambda": {"hgbr": 0.52, "elo": 0.29, "bayes": 0.19},
        "score_distribution": {"poisson": 0.48, "dc": 0.32, "nb": 0.20},
        "outcome": {"shape": 0.73, "gbm": 0.27},
        "tau": -0.071,
        "phi": 2.31,
        "gbm_model_path": "",
        "gbm_model_hash": "",
        "calibration": None,
        "prior": None,
        "lineage": {
            "artifact_version": "ensemble-v3",
            "schema_version": 1,
            "model_version": "test",
            "feature_version": "test",
            "training_cutoff": "2026-08-01",
            "oof_method": "expanding-window",
            "oof_segments": 6,
            "oof_n": 600,
            "shrinkage": 0.15,
            "created_at": "2026-08-21T00:00:00+00:00",
            "training_data_hash": "test",
            "calibration_hash": "",
            "prior_hash": "",
            "gbm_hash": ""
        }
    }
    
    # Engine looks for: ARTIFACTS_DIR / "ensemble" / league_type_value
    artifact_dir = tmp_path / "ensemble" / "premier_league"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "production_artifact.json"
    artifact_path.write_text(json.dumps(fixture), encoding="utf-8")
    
    # Monkeypatch the ARTIFACTS_DIR to point to tmp_path
    import app.core.paths as _paths
    monkeypatch.setattr(_paths, "ARTIFACTS_DIR", tmp_path, raising=False)


def test_predict_match_full_output(db_ctx, _setup_artifact):
    """§4 输出完整:胜平负 + Top5 + Over/Under + BTTS + xG。"""
    from app.core.config import LeagueType
    from app.prediction.service import predict_match

    r = predict_match(LeagueType.PREMIER_LEAGUE, "阿森纳", "切尔西")
    for k in (
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
        "top_scores",
        "over_2_5",
        "under_2_5",
        "btts",
        "expected_xg",
    ):
        assert k in r, f"缺少 {k}"
    # 输出为 round(p, 4) 的独立四舍五入,和差最大 ~1.5e-4;用 1e-3 容差
    assert (
        abs(
            r["home_win_probability"]
            + r["draw_probability"]
            + r["away_win_probability"]
            - 1.0
        )
        < 1e-3
    )
    assert len(r["top_scores"]) == 5
    assert abs(r["over_2_5"] + r["under_2_5"] - 1.0) < 1e-6
    assert len(r["expected_xg"]) == 2


def test_predict_snapshot_idempotent(db_ctx):
    """快照幂等:同日同对阵不重复。"""
    from app.api.db import PredictionSnapshot
    from app.core.config import LeagueType
    from app.prediction.service import predict_match

    before = PredictionSnapshot.query.filter_by(
        home_team="Arsenal FC", away_team="Chelsea FC"
    ).count()
    predict_match(LeagueType.PREMIER_LEAGUE, "阿森纳", "切尔西")
    after = PredictionSnapshot.query.filter_by(
        home_team="Arsenal FC", away_team="Chelsea FC"
    ).count()
    assert after == before


def test_predict_unknown_team_rejected(db_ctx):
    """未知球队拒绝。"""
    import pytest

    from app.core.config import LeagueType
    from app.prediction.service import predict_match

    with pytest.raises(ValueError):
        predict_match(LeagueType.PREMIER_LEAGUE, "测试队XYZ", "切尔西")
