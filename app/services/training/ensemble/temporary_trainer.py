"""临时模型训练(审查九 P1-9 拆分):段前 HGBR + GBM(OOF 口径,不落盘)。"""
from __future__ import annotations



def train_temp_model(prefix_df, lt):
    """用前缀数据训练临时 HGBR(模型只见过该段之前的数据)。"""
    from app.services.training.model_trainer import ModelTrainer
    mt = ModelTrainer()
    mt.train_model(prefix_df, lt, cross_validation=False)
    return mt.model


def train_temp_gbm(prefix_df, prefix_prepared, model):
    """段前 GBM(与主模型同段前缀,内部 80/20 + 100% 重训)。"""
    from app.models.ensemble.gbm import GbmClassifier

    def _outcome(hg, ag):
        return 0 if hg > ag else (1 if hg == ag else 2)

    try:
        y = prefix_df.apply(lambda r: _outcome(r["home_goals"], r["away_goals"]), axis=1)
        gcols = [c for c in model.feature_columns_ if c in prefix_prepared.columns]
        _gbm = GbmClassifier()
        _gbm.train(prefix_prepared[gcols], y)
        return _gbm
    except Exception as e:
        print(f"    [warn] 段前 GBM 训练失败(该段无 gbm 成员): {e}", flush=True)
        return None
