"""OOF 样本生成(时间分段 OOF)。

时间分段 OOF:
 段0 ──训练──> 段1 ──训练(段0+1)──> 段2 ──> ... ──> 段K-1
 │ │ │
 预测段1 预测段2 预测段K-1

样本量 ≥600 —— K_SEG=6 × SAMPLE_PER_SEG=120(每联赛最多 600,实际 ~500+,下限 300)。
段前训练临时 HGBR+GBM,段内采样场次用"截止该场"历史预测(严格赛前视角)。

注意:OOF 生成的是原子预测(lambda/att_diff/bayes/gbm),后续由 member_builder
基于 fused λ 构建成员概率样本,确保与生产 LayeredPipeline 一致。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.data.adapters import matches_to_dataframe

K_SEG = 6
SAMPLE_PER_SEG = 120
MIN_PREFIX_ROWS = 100
MIN_HISTORY = 50
MIN_OOF_SAMPLES = 300


def _outcome(hg, ag):
 return 0 if hg > ag else (1 if hg == ag else 2)


def generate(lt, league, matches, verbose=True):
 """生成该联赛 OOF 样本列表(原子预测,非成员概率)。"""
 from .temporary_trainer import train_temp_gbm, train_temp_model

 n = len(matches)
 if n < 200:
 print(f" {lt.value}: 仅 {n} 场,不足 200,跳过")
 return []
 boundaries = [int(n * k / K_SEG) for k in range(1, K_SEG)] + [n]
 oof_samples = []
 for k in range(1, K_SEG):
 seg_start, seg_end = boundaries[k - 1], boundaries[k]
 seg = matches[seg_start:seg_end]
 prefix = matches[:seg_start]
 if len(seg) < 15 or len(prefix) < MIN_PREFIX_ROWS:
 continue
 if verbose:
 print(
 f" {lt.value}: 段{k}/{K_SEG - 1} 前缀 {len(prefix)} 场 → 训练临时模型...",
 flush=True,
 )
 prefix_df = matches_to_dataframe(prefix)
 temp_model = train_temp_model(prefix_df, lt)
 prefix_prepared = temp_model.prepare_features(
 temp_model._sort_by_date(prefix_df)
 )
 _gbm = train_temp_gbm(prefix_df, prefix_prepared, temp_model)
 rng = np.random.default_rng(42)
 idx = rng.choice(len(seg), size=min(SAMPLE_PER_SEG, len(seg)), replace=False)
 got = 0
 for i in idx:
 m = seg[int(i)]
 match_dt = pd.Timestamp(m.match_date)
 history = prefix + [x for x in seg if pd.Timestamp(x.match_date) < match_dt]
 if len(history) < MIN_HISTORY:
 continue
 hist_df = matches_to_dataframe(history)
 rows = [
 {
 "date": match_dt,
 "home_team": m.home_team,
 "away_team": m.away_team,
 "home_goals": np.nan,
 "away_goals": np.nan,
 "goals": np.nan,
 "league": league.name,
 "season": league.season or "",
 },
 {
 "date": match_dt,
 "home_team": m.away_team,
 "away_team": m.home_team,
 "home_goals": np.nan,
 "away_goals": np.nan,
 "goals": np.nan,
 "league": league.name,
 "season": league.season or "",
 },
 ]
 _df = pd.concat([hist_df, pd.DataFrame(rows)], ignore_index=True)
 try:
 _df = temp_model._sort_by_date(_df)
 feats = temp_model.prepare_features(_df)
 fcols = [c for c in temp_model.feature_columns_ if c in feats.columns]
 lams = temp_model.model.predict(feats[fcols])
 lam_h, lam_a = float(lams[-2]), float(lams[-1])
 att_diff = float(feats["attack_elo_diff"].iloc[-2])
 from app.models.bayes_team import bayes_lambda

 lam_bh, lam_ba = bayes_lambda(hist_df, m.home_team, m.away_team)
 except Exception as _exc:
 import logging as _lg

 _lg.getLogger(__name__).debug("OOF 场次失败,跳过: %s", _exc)
 continue
 gprob = None
 if _gbm is not None:
 try:
 _gc = [c for c in _gbm.feature_columns_ if c in feats.columns]
 gprob = list(_gbm.predict_proba(feats[_gc].iloc[[-2]])[0])
 except Exception:
 gprob = None
 oof_samples.append(
 {
 "hgbr_lam_h": lam_h,
 "hgbr_lam_a": lam_a,
 "att_diff": att_diff,
 "bayes_lam_h": lam_bh,
 "bayes_lam_a": lam_ba,
 "home_goals": m.home_goals,
 "away_goals": m.away_goals,
 "actual": _outcome(m.home_goals, m.away_goals),
 "gbm": gprob,
 }
 )
 got += 1
 if verbose:
 print(f" → 段{k} 采样 {got} 场", flush=True)
 if len(oof_samples) < MIN_OOF_SAMPLES:
 print(
 f" {lt.value}: OOF 样本仅 {len(oof_samples)}(<{MIN_OOF_SAMPLES}),跳过权重学习"
 )
 return []
 return oof_samples
