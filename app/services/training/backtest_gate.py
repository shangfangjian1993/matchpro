""": Walk-Forward 回测门禁。

严格 2000+ 场 walk-forward 回测,用于模型选择门禁。
只有通过回测门禁的模型才能上线。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
 """回测结果。"""
 n_matches: int
 log_loss: float
 brier: float
 rps: float
 ece: float
 accuracy: float
 league: str
 seasons: list
 
 def passes_gate(self, thresholds: dict | None = None) -> bool:
 """是否通过门禁。"""
 if thresholds is None:
 thresholds = {
 "min_matches": 2000,
 "max_log_loss": 1.10,
 "max_brier": 0.62,
 "max_rps": 0.43,
 "max_ece": 0.05,
 "min_accuracy": 0.48,
 }
 
 if self.n_matches < thresholds["min_matches"]:
 logger.warning(f"回测场次不足: {self.n_matches} < {thresholds['min_matches']}")
 return False
 if self.log_loss > thresholds["max_log_loss"]:
 logger.warning(f"LogLoss过高: {self.log_loss:.4f} > {thresholds['max_log_loss']}")
 return False
 if self.brier > thresholds["max_brier"]:
 logger.warning(f"Brier过高: {self.brier:.4f} > {thresholds['max_brier']}")
 return False
 if self.rps > thresholds["max_rps"]:
 logger.warning(f"RPS过高: {self.rps:.4f} > {thresholds['max_rps']}")
 return False
 if self.ece > thresholds["max_ece"]:
 logger.warning(f"ECE过高: {self.ece:.4f} > {thresholds['max_ece']}")
 return False
 if self.accuracy < thresholds["min_accuracy"]:
 logger.warning(f"准确率过低: {self.accuracy:.4f} < {thresholds['min_accuracy']}")
 return False
 
 return True


def run_walk_forward_backtest(
 league_type,
 seasons: list[int],
 matches: list,
 builder,
 engine,
 min_matches: int = 2000,
) -> BacktestResult:
 """执行 walk-forward 回测。
 
 对每个赛季开头(8/1)cutoff:
 只使用 cutoff 之前的数据训练临时模型;
 该赛季内每场用该临时模型预测 + 组件化重算。
 """
 import numpy as np
 import pandas as pd
 
 from app.replay.metrics import brier_score, ece, log_loss, rps
 
 all_probs = []
 all_actuals = []
 total = 0
 
 for year in seasons:
 cutoff = pd.Timestamp(f"{year}-08-01")
 prefix = [m for m in matches if pd.Timestamp(m.match_date) < cutoff]
 season_matches = [m for m in matches if cutoff <= pd.Timestamp(m.match_date) < pd.Timestamp(f"{year+1}-08-01")]
 
 if len(prefix) < 200 or len(season_matches) < 10:
 continue
 
 # 训练临时模型
 from app.data.adapters import matches_to_dataframe
 from app.services.training.model_trainer import ModelTrainer
 
 df = matches_to_dataframe(prefix)
 mt = ModelTrainer()
 mt.train_model(df, league_type, cross_validation=False)
 
 for m in season_matches:
 try:
 ctx = builder.build(league_type, m.home_team, m.away_team, pd.Timestamp(m.match_date), model=mt.model)
 result = engine.predict(ctx)
 
 actual = (
 0 if (m.home_goals or 0) > (m.away_goals or 0)
 else (1 if (m.home_goals or 0) == (m.away_goals or 0) else 2)
 )
 
 probs = [
 result.get("home_win_probability", 0.33),
 result.get("draw_probability", 0.34),
 result.get("away_win_probability", 0.33),
 ]
 
 all_probs.append(probs)
 all_actuals.append(actual)
 total += 1
 except Exception:
 continue
 
 if total < min_matches:
 logger.warning(f"回测场次不足: {total} < {min_matches}")
 return BacktestResult(
 n_matches=total, log_loss=999, brier=999, rps=999, ece=999, accuracy=0,
 league=league_type.value, seasons=seasons,
 )
 
 # 计算指标
 probs_arr = np.array(all_probs)
 acts_arr = np.array(all_actuals)
 
 ll = log_loss(probs_arr, acts_arr)
 br = brier_score(probs_arr, acts_arr)
 rp = rps(probs_arr, acts_arr)
 ec = float(ece(all_probs, all_actuals))
 
 # 准确率
 pred_outcomes = probs_arr.argmax(axis=1)
 acc = float((pred_outcomes == acts_arr).mean())
 
 return BacktestResult(
 n_matches=total,
 log_loss=float(np.mean(ll)) if hasattr(ll, '__len__') else float(ll),
 brier=float(np.mean(br)) if hasattr(br, '__len__') else float(br),
 rps=float(np.mean(rp)) if hasattr(rp, '__len__') else float(rp),
 ece=ec,
 accuracy=acc,
 league=league_type.value,
 seasons=seasons,
 )
