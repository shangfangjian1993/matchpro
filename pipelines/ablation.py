"""Season-start Expanding-Window 层级消融评估。

所有层级调用 LayeredPipeline.compute_prediction(),不使用硬编码权重。
Ablation 定义为"阶段快照":每级只启用到该层,后续层显式禁用。
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
 sys.path.insert(0, str(_ROOT))

from app.api.db import League, Match, init_db
from app.core.config import LeagueType
from app.core.paths import MODELS_DIR
from app.data.adapters import matches_to_dataframe
from app.models.ensemble.matrix import extract_dc_low_score_probs, extract_nb_tail_probs, compute_tail_mass
from app.prediction.context import ContextBuilder
from app.prediction.engine import PredictionEngine
from app.prediction.layered_pipeline import ABLATION_MASKS, compute_prediction
from app.replay.metrics import brier_score, ece, log_loss, rps


def _component_probs(internal: dict, league_id, match_dt, pf_matrix) -> dict:
 """层级消融 → 各组件 1X2 + score_matrix。"""
 lam = internal.get("member_lambdas") or {}
 w_all = internal.get("member_weights") or {}
 gbm = internal.get("gbm_probs")
 tau = internal.get("tau", 0.0)
 phi = internal.get("phi", 1e9)

 hg = lam.get("hgbr")
 el = lam.get("elo")
 ba = lam.get("bayes")

 if hg is None:
 return None

 raw_matrix = internal.get("raw_fused_matrix")
 if raw_matrix is None:
 raw_matrix = pf_matrix

 out = {}

 for comp_name, mask in ABLATION_MASKS.items():
 result = compute_prediction(
 lam_h=hg[0], lam_a=hg[1],
 lam_eh=el[0] if el else 0.0, lam_ea=el[1] if el else 0.0,
 tau=tau, phi=phi, weights=w_all,
 lam_bh=ba[0] if ba else None, lam_ba=ba[1] if ba else None,
 gbm_probs=gbm,
 prior_context={"league_id": league_id, "match_dt": match_dt, "raw_matrix": raw_matrix},
 calibration_context={"models_dir": str(MODELS_DIR), "league_type": LeagueType.PREMIER_LEAGUE},
 ablation_mask=mask,
 )
 if result is None:
 out[comp_name] = None
 continue

 if comp_name in ["A", "B", "C", "D", "E"]:
 out[comp_name] = (result.shape_1x2, result.score_matrix)
 elif comp_name == "F":
 out[comp_name] = (result.outcome_1x2, result.score_matrix)
 elif comp_name in ["G", "H"]:
 out[comp_name] = (result.final_1x2, result.score_matrix)

 return out


def _train_at(prefix_matches, lt: LeagueType, league):
 from app.services.training.model_trainer import ModelTrainer
 if len(prefix_matches) < 200:
 return None
 df = matches_to_dataframe(prefix_matches, league_name=league.name, league_season=league.season or "")
 mt = ModelTrainer()
 mt.train_model(df, lt, cross_validation=False)
 return mt.model


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--league", default="premier_league")
 ap.add_argument("--seasons", default="2022,2023,2024,2025")
 ap.add_argument("--limit", type=int, default=0)
 ap.add_argument("--out", default="/opt/data/ablation_season_start_report.json")
 args = ap.parse_args()

 init_db()
 lt = LeagueType(args.league)
 league = League.query.filter_by(league_type=lt.value).first()
 if league is None:
 print("无联赛")
 return
 matches = (
 Match.query.filter_by(league_id=league.id, match_status="finished")
 .order_by(Match.match_date.asc())
 .all()
 )
 builder = ContextBuilder(str(MODELS_DIR))
 engine = PredictionEngine(str(MODELS_DIR))
 seasons = [int(s) for s in args.seasons.split(",")]
 comps = ["A", "B", "C", "D", "E", "F", "G", "H"]
 report = {
 s: {cc: {"n": 0, "ll": 0.0, "brier": 0.0, "rps": 0.0,
 "pvecs": [], "acts": [], "xgm_h": 0.0, "xgm_a": 0.0,
 "dc_low_score_n": 0, "nb_tail_n": 0, "total_goals": 0,
 "dc_p00": 0.0, "dc_p10": 0.0, "dc_p01": 0.0, "dc_p11": 0.0,
 "nb_pge4": 0.0, "nb_pge5": 0.0, "tail_mass": 0.0}
 for cc in comps}
 for s in seasons
 }
 t0 = time.time()
 total = 0

 for year in seasons:
 start_ts = pd.Timestamp(f"{year}-08-01")
 next_ts = pd.Timestamp(f"{year + 1}-08-01")
 prefix = [m for m in matches if pd.Timestamp(m.match_date) < start_ts]
 season_matches = [m for m in matches if start_ts <= pd.Timestamp(m.match_date) < next_ts]
 model = _train_at(prefix, lt, league)
 if model is None:
 print(f" {year}: prefix {len(prefix)} < 200,跳过")
 continue
 print(f" {year}: prefix {len(prefix)} 训练 OK,赛季场 {len(season_matches)}", flush=True)
 for m in season_matches:
 md = pd.Timestamp(m.match_date)
 try:
 ctx = builder.build(lt, m.home_team, m.away_team, md, model=model, hist_limit=500)
 result = engine.predict(ctx)
 int_ = result["_internal"]
 except Exception as _e:
 print(f"[skip] {year} {m.home_team} vs {m.away_team}: {_e}", file=sys.stderr)
 continue
 actual = (
 0 if (m.home_goals or 0) > (m.away_goals or 0)
 else (1 if (m.home_goals or 0) == (m.away_goals or 0) else 2)
 )
 raw_m = int_.get("raw_fused_matrix")
 if raw_m is None:
 raw_m = np.eye(10)
 probs_c = _component_probs(int_, league.id, md, raw_m)
 if probs_c is None:
 continue
 for comp, pv_data in probs_c.items():
 if pv_data is None:
 continue
 pv, score_matrix = pv_data
 d = report[year][comp]
 d["n"] += 1
 d["ll"] += log_loss(pv, actual)
 d["brier"] += brier_score(pv, actual)
 d["rps"] += rps(pv, actual)
 d["pvecs"].append(list(pv))
 d["acts"].append(actual)
 if score_matrix is not None:
 mtx = np.asarray(score_matrix, dtype=float)
 grid = np.arange(mtx.shape[0], dtype=float)
 xh = float((mtx * grid[:, None]).sum())
 xa = float((mtx * grid[None, :]).sum())
 d["xgm_h"] += abs(xh - (m.home_goals or 0))
 d["xgm_a"] += abs(xa - (m.away_goals or 0))
 # opt-1: tail mass
 tail = compute_tail_mass(mtx)
 d["tail_mass"] += tail["tail_mass"]
 # opt-2: DC low-score calibration
 dc_cal = extract_dc_low_score_probs(mtx)
 d["dc_p00"] += dc_cal["p_00"]
 d["dc_p10"] += dc_cal["p_10"]
 d["dc_p01"] += dc_cal["p_01"]
 d["dc_p11"] += dc_cal["p_11"]
 # opt-3: NB tail calibration
 nb_cal = extract_nb_tail_probs(mtx)
 d["nb_pge4"] += nb_cal["p_total_ge4"]
 d["nb_pge5"] += nb_cal["p_total_ge5"]
 gh, ga = m.home_goals or 0, m.away_goals or 0
 if gh <= 1 and ga <= 1:
 d["dc_low_score_n"] += 1
 if gh + ga >= 4:
 d["nb_tail_n"] += 1
 d["total_goals"] += gh + ga
 total += 1
 if args.limit and total >= args.limit:
 break
 if args.limit and total >= args.limit:
 break

 summary = {}
 for s in seasons:
 row = {}
 for cc in comps:
 d = report[s][cc]
 if d["n"] == 0:
 row[cc] = {"n": 0}
 continue
 n = d["n"]
 row[cc] = {
 "n": n,
 "ll": round(d["ll"] / n, 5),
 "brier": round(d["brier"] / n, 5),
 "rps": round(d["rps"] / n, 5),
 "ece": round(float(ece(d["pvecs"], d["acts"])), 5),
 "xgm_h": round(d["xgm_h"] / n, 4),
 "xgm_a": round(d["xgm_a"] / n, 4),
 "dc_low_score_n": d["dc_low_score_n"],
 "nb_tail_n": d["nb_tail_n"],
 "avg_total_goals": round(d["total_goals"] / n, 3),
 # opt-1: tail mass
 "avg_tail_mass": round(d["tail_mass"] / n, 6),
 # opt-2: DC low-score predicted probability
 "dc_p00_pred": round(d["dc_p00"] / n, 4),
 "dc_p10_pred": round(d["dc_p10"] / n, 4),
 "dc_p01_pred": round(d["dc_p01"] / n, 4),
 "dc_p11_pred": round(d["dc_p11"] / n, 4),
 # opt-3: NB tail predicted probability
 "nb_pge4_pred": round(d["nb_pge4"] / n, 4),
 "nb_pge5_pred": round(d["nb_pge5"] / n, 4),
 }
 summary[s] = row
 payload = {
 "league": args.league,
 "seasons": summary,
 "method": "season-start-expanding-window",
 "evaluation_type": "no-future-model",
 "calibration_temporal_oof": False,
 "secs": round(time.time() - t0, 1),
 "matches": total,
 }
 with open(args.out, "w", encoding="utf-8") as f:
 json.dump(payload, f, ensure_ascii=False, indent=2)

 print(f"\n===== Season-Start Expanding-Window Evaluation: {args.league} ({total} 场) =====")
 for s in seasons:
 for cc in comps:
 r_ = summary[s].get(cc, {})
 if r_.get("n", 0) == 0:
 continue
 print(
 f"{s} {cc:>2} n={r_['n']:>4} LL={r_['ll']:.4f} B={r_['brier']:.4f} "
 f"RPS={r_['rps']:.4f} ECE={r_['ece']:.4f} xG_h={r_['xgm_h']:.3f} xG_a={r_['xgm_a']:.3f} "
 f"dc_low={r_.get('dc_low_score_n',0)} nb_tail={r_.get('nb_tail_n',0)} "
 f"tail={r_.get('avg_tail_mass',0):.4f}"
 )
 print("✅ 报告:", args.out)


if __name__ == "__main__":
 main()
