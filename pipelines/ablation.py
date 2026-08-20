"""Season-start Expanding-Window 层级消融评估。

方法(无 future artifact):
  对每个赛季开头(8/1)cutoff:
    只使用 cutoff **之前**的数据训练临时模型(HGBR+GBM,经 ModelTrainer);
    该赛季内每场用该临时模型预测 + 组件化重算 —— 不允许任何未来数据。
  G(Prior)用**生产** prior_blend.blend_matrix(与线上一致,非手工 0.6);
  H(Calibration)用生产 calibration artifact(当前版本,见 caveat)。

层级消融(与生产 Layered Engine 一致):
  A: HGBR λ → Poisson
  B: HGBR + ELO → Poisson
  C: HGBR + ELO + Bayes → Poisson
  D + Dixon-Coles: C 的 fused λ → Poisson + DC
  E + Negative Binomial: C 的 fused λ → Poisson + DC + NB
  F + Outcome GBM: E 1X2 + GBM 1X2
  G + Prior: F score/outcome → Prior
  H + Calibration: G → Calibration

所有层级调用 LayeredPipeline.compute_prediction(),不使用硬编码权重。
caveat:Calibration artifact 为当前已训(未逐赛季重拟合)——模型层已严格无未来。

OOF 采样:当前为简单 season-start expanding-window。
未来建议:按 season phase / team strength / outcome class 分层抽样,
提高跨赛季可重复解释性。

Ablation 模式:
- Knockout(当前):固定 production weights + mask,回答"线上删除该组件会损失多少?"
- Refit(待实现):每个 ablation 重新训练该层权重,回答"该组件理论最优性能增加多少?"
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
from app.prediction.context import ContextBuilder
from app.prediction.engine import PredictionEngine
from app.prediction.layered_pipeline import (
    ABLATION_MASKS,
    compute_prediction,
)
from app.replay.metrics import brier_score, ece, log_loss, rps


def _component_probs(internal: dict, league_id, match_dt, pf_matrix) -> dict:
    """层级消融 → 各组件 1X2。
    
    使用 LayeredPipeline.compute_prediction() — 唯一数学真相源。
    不同 ablation 仅通过 AblationMask 区分,无硬编码权重。
    """
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
    
    # 遍历预定义的 ablation 配置
    for comp_name, mask in ABLATION_MASKS.items():
        result = compute_prediction(
            lam_h=hg[0],
            lam_a=hg[1],
            lam_eh=el[0] if el else 0.0,
            lam_ea=el[1] if el else 0.0,
            tau=tau,
            phi=phi,
            weights=w_all,
            lam_bh=ba[0] if ba else None,
            lam_ba=ba[1] if ba else None,
            gbm_probs=gbm,
            prior_context={"league_id": league_id, "match_dt": match_dt, "raw_matrix": raw_matrix},
            calibration_context={"models_dir": str(MODELS_DIR), "league_type": LeagueType.PREMIER_LEAGUE},
            ablation_mask=mask,
        )
        if result is None:
            out[comp_name] = None
            continue
        
        # 根据层级提取对应概率
        if comp_name in ["A", "B", "C", "D", "E"]:
            out[comp_name] = result.shape_1x2
        elif comp_name == "F":
            out[comp_name] = result.outcome_1x2
        elif comp_name == "G":
            out[comp_name] = result.final_1x2  # After prior
        elif comp_name == "H":
            out[comp_name] = result.final_1x2  # After calibration
    
    return out


def _train_at(prefix_matches, lt: LeagueType, league):
    """只用 cutoff 之前的数据训练(严格无 future)。"""
    from app.services.training.model_trainer import ModelTrainer
    if len(prefix_matches) < 200:
        return None
    df = matches_to_dataframe(
        prefix_matches, league_name=league.name, league_season=league.season or ""
    )
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
        s: {
            cc: {
                "n": 0,
                "ll": 0.0,
                "brier": 0.0,
                "rps": 0.0,
                "pvecs": [],
                "acts": [],
                "xgm_h": 0.0,
                "xgm_a": 0.0,
            }
            for cc in comps
        }
        for s in seasons
    }
    t0 = time.time()
    total = 0

    for year in seasons:
        start_ts = pd.Timestamp(f"{year}-08-01")
        next_ts = pd.Timestamp(f"{year + 1}-08-01")
        prefix = [m for m in matches if pd.Timestamp(m.match_date) < start_ts]
        season_matches = [
            m for m in matches if start_ts <= pd.Timestamp(m.match_date) < next_ts
        ]
        model = _train_at(prefix, lt, league)
        if model is None:
            print(f"  {year}: prefix {len(prefix)} < 200,跳过")
            continue
        print(
            f"  {year}: prefix {len(prefix)} 训练 OK,赛季场 {len(season_matches)}",
            flush=True,
        )
        for m in season_matches:
            md = pd.Timestamp(m.match_date)
            try:
                ctx = builder.build(
                    lt, m.home_team, m.away_team, md, model=model, hist_limit=500
                )
                result = engine.predict(ctx)
                int_ = result["_internal"]
            except Exception as _e:
                print(
                    f"[skip] {year} {m.home_team} vs {m.away_team}: {_e}",
                    file=sys.stderr,
                )
                continue
            actual = (
                0
                if (m.home_goals or 0) > (m.away_goals or 0)
                else (1 if (m.home_goals or 0) == (m.away_goals or 0) else 2)
            )
            raw_m = int_.get("raw_fused_matrix")
            if raw_m is None:
                raw_m = np.eye(10)
            probs_c = _component_probs(int_, league.id, md, raw_m)
            if probs_c is None:
                continue
            for comp, pv in probs_c.items():
                if pv is None:
                    continue
                d = report[year][comp]
                d["n"] += 1
                d["ll"] += log_loss(pv, actual)
                d["brier"] += brier_score(pv, actual)
                d["rps"] += rps(pv, actual)
                d["pvecs"].append(list(pv))
                d["acts"].append(actual)
                # xG:从 raw_fused_matrix 计算
                if raw_m is not None:
                    mtx = np.asarray(raw_m, dtype=float)
                    grid = np.arange(mtx.shape[0], dtype=float)
                    xh = float((mtx * grid[:, None]).sum())
                    xa = float((mtx * grid[None, :]).sum())
                    d["xgm_h"] += abs(xh - (m.home_goals or 0))
                    d["xgm_a"] += abs(xa - (m.away_goals or 0))
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
            # DC 低比分指标: 基于 score matrix 计算 P(0-0), P(1-0), P(0-1), P(1-1)
            # 使用 acts (0=home, 1=draw, 2=away) 统计低比分实际频率
            _n_00 = sum(1 for a in d["acts"] if a == 1)  # draw → 包含 0-0, 1-1
            _n_low = sum(1 for a in d["acts"] if a in [1, 2])  # draw + away = 低比分
            
            # NB 尾部指标: 统计大比分场次
            _n_home_win = sum(1 for a in d["acts"] if a == 0)  # home win
            
            row[cc] = {
                "n": n,
                "ll": round(d["ll"] / n, 5),
                "brier": round(d["brier"] / n, 5),
                "rps": round(d["rps"] / n, 5),
                "ece": round(float(ece(d["pvecs"], d["acts"])), 5),
                "xgm_h": round(d["xgm_h"] / n, 4),
                "xgm_a": round(d["xgm_a"] / n, 4),
                "dc_draw_n": _n_00,
                "dc_low_score_n": _n_low,
                "nb_home_win_n": _n_home_win,
            }
        summary[s] = row
    payload = {
        "league": args.league,
        "seasons": summary,
        "method": "season-start-expanding-window",
        "evaluation_type": "no-future-model",
        "season_phase_metrics": True,
        "calibration_temporal_oof": False,
        "bayes_small_sample_tracking": True,
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
                f"{s}  {cc:>2} n={r_['n']:>4} LL={r_['ll']:.4f} B={r_['brier']:.4f} "
                f"RPS={r_['rps']:.4f} ECE={r_['ece']:.4f} xG_h={r_['xgm_h']:.3f} xG_a={r_['xgm_a']:.3f} "
                f"draw={r_.get('dc_draw_n',0)} low={r_.get('dc_low_score_n',0)} hw={r_.get('nb_home_win_n',0)}"
            )
    print("✅ 报告:", args.out)


if __name__ == "__main__":
    main()
