#!/usr/bin/env python3
"""滚动重训 Walk-Forward 回测门禁(第五轮审查 二十四/二十五 推进第二项)。

替代旧 backtest.py(其 app.api.model_service 引用已随重构删除)。

严格防泄漏:
- 每场测试比赛 m 只用 match_date < m.match_date 的完赛比赛构造特征
- 滚动重训:每 RETRAIN_EVERY 场,用"该场之前"的全部数据训练新模型(内存,不落盘,
  不污染 artifacts)—— 不是"最终模型回看历史"
- baseline = 测试场次 hist 窗口的 1X2 频率概率(A/B 对比对象)

指标(模型 vs baseline):
- 1X2:log-loss / brier / rps / accuracy
- 比分:score-hit / top3 / top5;让球:半球/一球
- 校准:ECE

门禁判定:模型 1X2 log-loss 优于 baseline 且 brier/rps 不退化(≥Δ 阈值)才算"通过"。

输出:artifacts/experiments/backtest/report_<ts>.json(按联赛)。

用法:
    python walkforward_backtest.py --sample 400 --retrain-every 100
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("JWT_SECRET_KEY", "backtest-secret-key-0123456789")
os.environ.setdefault("ADMIN_PASSWORD", "TestAdmin123")
os.environ.setdefault("DATABASE_URL", "sqlite:////data/matchpro/data/football.db")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

from app.api.db import Match, League, init_db
from app.core.config import LeagueType
from app.data.adapters import matches_to_dataframe
from app.models.distributions import pois_matrix
from app.replay.metrics import brier_score, ece, log_loss, rps

LEAGUE_TYPES = [LeagueType.PREMIER_LEAGUE, LeagueType.LA_LIGA, LeagueType.BUNDESLIGA,
                LeagueType.SERIE_A, LeagueType.LIGUE_1]


def _train_on(prefix_matches, lt, league):
    """用前缀比赛训练内存模型(不落盘)。"""
    from app.services.training.model_trainer import ModelTrainer
    df = matches_to_dataframe(prefix_matches)
    mt = ModelTrainer()
    mt.train_model(df, lt, cross_validation=False)
    return mt.model, mt


def _predict_once(builder, engine, lt, m, md, model):
    """统一预测链路(审查 P0-1):ContextBuilder + PredictionEngine ——
    与生产 predict_match 完全同一代码路径(Goal/GBM/Calibration/Regime)。"""
    ctx = builder.build(lt, m.home_team, m.away_team, md, model=model,
                        hist_limit=500)
    result = engine.predict(ctx)
    internal = result.get("_internal", {})
    return result, internal


def backtest_league(lt: LeagueType, sample: int, retrain_every: int) -> dict:
    from app.core.paths import MODELS_DIR as _MD
    from app.prediction.context import ContextBuilder
    from app.prediction.engine import PredictionEngine
    builder = ContextBuilder(str(_MD))
    engine = PredictionEngine(str(_MD))
    league = League.query.filter_by(league_type=lt.value).first()
    if league is None:
        return {"league": lt.value, "error": "无联赛"}
    matches = (Match.query.filter_by(league_id=league.id, match_status="finished")
               .order_by(Match.match_date.asc()).all())
    if len(matches) < 200:
        return {"league": lt.value, "error": f"数据不足 {len(matches)}"}
    test = matches
    if sample and len(test) > sample:
        idx = np.linspace(0, len(test) - 1, sample).astype(int)
        test = [test[i] for i in sorted(set(idx))]
    hist_max = 500

    model = None
    st = {"n": 0, "score_hit": 0, "top3_hit": 0, "top5_hit": 0, "result_hit": 0,
          "hcap_05_n": 0, "hcap_05_hit": 0, "hcap_1_n": 0, "hcap_1_hit": 0,
          "draw_n": 0, "ou25_n": 0, "ou25_hit": 0, "btts_n": 0, "btts_hit": 0,
          "ll": 0.0, "brier": 0.0, "rps": 0.0, "ece": 0.0,
          "base_ll": 0.0, "base_brier": 0.0, "base_rps": 0.0,
          "pvecs": [], "acts": [], "bvecs": [], "retrains": 0, "secs": 0.0}
    t_start = time.time()

    for i, m in enumerate(test):
        md = pd.Timestamp(m.match_date)
        # 暂无其他历史(严格更早)
        cutoff = [x for x in matches if pd.Timestamp(x.match_date) < md]
        # 滚动重训:每个 retrain_every 场次用"该场之前"最多 6000 场训练
        if model is None or (i > 0 and i % retrain_every == 0):
            prefix = cutoff[-6000:] if len(cutoff) > 6000 else cutoff
            prefix = prefix[-3000:]  # 训练窗口 3000(足够滚动特征收敛,控时)
            if len(prefix) >= 200:
                t0 = time.time()
                try:
                    model, _ = _train_on(prefix, lt, league)
                    st["retrains"] += 1
                    print(f"  {lt.value} 重训#{st['retrains']} @场{i} "
                          f"(前缀 {len(prefix)} 场, {time.time()-t0:.0f}s)", flush=True)
                except Exception as e:
                    print(f"  {lt.value} 重训失败: {e}", flush=True)
                    model = None
        if model is None:
            continue
        if len(cutoff) < 30:
            continue
        try:
            result, internal = _predict_once(builder, engine, lt, m, md, model)
        except Exception:
            continue
        lam_h, lam_a = internal.get("home_lambda"), internal.get("away_lambda")
        if not lam_h or not lam_a or not (lam_h > 0 and lam_a > 0):
            continue
        gh, ga = (m.home_goals or 0), (m.away_goals or 0)
        actual = 0 if gh > ga else (1 if gh == ga else 2)
        # 模型 1X2(统一链路:Goal+GBM+校准+Regime 调整)
        phw, pdr, paw = result["home_win_probability"], result["draw_probability"], result["away_win_probability"]
        pvec = [phw, pdr, paw]
        # baseline:hist 窗口频率(截止该场,时间安全)
        _hist_for_base = matches_to_dataframe(cutoff[-hist_max:])
        hh = _hist_for_base[_hist_for_base["home_goals"].notna()]
        if len(hh) >= 30:
            b_home = ((hh["home_goals"] > hh["away_goals"]).mean())
            b_draw = ((hh["home_goals"] == hh["away_goals"]).mean())
            b_away = 1 - b_home - b_draw
        else:
            b_home, b_draw, b_away = 0.45, 0.28, 0.27
        bvec = [b_home, b_draw, b_away]

        # 统一链路:score matrix(Regime 调整后,与线上同源)
        _fm = internal.get("fused_matrix")
        if isinstance(_fm, list) and _fm:
            grid = np.array(_fm, dtype=float)
        else:
            grid = pois_matrix(lam_h, lam_a)
        pred_h, pred_a = np.unravel_index(int(np.argmax(grid)), grid.shape)
        pred_dir = np.argmax(pvec)
        actual_net = gh - ga
        pred_net = lam_h - lam_a

        st["n"] += 1
        if pred_h == gh and pred_a == ga:
            st["score_hit"] += 1
        flat = grid.flatten()
        top = np.argsort(flat)[::-1]
        if (gh, ga) in set(map(tuple, [np.unravel_index(i, grid.shape) for i in top[:3]])):
            st["top3_hit"] += 1
        if (gh, ga) in set(map(tuple, [np.unravel_index(i, grid.shape) for i in top[:5]])):
            st["top5_hit"] += 1
        if pred_dir == actual:
            st["result_hit"] += 1
        st["hcap_05_n"] += 1
        if (pred_net > 0.5) == (actual_net > 0.5):
            st["hcap_05_hit"] += 1
        if actual_net != 1:
            st["hcap_1_n"] += 1
            if (pred_net > 1.0) == (actual_net > 1.0):
                st["hcap_1_hit"] += 1
        st["ou25_n"] += 1
        if (lam_h + lam_a > 2.5) == (gh + ga > 2.5):
            st["ou25_hit"] += 1
        st["btts_n"] += 1
        p_btts = 1 - grid[:, 0].sum() - grid[0, :].sum() + grid[0, 0]
        p_btts = float(np.clip(p_btts, 0, 1))
        if (p_btts > 0.5) == (gh > 0 and ga > 0):
            st["btts_hit"] += 1
        st["ll"] += log_loss(pvec, actual)
        st["brier"] += brier_score(pvec, actual)
        st["rps"] += rps(pvec, actual)
        st["base_ll"] += log_loss(bvec, actual)
        st["base_brier"] += brier_score(bvec, actual)
        st["base_rps"] += rps(bvec, actual)
        st["pvecs"].append(pvec)
        st["acts"].append(actual)
        st["bvecs"].append(bvec)
    st["secs"] = round(time.time() - t_start, 1)
    n = max(1, st["n"])
    out = {
        "league": lt.value, "n": st["n"], "retrains": st["retrains"],
        "secs": st["secs"],
        "score_hit_rate": round(st["score_hit"] / n, 4),
        "top3_hit_rate": round(st["top3_hit"] / n, 4),
        "top5_hit_rate": round(st["top5_hit"] / n, 4),
        "result_accuracy": round(st["result_hit"] / n, 4),
        "hcap_05": round(st["hcap_05_hit"] / max(1, st["hcap_05_n"]), 4),
        "hcap_1": round(st["hcap_1_hit"] / max(1, st["hcap_1_n"]), 4),
        "over25": round(st["ou25_hit"] / max(1, st["ou25_n"]), 4),
        "btts": round(st["btts_hit"] / max(1, st["btts_n"]), 4),
        # A/B:模型 vs baseline(1X2)
        "log_loss": round(st["ll"] / n, 5),
        "brier": round(st["brier"] / n, 5),
        "rps": round(st["rps"] / n, 5),
        "base_log_loss": round(st["base_ll"] / n, 5),
        "base_brier": round(st["base_brier"] / n, 5),
        "base_rps": round(st["base_rps"] / n, 5),
        "d_ll": round((st["ll"] - st["base_ll"]) / n, 5),
        "d_brier": round((st["brier"] - st["base_brier"]) / n, 5),
        "d_rps": round((st["rps"] - st["base_rps"]) / n, 5),
        "ece": round(ece(np.array(st["pvecs"]), np.array(st["acts"])), 4),
        # 门禁判定:ll 优于 baseline 且 brier/rps 不显著退化(Δ ≤ +0.005)
        "gate_pass": bool(st["ll"] < st["base_ll"] and st["brier"] <= st["base_brier"] + 0.005
                          and st["rps"] <= st["base_rps"] + 0.005),
    }
    print(f"  {lt.value}: n={out['n']} retrains={out['retrains']} "
          f"acc={out['result_accuracy']:.4f} ll={out['log_loss']:.4f} "
          f"base_ll={out['base_log_loss']:.4f} gate={'PASS' if out['gate_pass'] else 'FAIL'}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="滚动重训 Walk-Forward 回测门禁")
    ap.add_argument("--sample", type=int, default=400, help="每联赛最大测试场数(默认 400≈5 联赛 2000 总)")
    ap.add_argument("--retrain-every", type=int, default=100, help="每 N 场重训一次(防泄漏滚动)")
    ap.add_argument("--league", default=None)
    args = ap.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING)
    init_db()
    report = {"script": "walkforward_backtest", "sample": args.sample,
              "retrain_every": args.retrain_every, "leagues": {}}
    for lt in LEAGUE_TYPES:
        if args.league and lt.value != args.league:
            continue
        report["leagues"][lt.value] = backtest_league(lt, args.sample, args.retrain_every)

    out_dir = os.path.join(_ROOT, "artifacts", "experiments", "backtest")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"report_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 回测报告: {path}")


if __name__ == "__main__":
    main()