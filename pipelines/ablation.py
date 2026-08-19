"""严格 Walk-Forward 组件 Ablation(审查 e752f5f P0-4)。

方法(无 future artifact):
  对每个赛季开头(8/1)cutoff:
    只使用 cutoff **之前**的数据训练临时模型(HGBR+GBM,经 ModelTrainer);
    该赛季内每场用该临时模型预测 + 组件化重算 —— 不允许任何未来数据。
  G(Prior)用**生产** prior_blend.blend_matrix(与线上一致,非手工 0.6);
  H(Calibration)用生产 calibration artifact(当前版本,见 caveat)。

每组件自算 xG(自身 λ/矩阵 → 期望),不再共用最终 xG。
caveat:Calibration artifact 为当前已训(未逐赛季重拟合)——模型层已严格无未来。
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
from app.models.ensemble.fusion import fuse_goal_outcome, fuse_probs
from app.prediction.context import ContextBuilder
from app.prediction.engine import PredictionEngine
from app.replay.metrics import brier_score, ece, log_loss, rps

GOAL_ORDER = ["hgbr", "dc", "nb", "elo", "bayes"]


def _subset_weights(w_all: dict, names) -> dict:
    w = {n: w_all.get(n, 0.0) for n in names}
    tot = sum(w.values())
    return {n: v / tot for n, v in w.items()} if tot > 0 else {names[0]: 1.0}


def _matrix_xg(matrix) -> list[float]:
    m = np.asarray(matrix, dtype=float)
    if m.size == 0:
        return [0.0, 0.0]
    grid = np.arange(m.shape[0], dtype=float)
    return [float((m * grid[:, None]).sum()), float((m * grid[None, :]).sum())]


def _component_probs(internal: dict, league_id, match_dt, pf_matrix) -> dict:
    """成员分解 → 各组件 1X2;G 用生产 prior_blend;H 用生产 calibration。"""
    members = internal["members"]
    w_all = internal["member_weights"]
    gbm = internal.get("gbm_probs")
    goal = fuse_probs(members, w_all)
    raw_matrix = internal.get("raw_fused_matrix")
    if raw_matrix is None:
        raw_matrix = pf_matrix
    out = {
        "A_hgbr": fuse_probs({k: members[k] for k in ["hgbr"]}, {"hgbr": 1.0}),
        "B_elo": fuse_probs(
            {k: members[k] for k in ["hgbr", "elo"]},
            _subset_weights(w_all, ["hgbr", "elo"]),
        ),
        "C_bayes": fuse_probs(
            {k: members[k] for k in ["hgbr", "elo", "bayes"]},
            _subset_weights(w_all, ["hgbr", "elo", "bayes"]),
        ),
        "D_dc": fuse_probs(
            {k: members[k] for k in ["hgbr", "elo", "bayes", "dc"]},
            _subset_weights(w_all, ["hgbr", "elo", "bayes", "dc"]),
        ),
        "E_nb": fuse_probs(
            {k: members[k] for k in GOAL_ORDER}, _subset_weights(w_all, GOAL_ORDER)
        ),
        "F_gbm": fuse_goal_outcome(goal, gbm, w_all),
    }
    # G:生产 prior_blend(blend_matrix,与线上同一实现) —— 不再手工 α=0.6
    pf = out["F_gbm"]
    from app.prediction.prior_blend import blend_matrix as _prod_blend

    try:
        m2, _info = _prod_blend(league_id, match_dt, list(pf), np.asarray(raw_matrix))
        pG = (
            tuple(
                float(x)
                for x in (
                    np.asarray(m2)[np.tril_indices(len(m2), -1)].sum(),
                    np.trace(np.asarray(m2)),
                    np.asarray(m2)[np.triu_indices(len(m2), 1)].sum(),
                )
            )
            if m2 is not None
            else pf
        )
    except Exception:
        pG = pf
    out["G_prior"] = pG
    # H:生产 calibration(当前 artifact;逐赛季重拟合见 caveat)
    out["H_calib"] = _calibrate(pG, LeagueType.PREMIER_LEAGUE)
    return out


def _calibrate(pv, lt: LeagueType) -> tuple:
    from app.prediction import calibration as cal

    fake = {
        "home_win_probability": pv[0],
        "draw_probability": pv[1],
        "away_win_probability": pv[2],
    }
    out, _i, _d = cal.apply(fake, str(MODELS_DIR), lt)
    return (
        out["home_win_probability"],
        out["draw_probability"],
        out["away_win_probability"],
    )


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
    ap.add_argument("--out", default="/opt/data/ablation_wf_report.json")
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
    comps = [
        "A_hgbr",
        "B_elo",
        "C_bayes",
        "D_dc",
        "E_nb",
        "F_gbm",
        "G_prior",
        "H_calib",
    ]
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
            for comp, pv in probs_c.items():
                d = report[year][comp]
                d["n"] += 1
                d["ll"] += log_loss(pv, actual)
                d["brier"] += brier_score(pv, actual)
                d["rps"] += rps(pv, actual)
                d["pvecs"].append(list(pv))
                d["acts"].append(actual)
                # 每组件自算 xG(自身责任矩阵)
                mtx = _component_matrix(int_, comp)
                if mtx is not None:
                    xh, xa = _matrix_xg(mtx)
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
            row[cc] = {
                "n": n,
                "ll": round(d["ll"] / n, 5),
                "brier": round(d["brier"] / n, 5),
                "rps": round(d["rps"] / n, 5),
                "ece": round(float(ece(d["pvecs"], d["acts"])), 5),
                "xgm_h": round(d["xgm_h"] / n, 4),
                "xgm_a": round(d["xgm_a"] / n, 4),
            }
        summary[s] = row
    payload = {
        "league": args.league,
        "seasons": summary,
        "method": "strict-walkforward-no-future-model",
        "secs": round(time.time() - t0, 1),
        "matches": total,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n===== Strict W-F Ablation: {args.league} ({total} 场) =====")
    for s in seasons:
        for cc in comps:
            r_ = summary[s].get(cc, {})
            if r_.get("n", 0) == 0:
                continue
            print(
                f"{s}  {cc:>7} n={r_['n']:>4} LL={r_['ll']:.4f} B={r_['brier']:.4f} "
                f"RPS={r_['rps']:.4f} ECE={r_['ece']:.4f} xG_h={r_['xgm_h']:.3f} xG_a={r_['xgm_a']:.3f}"
            )
    print("✅ 报告:", args.out)


def _component_matrix(internal: dict, comp: str):
    """每组件自算矩阵(成员 λ → 各自矩阵 → 子集权重融合 → 自身 xG)。

    A hgbr → _pois(λ_h);B +elo → pois 子集融合;…E 全 goal 融合;D/E 引入
    _dc/_nb。F/G/H 与 E 同 λ 梯度(GBM 是 outcome 层,不进矩阵)。
    """
    from app.models.distributions import pois_matrix as _pois
    from app.models.ensemble.matrix import _dc_matrix, _nb_matrix

    lam = internal.get("member_lambdas") or {}
    if not lam:
        return None  # lambdas 不全无法可靠重建

    def _safe(t):
        return t if t is not None else (1.5, 1.4)

    hg = _safe(lam.get("hgbr"))
    el = _safe(lam.get("elo"))
    ba = _safe(lam.get("bayes"))
    dc = _safe(lam.get("dc"))
    nb = _safe(lam.get("nb"))
    w = internal.get("member_weights") or {}
    mats = {
        "hgbr": _pois(hg[0], hg[1]),
        "elo": _pois(el[0], el[1]),
        "bayes": _pois(ba[0], ba[1]),
        "dc": _dc_matrix(dc[0], dc[1], internal.get("tau", 0.0)),
        "nb": _nb_matrix(nb[0], nb[1], internal.get("phi", 1e9)),
    }
    members_of = {
        "A_hgbr": ["hgbr"],
        "B_elo": ["hgbr", "elo"],
        "C_bayes": ["hgbr", "elo", "bayes"],
        "D_dc": ["hgbr", "elo", "bayes", "dc"],
        "E_nb": ["hgbr", "elo", "bayes", "dc", "nb"],
        "F_gbm": ["hgbr", "elo", "bayes", "dc", "nb"],
        "G_prior": ["hgbr", "elo", "bayes", "dc", "nb"],
        "H_calib": ["hgbr", "elo", "bayes", "dc", "nb"],
    }
    names = members_of.get(comp)
    if not names:
        return None
    wsum = sum(w.get(n, 0.0) for n in names)
    if wsum <= 0:
        return None
    out = np.zeros_like(mats[names[0]])
    for n in names:
        out = out + (w.get(n, 0.0) / wsum) * np.asarray(mats[n])
    return out


if __name__ == "__main__":
    main()
