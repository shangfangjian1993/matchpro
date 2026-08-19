"""Walk-Forward 组件 Ablation(审查 ac2196b §14/§21)。

目标:逐赛季(2022/23→2025/26)逐组件梯,输出每格 LogLoss/Brier/RPS/ECE/xG-MAE,
回答"哪个组件真实贡献预测能力":
  A = HGBR only
  B = A + ELO
  C = B + Bayes
  D = C + DC(分布层)
  E = D + NB(分布层)= Goal 全融合
  F = E + GBM(Outcome 两层融合)
  G = F + Prior Blend(近期频率)
H = G + Calibration

方法:复用生产同链路(engine.predict 全链一次/场),再从 _internal 的成员
分解重算各组件 1X2 —— 不重跑模型、不改生产逻辑。权重 = OOF 学习权重子集。
注意:模型为**当前已训版本**(非历史上当季模型),衡量"当前模型下各组件增量"。
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd

_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.api.db import League, Match, init_db
from app.core.config import LeagueType
from app.core.paths import MODELS_DIR
from app.models.ensemble.fusion import fuse_goal_outcome, fuse_probs
from app.prediction.context import ContextBuilder
from app.prediction.engine import PredictionEngine
from app.replay.metrics import brier_score, ece, log_loss, rps

GOAL_ORDER = ["hgbr", "dc", "nb", "elo", "bayes"]


def _subset_weights(w_all: dict, names) -> dict:
    """取子集并归一化为目标层权重(保持 OOF 比例)。"""
    w = {n: w_all.get(n, 0.0) for n in names}
    tot = sum(w.values())
    if tot <= 0:
        return {names[0]: 1.0}
    return {n: v / tot for n, v in w.items()}


def _component_probs(internal: dict) -> dict[str, tuple]:
    """逐组件 1X2(生产全链已跑,从成员分解重算)。"""
    members = internal["members"]
    w_all = internal["member_weights"]
    gbm = internal.get("gbm_probs")
    goal = fuse_probs(members, w_all)
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
    return out


def main():
    ap = argparse.ArgumentParser(description="walk-forward 组件 ablation")
    ap.add_argument("--league", default="premier_league")
    ap.add_argument(
        "--seasons", default="2022,2023,2024,2025", help="起始年份(按 8 月切),逗号分隔"
    )
    ap.add_argument("--limit", type=int, default=0, help="每赛季最大场数(冒烟用)")
    ap.add_argument("--out", default="/opt/data/ablation_report.json")
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
    if len(matches) < 200:
        print(f"数据不足 {len(matches)}")
        return

    builder = ContextBuilder(str(MODELS_DIR))
    engine = PredictionEngine(str(MODELS_DIR))
    seasons = [int(s) for s in args.seasons.split(",")]

    # 结果: season -> component -> metrics dict
    report = {
        s: {
            c: {
                "n": 0,
                "ll": 0.0,
                "brier": 0.0,
                "rps": 0.0,
                "pvecs": [],
                "acts": [],
                "xgm_h": 0.0,
                "xgm_a": 0.0,
            }
            for c in [
                "A_hgbr",
                "B_elo",
                "C_bayes",
                "D_dc",
                "E_nb",
                "F_gbm",
                "G_prior",
                "H_calib",
            ]
        }
        for s in seasons
    }

    def _season_of(m):
        y = pd.Timestamp(m.match_date).year
        mth = pd.Timestamp(m.match_date).month
        start = y - 1 if mth < 8 else y
        return start

    t0 = time.time()
    done = 0
    for m in matches:
        start = _season_of(m)
        if start not in seasons:
            continue
        md = pd.Timestamp(m.match_date)
        try:
            ctx = builder.build(lt, m.home_team, m.away_team, md, hist_limit=500)
            result = engine.predict(ctx)
            internal = result["_internal"]
        except Exception as _e:
            print(f"[skip] {m.home_team} vs {m.away_team} {md}: {_e}", file=sys.stderr)
            continue
        actual = (
            0
            if (m.home_goals or 0) > (m.away_goals or 0)
            else (1 if (m.home_goals or 0) == (m.away_goals or 0) else 2)
        )
        pvecs_c = _component_probs(internal)
        # G/H:基于 F
        pf = pvecs_c["F_gbm"]
        freqs = _recent_freqs(league.id, md)
        pG = pf
        if freqs is not None:
            a = 0.6
            pG = tuple(a * x + (1 - a) * y for x, y in zip(pf, freqs))
            s = sum(pG)
            pG = tuple(x / s for x in pG)
        pvecs_c["G_prior"] = pG
        pvecs_c["H_calib"] = _calibrate(pG, lt)

        for comp, pv in pvecs_c.items():
            d = report[start][comp]
            d["n"] += 1
            d["ll"] += log_loss(pv, actual)
            d["brier"] += brier_score(pv, actual)
            d["rps"] += rps(pv, actual)
            d["pvecs"].append(list(pv))
            d["acts"].append(actual)
        # xG MAE:各组件共享同一 λ/最终矩阵期望的进失球 MAE(标注见表尾)
        fx = result.get("expected_xg") or [
            internal["home_lambda"],
            internal["away_lambda"],
        ]
        gm_h = abs(fx[0] - (m.home_goals or 0))
        gm_a = abs(fx[1] - (m.away_goals or 0))
        for comp in pvecs_c:
            report[start][comp]["xgm_h"] += gm_h
            report[start][comp]["xgm_a"] += gm_a
        done += 1
        if args.limit and done >= args.limit:
            break

    # 汇总:每赛季每组件 指标
    summary = {}
    for s in seasons:
        row = {}
        for comp, d in report[s].items():
            if d["n"] == 0:
                row[comp] = {"n": 0}
                continue
            n = d["n"]
            e = ece(d["pvecs"], d["acts"])
            row[comp] = {
                "n": n,
                "ll": round(d["ll"] / n, 5),
                "brier": round(d["brier"] / n, 5),
                "rps": round(d["rps"] / n, 5),
                "ece": round(float(e), 5),
                "xgm_h": round(d["xgm_h"] / n, 4),
                "xgm_a": round(d["xgm_a"] / n, 4),
            }
        summary[s] = row
    payload = {
        "league": args.league,
        "seasons": summary,
        "secs": round(time.time() - t0, 1),
        "matches": done,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n===== Ablation: {args.league} ({done} 场) =====")
    hdr = ["赛季", "组件", "n", "LL", "Brier", "RPS", "ECE", "xG_h", "xG_a"]
    print("  ".join(f"{h:>6}" for h in hdr))
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
    for s in seasons:
        for comp in comps:
            r_ = summary[s].get(comp, {})
            if r_.get("n", 0) == 0:
                continue
            print(
                f"{s:>6}  {comp:>7} {r_['n']:>6} "
                f"{r_['ll']:>6.4f} {r_['brier']:>7.4f} {r_['rps']:>7.4f} "
                f"{r_['ece']:>7.4f} {r_['xgm_h']:>6.3f} {r_['xgm_a']:>6.3f}"
            )
    print("\n✅ 报告:", args.out)


def _recent_freqs(league_id, md):
    rows = (
        Match.query.filter(
            Match.league_id == league_id,
            Match.match_status == "finished",
            Match.match_date < md,
        )
        .order_by(Match.match_date.desc())
        .limit(100)
        .all()
    )
    if len(rows) < 20:
        return None
    n = len(rows)
    h = sum(1 for x in rows if (x.home_goals or 0) > (x.away_goals or 0))
    d = sum(1 for x in rows if (x.home_goals or 0) == (x.away_goals or 0))
    return [h / n, d / n, (n - h - d) / n]


def _calibrate(pv, lt: LeagueType) -> tuple:
    from app.prediction import calibration as cal

    fake = {
        "home_win_probability": pv[0],
        "draw_probability": pv[1],
        "away_win_probability": pv[2],
    }
    out, _info, _degraded = cal.apply(fake, str(MODELS_DIR), lt.value)
    return (
        out["home_win_probability"],
        out["draw_probability"],
        out["away_win_probability"],
    )


if __name__ == "__main__":
    main()
