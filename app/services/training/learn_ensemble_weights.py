"""P2-A:Ensemble 权重滚动学习(§4 严格:四成员概率层 + τ/φ 拟合)。

对每联赛抽样历史完赛场次(2024 起),独立计算 A/HGBR、B/DC、C/NB、D/ELO-Goal
四成员的三分类概率(与预测链路同口径、防泄漏),先拟合 τ(fit_dc_tau)与
φ(fit_nb_phi),再以 SLSQP 约束(w≥0, Σw=1)最小化 log-loss 学习权重,
写入 artifacts/ensemble/ 下(§42)。
"""
import json
import os
import sys

_ROOT = str(__import__('app.core.paths', fromlist=['PROJECT_ROOT']).PROJECT_ROOT)
for _p in (os.path.join(_ROOT, "src"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

from app.api.db import League, Match, init_db, session_scope
from app.core.config import LeagueType
from app.data.adapters import matches_to_dataframe
from app.models.ensemble import (
    dc_probs,
    elo_goal_lambda,
    fit_nb_phi,
    learn_weights,
    match_probs,
    nb_probs,
)
from app.models.loader import _load_model

# §4 权重学习:时间滚动窗口(最近 500 场),窗口内均匀采样(默认 150,上限 500)
WINDOW_SIZE = 500
SAMPLE_PER_LEAGUE = 150
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(_ROOT, "app", "models"))
W_OUT = os.path.join(_ROOT, "artifacts", "ensemble", "ensemble_weights.json")
P_OUT = os.path.join(_ROOT, "artifacts", "ensemble", "dc_nb_params.json")


def main():
    from app.services.cli import add_log_level_arg, make_parser, setup_logging
    ap = make_parser("Ensemble 四成员权重滚动学习(τ/φ 拟合 + SLSQP log-loss 优化)")
    add_log_level_arg(ap)
    args = ap.parse_args()
    setup_logging(args.log_level)

    init_db()
    weights_out, params_out = {}, {}
    with session_scope():
        for lt in [LeagueType.PREMIER_LEAGUE, LeagueType.LA_LIGA, LeagueType.BUNDESLIGA,
                   LeagueType.SERIE_A, LeagueType.LIGUE_1]:
            league = League.query.filter_by(league_type=lt.value).first()
            if league is None:
                print(f"  {lt.value}: 无联赛记录,跳过")
                continue
            model = _load_model(lt, MODELS_DIR)
            matches = Match.query.filter_by(league_id=league.id, match_status="finished").all()
            matches.sort(key=lambda m: str(m.match_date or ""))
            # 时间滚动窗口:最近 WINDOW_SIZE 场(§4)
            cand = matches[-WINDOW_SIZE:]
            rng = np.random.default_rng(42)
            idx = rng.choice(len(cand), size=min(SAMPLE_PER_LEAGUE, len(cand)), replace=False)
            raw_samples, done = [], 0
            for i in idx:
                m = cand[int(i)]
                match_dt = pd.Timestamp(m.match_date)
                history = [x for x in matches if pd.Timestamp(x.match_date) < match_dt]
                if len(history) < 50:
                    continue
                hist_df = matches_to_dataframe(history)
                # §18:ELO 由模型 prepare(factory)注入
                if "attack_elo_diff" not in hist_df.columns:
                    continue
                att_diff = float(hist_df["attack_elo_diff"].iloc[-1])
                lams = []
                for _row in ({"date": match_dt, "home_team": m.home_team, "away_team": m.away_team,
                              "home_goals": np.nan, "away_goals": np.nan, "goals": np.nan,
                              "league": league.name, "season": league.season or ""},
                             {"date": match_dt, "home_team": m.away_team, "away_team": m.home_team,
                              "home_goals": np.nan, "away_goals": np.nan, "goals": np.nan,
                              "league": league.name, "season": league.season or ""}):
                    _df = pd.concat([hist_df, pd.DataFrame([_row])], ignore_index=True)
                    lams.append(float(model.predict(_df)["predictions"][-1]))
                lam_h, lam_a = lams
                raw_samples.append({
                    "hgbr_lam_h": lam_h, "hgbr_lam_a": lam_a, "att_diff": att_diff,
                    "home_goals": m.home_goals, "away_goals": m.away_goals,
                    "actual": 0 if m.home_goals > m.away_goals
                              else (1 if m.home_goals == m.away_goals else 2),
                    "_df": _df,  # 供 gbm 成员特征复用
                })
                done += 1
            if not raw_samples:
                continue
            # 1) τ/φ 拟合(数据驱动)
            tau = _fit_tau(raw_samples)
            phi = fit_nb_phi(raw_samples)
            params_out[lt.value] = {"tau": tau, "phi": phi}
            # 2) 五成员概率样本(含 gbm)
            from app.models.ensemble.gbm import GbmClassifier
            _gbm = GbmClassifier.load(os.path.join(MODELS_DIR, f"{lt.value}_gbm.pkl"))
            samples = []
            for s in raw_samples:
                lam_eh = elo_goal_lambda(s["att_diff"], True)
                lam_ea = elo_goal_lambda(s["att_diff"], False)
                gprob = [0.0, 0.0, 0.0]
                if _gbm is not None:
                    try:
                        _gfeat = model.prepare_features(s["_df"])
                        _gcols = [col for col in _gbm.feature_columns_ if col in _gfeat.columns]
                        gprob = list(_gbm.predict_proba(_gfeat[_gcols].iloc[[-1]])[0])
                    except Exception:
                        pass
                samples.append({
                    "hgbr": list(match_probs(s["hgbr_lam_h"], s["hgbr_lam_a"])),
                    "dc": list(dc_probs(s["hgbr_lam_h"], s["hgbr_lam_a"], tau)),
                    "nb": list(nb_probs(s["hgbr_lam_h"], s["hgbr_lam_a"], phi)),
                    "elo": list(match_probs(lam_eh, lam_ea)),
                    "gbm": gprob,
                    "actual": s["actual"],
                })
            w = learn_weights(samples, tau, phi)
            weights_out[lt.value] = {k: round(v, 4) for k, v in w.items()}
            print(f"  {lt.value}: τ={tau:.3f} φ={phi:.1f} w={ {k: round(v,3) for k,v in w.items() if k!='log_loss'} } "
                  f"ll={w['log_loss']:.4f} n={len(samples)}", flush=True)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(W_OUT, "w", encoding="utf-8") as _wf:
        json.dump(weights_out, _wf, ensure_ascii=False, indent=2)
    with open(P_OUT, "w", encoding="utf-8") as _pf:
        json.dump(params_out, _pf, ensure_ascii=False, indent=2)
    print(f"\n✅ 权重: {W_OUT} | τ/φ: {P_OUT}")


def _fit_tau(samples):
    """τ 拟合:在 (0,0)/(0,1)/(1,0)/(1,1) 低比分格点上最大似然。"""
    best_t, best_ll = 0.0, float("inf")
    for t in np.arange(-0.2, 0.201, 0.01):
        ll = 0.0
        for s in samples:
            x, y = s["home_goals"], s["away_goals"]
            if x > 1 or y > 1:
                continue
            from app.models.distributions import pois_pmf as _pois_pmf
            from app.models.dixon_coles.dc import _dc_tau
            p = _dc_tau(x, y, s["hgbr_lam_h"], s["hgbr_lam_a"], t) * \
                _pois_pmf(s["hgbr_lam_h"], x) * _pois_pmf(s["hgbr_lam_a"], y)
            ll += np.log(max(1e-12, p))
        if ll < best_ll:
            best_t, best_ll = t, ll
    return float(best_t)


if __name__ == "__main__":
    from app.services.cli import run
    raise SystemExit(run(main))
