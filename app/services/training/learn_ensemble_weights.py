"""P2-A:Ensemble 权重滚动学习 —— 时间分段 OOF(审查 P0-3/P0-4,2026-08 重写)。

原实现用"当前全量 production 模型"回头预测历史完赛场次再学习权重,
属二级时间泄漏 —— 该模型已见过这些场次结果,学出的权重偏乐观。
本次重写为时间分段 OOF(Out-Of-Fold):

  段0 ──训练──> 段1 ──训练(段0+1)──> 段2 ──训练(段0..2)──> 段3
               │                    │                     │
           预测段1               预测段2                预测段3

- 每段用"该段之前"的数据训练临时 HGBR 与 GBM(只见过前缀数据);
- 段内采样场次的预测仅使用"截止该场"的历史(严格的赛前视角);
- 收集 OOF 概率样本 → 拟合 τ/φ → SLSQP 学习权重(GBM 不可用则移除)。

P0-4 修复:ELO 由 prepare(factory 注入)提供,att_diff 读 prepare 后的主队行;
GBM 从 artifacts/<league>/gbm.pkl 统一路径加载。
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
from app.models.distributions import pois_pmf as _pois_pmf
from app.models.dixon_coles.dc import _dc_tau
from app.models.ensemble import (
    dc_probs,
    elo_goal_lambda,
    fit_nb_phi,
    learn_weights,
    match_probs,
    nb_probs,
)
from app.models.ensemble.gbm import GbmClassifier
from app.services.training.model_trainer import ModelTrainer

# §4 权重学习:时间分段 OOF(每段采样场次,上限 SAMPLE_PER_SEG×K)
K_SEG = 4                      # 时间段数
SAMPLE_PER_SEG = 40            # 每段最大采样场次
MIN_PREFIX_ROWS = 100          # 前缀最少行数(联赛阈值)
MIN_HISTORY = 50               # 单场最少历史
_AD = str(__import__("app.core.paths", fromlist=["ARTIFACTS_DIR"]).ARTIFACTS_DIR)
W_OUT = os.path.join(_AD, "ensemble", "ensemble_weights.json")
P_OUT = os.path.join(_AD, "ensemble", "dc_nb_params.json")


def _outcome(hg, ag):
    return 0 if hg > ag else (1 if hg == ag else 2)


def _train_temp_model(prefix_df: pd.DataFrame, lt: LeagueType):
    """用前缀数据训练临时模型(OOF 口径:模型只见过该段之前的数据)。"""
    mt = ModelTrainer()
    mt.train_model(prefix_df, lt, cross_validation=False)
    return mt.model


def _train_temp_gbm(prefix_df: pd.DataFrame, prefix_df_prepared, model) -> GbmClassifier | None:
    """训练段前 GBM(与主模型同段前缀,内部 80/20 + 100% 重训)。"""
    try:
        y = prefix_df.apply(lambda r: _outcome(r["home_goals"], r["away_goals"]), axis=1)
        gcols = [c for c in model.feature_columns_ if c in prefix_df_prepared.columns]
        _gbm = GbmClassifier()
        _gbm.train(prefix_df_prepared[gcols], y)
        return _gbm
    except Exception as e:
        print(f"    [warn] 段前 GBM 训练失败(该段无 gbm 成员): {e}", flush=True)
        return None


def main():
    from app.services.cli import add_log_level_arg, make_parser, setup_logging
    ap = make_parser("Ensemble 权重学习(时间分段 OOF:段前训练 → 段内预测 → SLSQP)")
    add_log_level_arg(ap)
    args = ap.parse_args()
    setup_logging(args.log_level)

    init_db()
    weights_out, params_out, meta_out = {}, {}, {}
    with session_scope():
        for lt in [LeagueType.PREMIER_LEAGUE, LeagueType.LA_LIGA, LeagueType.BUNDESLIGA,
                   LeagueType.SERIE_A, LeagueType.LIGUE_1]:
            league = League.query.filter_by(league_type=lt.value).first()
            if league is None:
                print(f"  {lt.value}: 无联赛记录,跳过")
                continue
            matches = Match.query.filter_by(league_id=league.id, match_status="finished").all()
            matches.sort(key=lambda m: str(m.match_date or ""))
            n = len(matches)
            if n < 200:
                print(f"  {lt.value}: 仅 {n} 场,不足 200,跳过")
                continue
            boundaries = [int(n * k / K_SEG) for k in range(1, K_SEG)] + [n]
            oof_samples, seg_done = [], 0
            for k in range(1, K_SEG):
                seg_start, seg_end = boundaries[k - 1], boundaries[k]
                seg = matches[seg_start:seg_end]
                prefix = matches[:seg_start]
                if len(seg) < 15 or len(prefix) < MIN_PREFIX_ROWS:
                    continue
                print(f"  {lt.value}: 段{k}/{K_SEG-1} 前缀 {len(prefix)} 场 → 训练临时模型...", flush=True)
                prefix_df = matches_to_dataframe(prefix)
                temp_model = _train_temp_model(prefix_df, lt)
                prefix_prepared = temp_model.prepare_features(temp_model._sort_by_date(prefix_df))
                _gbm = _train_temp_gbm(prefix_df, prefix_prepared, temp_model)
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
                    rows = [{"date": match_dt, "home_team": m.home_team, "away_team": m.away_team,
                             "home_goals": np.nan, "away_goals": np.nan, "goals": np.nan,
                             "league": league.name, "season": league.season or ""},
                            {"date": match_dt, "home_team": m.away_team, "away_team": m.home_team,
                             "home_goals": np.nan, "away_goals": np.nan, "goals": np.nan,
                             "league": league.name, "season": league.season or ""}]
                    _df = pd.concat([hist_df, pd.DataFrame(rows)], ignore_index=True)
                    try:
                        _df = temp_model._sort_by_date(_df)
                        feats = temp_model.prepare_features(_df)
                        fcols = [c for c in temp_model.feature_columns_ if c in feats.columns]
                        lams = temp_model.model.predict(feats[fcols])
                        lam_h, lam_a = float(lams[-2]), float(lams[-1])
                        att_diff = float(feats["attack_elo_diff"].iloc[-2])  # P0-4:prepare 后取主队行
                    except Exception:
                        continue
                    gprob = None
                    if _gbm is not None:
                        try:
                            _gc = [c for c in _gbm.feature_columns_ if c in feats.columns]
                            gprob = list(_gbm.predict_proba(feats[_gc].iloc[[-2]])[0])
                        except Exception:
                            gprob = None
                    oof_samples.append({
                        "hgbr_lam_h": lam_h, "hgbr_lam_a": lam_a, "att_diff": att_diff,
                        "home_goals": m.home_goals, "away_goals": m.away_goals,
                        "actual": _outcome(m.home_goals, m.away_goals),
                        "gbm": gprob,
                    })
                    got += 1
                print(f"    → 段{k} 采样 {got} 场", flush=True)
                seg_done += 1
            if len(oof_samples) < 30:
                print(f"  {lt.value}: OOF 样本仅 {len(oof_samples)}(<30),跳过权重学习")
                continue
            # 1) τ/φ 拟合(OOF 概率)
            tau = _fit_tau(oof_samples)
            phi = fit_nb_phi(oof_samples)
            params_out[lt.value] = {"tau": tau, "phi": phi}
            # 2) 成员概率样本(各成员均来自 OOF 预测)
            samples = []
            for s in oof_samples:
                lam_eh = elo_goal_lambda(s["att_diff"], True)
                lam_ea = elo_goal_lambda(s["att_diff"], False)
                rec = {
                    "hgbr": list(match_probs(s["hgbr_lam_h"], s["hgbr_lam_a"])),
                    "dc": list(dc_probs(s["hgbr_lam_h"], s["hgbr_lam_a"], tau)),
                    "nb": list(nb_probs(s["hgbr_lam_h"], s["hgbr_lam_a"], phi)),
                    "elo": list(match_probs(lam_eh, lam_ea)),
                    "actual": s["actual"],
                }
                if s.get("gbm") is not None:
                    rec["gbm"] = s["gbm"]   # P1-16:不可用时不移入样本
                samples.append(rec)
            w = learn_weights(samples, tau, phi)
            weights_out[lt.value] = {k: round(v, 4) for k, v in w.items() if k not in ("log_loss", "n")}
            meta_out[lt.value] = {"log_loss": round(w["log_loss"], 4), "n": len(samples),
                                  "segments": seg_done, "method": "time-segmented-oof",
                                  "k_seg": K_SEG}
            print(f"  {lt.value}: τ={tau:.3f} φ={phi:.1f} w={ {k: round(v,3) for k,v in weights_out[lt.value].items()} } "
                  f"ll={w['log_loss']:.4f} n={len(samples)}", flush=True)

    os.makedirs(os.path.dirname(W_OUT), exist_ok=True)
    with open(W_OUT, "w", encoding="utf-8") as _wf:
        json.dump(weights_out, _wf, ensure_ascii=False, indent=2)
    with open(P_OUT, "w", encoding="utf-8") as _pf:
        json.dump(params_out, _pf, ensure_ascii=False, indent=2)
    with open(os.path.join(os.path.dirname(W_OUT), "oof_meta.json"), "w", encoding="utf-8") as _mf:
        json.dump(meta_out, _mf, ensure_ascii=False, indent=2)
    print(f"\n✅ 权重: {W_OUT} | τ/φ: {P_OUT} | meta: oof_meta.json")


def _fit_tau(samples):
    """τ 拟合:在 (0,0)/(0,1)/(1,0)/(1,1) 低比分格点上最大似然。"""
    best_t, best_ll = 0.0, float("inf")
    for t in np.arange(-0.2, 0.201, 0.01):
        ll = 0.0
        for s in samples:
            x, y = s["home_goals"], s["away_goals"]
            if x > 1 or y > 1:
                continue
            p = _dc_tau(x, y, s["hgbr_lam_h"], s["hgbr_lam_a"], t) * \
                _pois_pmf(s["hgbr_lam_h"], x) * _pois_pmf(s["hgbr_lam_a"], y)
            ll += np.log(max(1e-12, p))
        if ll < best_ll:
            best_t, best_ll = t, ll
    return float(best_t)


if __name__ == "__main__":
    from app.services.cli import run
    raise SystemExit(run(main))
