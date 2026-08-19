"""ContextBuilder(审查九 P1-8 拆分):预测上下文构建。

职责:球队解析/校验、历史查询、预测行构造、伤停信号(Info Layer +
failure taxonomy)、bayes λ、att_diff —— 输出 context dict 供
PredictionEngine 消费(生产/回测/OOF 同一入口)。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.api.db import League, Match
from app.data.adapters import matches_to_dataframe
from app.data.canonical.team_names_zh import to_en, to_zh
from app.models.loader import _load_model

logger = logging.getLogger(__name__)


class ContextBuilder:
    """构建预测上下文(数据准备,不包含模型推理)。"""

    def __init__(self, models_dir: str):
        self.models_dir = models_dir

    def build(
        self,
        league_type,
        home_team: str,
        away_team: str,
        match_date=None,
        model=None,
        hist_limit: int | None = None,
    ) -> dict:
        """model: 可选注入(回测滚动模型/测试用);默认加载 active 模型。

        hist_limit: 可选截断(回测性能用,最近 N 场;滚动特征已收敛,无影响)。
        """

        if not home_team or not away_team:
            raise ValueError("缺少主队或客队名称")
        if home_team == away_team:
            raise ValueError("主队和客队不能相同")
        home_team = to_en(home_team)
        away_team = to_en(away_team)
        # 审查 A70A601 §十五 + A 专项:队名归一化 —— 内存层统一规范名
        # ('AFC Bournemouth' 与 'Bournemouth' 合并为同 key),否则 FC/AFC
        # 后缀分裂令 known_teams 定位失败、特征 per-team 分组撕裂。
        from app.data.canonical.team_names import canonical_en

        home_team = canonical_en(home_team)
        away_team = canonical_en(away_team)

        league = League.query.filter_by(league_type=league_type.value).first()
        if league is None:
            raise ValueError(f"数据库中还没有 {league_type.value} 的联赛数据")

        model = model or _load_model(league_type, self.models_dir)
        matched = (
            Match.query.filter_by(
                league_id=league.id, home_team=home_team, away_team=away_team
            )
            .order_by(Match.match_date.desc())
            .first()
        )
        matched_match_id = matched.id if matched else None
        match_dt = pd.Timestamp(match_date) if match_date else pd.Timestamp.now()

        history = Match.query.filter_by(
            league_id=league.id, match_status="finished"
        ).all()
        if match_dt is not None:
            history = [m for m in history if pd.Timestamp(m.match_date) < match_dt]
        # 审查 ac2196b 关联修复:默认行序非时间序(rowid 受迁移/批量导入打乱),
        # 必须显式按比赛时间升序 —— 否则 hist_limit 截取到任意错乱窗口(曾取到
        # 1992 年段),既有时序泄漏风险(窗口含未来场次),又令球队定位失败。
        history.sort(key=lambda x: pd.Timestamp(x.match_date))
        if hist_limit is not None and len(history) > hist_limit:
            history = history[-hist_limit:]
        hist_df = matches_to_dataframe(
            history, league_name=league.name, league_season=league.season or ""
        )
        if hist_df.empty:
            raise ValueError("数据库中没有历史比赛数据,无法构造赛前特征")
        # 队名归一化:历史 two 份格式(FC/AFC)统一为同一规范 key,保证
        # 已知队集合与 per-team 特征分组一致(内存层,不改库/快照)
        _ce = __import__(
            "app.data.canonical.team_names", fromlist=["canonical_en"]
        ).canonical_en
        hist_df = hist_df.assign(
            home_team=hist_df["home_team"].map(_ce),
            away_team=hist_df["away_team"].map(_ce),
        )

        known_teams = set(hist_df["home_team"]) | set(hist_df["away_team"])
        unknown = [t for t in (home_team, away_team) if t not in known_teams]
        if unknown:
            # 审查 §8:全球分层模型为半成品分支,已删除;需要时再恢复。
            raise ValueError(
                f"球队「{'、'.join(unknown)}」在 {league_type.value} 的历史数据中不存在,"
                f"请先录入其比赛数据"
            )

        home_row = {
            "date": match_dt,
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": np.nan,
            "away_goals": np.nan,
            "goals": np.nan,
            "league": league.name,
            "season": league.season or "",
        }
        away_row = {
            "date": match_dt,
            "home_team": away_team,
            "away_team": home_team,
            "home_goals": np.nan,
            "away_goals": np.nan,
            "goals": np.nan,
            "league": league.name,
            "season": league.season or "",
        }
        _m = model[1] if isinstance(model, tuple) else model
        _pred_df = pd.concat(
            [hist_df, pd.DataFrame([home_row, away_row])], ignore_index=True
        )
        _raw = _m.predict(_pred_df)
        home_lambda, away_lambda = (
            float(_raw["predictions"][-2]),
            float(_raw["predictions"][-1]),
        )

        _degraded_components: list = []
        _failure_codes: list = []
        # 特征(含 ELO)与 att_diff —— 审查 21cd12b P1-4:异常细分,
        # 不再用 except Exception → 0 掩盖核心特征计算失败
        from app.core.exceptions import (
            FeatureComputationError,
            FeatureSchemaError,
        )

        _feat, _att_diff = None, 0.0
        try:
            _feat = _m.prepare_features(_pred_df)
        except Exception as _fe:
            raise FeatureComputationError(f"特征计算失败: {_fe}") from _fe
        if "attack_elo_diff" in _feat.columns:
            try:
                _att_diff = float(_feat["attack_elo_diff"].iloc[-2])
            except (TypeError, ValueError, IndexError) as _se:
                raise FeatureSchemaError(f"ELO 特征异常(schema/取值): {_se}") from _se
        else:
            # ELO 列缺失属数据可用性(可降级),非 schema 错 —— 记录组件
            _degraded_components.append("elo")
            _failure_codes.append("ELO_FEATURE_UNAVAILABLE")

        # 伤停 Info Layer + failure taxonomy(审查二十四)+ 阵容强度(P1-11)
        _h_mult, _a_mult = 1.0, 1.0
        _lineup = None
        try:
            import os as _os

            from app.data.sources.injuries.collector import InjuriesCollector
            from app.data.sources.injuries.signals import (
                injuries_to_signals,
                signal_brief,
            )
            from app.prediction.info_fusion import signals_to_adjust
            from app.prediction.lineup import build_strength_vector

            _ic = InjuriesCollector()
            _day = match_dt.strftime("%Y-%m-%d")
            _cache = _os.path.join(_ic.cache_dir, f"date_{_day}.json")
            if _os.path.exists(_cache):
                _recs = _ic.fetch_by_date(_day, use_cache=True)
                if not _recs:
                    _degraded_components.append("injury")
                    _failure_codes.append("INJURY_EMPTY")
                else:
                    _sig = injuries_to_signals(
                        _ic.filter_by_team(_recs, home_team),
                        _ic.filter_by_team(_recs, away_team),
                    )
                    _lineup = build_strength_vector(_sig)
                    if _sig["sources"]:
                        _h_mult, _a_mult = signals_to_adjust(_sig)
                        logger.info(
                            "伤停融合 %s: %s",
                            _day,
                            signal_brief(_sig).replace("\n", " | "),
                        )
            else:
                _degraded_components.append("injury")
                _failure_codes.append("INJURY_CACHE_MISSING")
        except Exception as _ie:
            _degraded_components.append("injury")
            _failure_codes.append("INJURY_PARSE_ERROR")
            logger.warning("伤停层降级: %s", _ie)

        # bayes λ(审查 P1-10)
        _lam_bh = _lam_ba = None
        try:
            from app.models.bayes_team import bayes_lambda

            _lam_bh, _lam_ba = bayes_lambda(hist_df, home_team, away_team)
        except Exception:
            _lam_bh = _lam_ba = None

        return {
            "league_type": league_type,
            "league": league,
            "home_team": home_team,
            "away_team": away_team,
            "home_team_zh": to_zh(home_team),
            "away_team_zh": to_zh(away_team),
            "match_dt": match_dt,
            "match_id": matched_match_id,
            "hist_df": hist_df,
            "model": model,
            "_m": _m,
            "_pred_df": _pred_df,
            "home_lambda": home_lambda,
            "away_lambda": away_lambda,
            "_feat": _feat,
            "_att_diff": _att_diff,
            "injury_mult": (_h_mult, _a_mult),
            "lineup": _lineup,
            "bayes_lam_h": _lam_bh,
            "bayes_lam_a": _lam_ba,
            "degraded_components": _degraded_components,
            "failure_codes": _failure_codes,
        }
