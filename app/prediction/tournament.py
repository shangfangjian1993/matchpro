"""赛事模拟(审查 §12/§42:自 model_service/prediction 拆分)。

Monte Carlo baseline(诚实定位:未建模主客场/阵容/抽签约束/加时点球)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.config import LeagueType
from app.core.timeutil import utcnow
from app.data.adapters import matches_to_dataframe
from app.models.loader import _load_model


def predict_tournament(
    league_type: LeagueType,
    teams: list,
    num_simulations: int = 1000,
    models_dir: str | None = None,
    seed: int | None = None,
) -> dict:
    """
    赛事模拟:对每支球队预测当前进球强度,单淘汰随机配对模拟,
    统计各队夺冠频率。

    seed: 随机种子。默认 None(每次调用使用系统熵,结果不同);
          传入固定值可复现结果。
    """
    from app.core.paths import MODELS_DIR as _MD

    models_dir = models_dir or str(_MD)
    from app.api.db import League, Match

    if not teams or len(teams) < 2:
        raise ValueError("至少需要 2 支球队参与模拟")

    league = League.query.filter_by(league_type=league_type.value).first()
    if league is None:
        raise ValueError(f"数据库中还没有 {league_type.value} 的联赛数据")

    model = _load_model(league_type, models_dir)

    history = Match.query.filter_by(league_id=league.id, match_status="finished").all()
    hist_df = matches_to_dataframe(history)
    if hist_df.empty:
        raise ValueError("数据库中没有历史比赛数据,无法构造赛前特征")

    # 无历史战绩球队防御:未在历史数据中出现过的球队无法构造强度特征
    known_teams = set(hist_df["home_team"]) | set(hist_df["away_team"])
    unknown = [t for t in teams if t not in known_teams]
    if unknown:
        raise ValueError(
            f"以下球队在 {league_type.value} 的历史数据中不存在: "
            f"{'、'.join(unknown)},请先录入其比赛数据"
        )

    # 球队强度 = 历史场均积分(胜3平1负0);对手取参赛队中最强者,
    # 使每队的 lambda 代表"对阵强敌"的进球强度(修复:原按单场最高进球选对手,噪声大)
    home_part = hist_df[["home_team", "home_goals", "away_goals"]].copy()
    home_part["team"] = home_part["home_team"]
    home_part["gf"] = home_part["home_goals"]
    home_part["ga"] = home_part["away_goals"]
    away_part = hist_df[["home_team", "away_team", "home_goals", "away_goals"]].copy()
    away_part["team"] = away_part["away_team"]
    away_part["gf"] = away_part["away_goals"]
    away_part["ga"] = away_part["home_goals"]
    long = pd.concat([home_part, away_part])[["team", "gf", "ga"]]
    long["pts"] = np.where(
        long["gf"] > long["ga"], 3.0, np.where(long["gf"] == long["ga"], 1.0, 0.0)
    )
    strength = long.groupby("team")["pts"].mean()

    # 一次性构造全部球队的预测行(对手=参赛队中最强者),单次 predict 拿到所有 λ。
    # 特征只依赖主/客队各自的历史(与具体对手无关),各行特征互不影响;
    # 同日期的 padding 行经 mergesort 稳定排序后保持原序,predictions[-N:] 与 teams 一一对应。
    now_ts = pd.Timestamp.now()
    pad_rows = []
    for team in teams:
        opponent = max(
            (t for t in teams if t != team),
            key=lambda t: float(strength.get(t, 0.0)),
        )
        pad_rows.append(
            {
                "date": now_ts,
                "home_team": team,
                "away_team": opponent,
                "home_goals": np.nan,
                "away_goals": np.nan,
                "goals": np.nan,
                "league": league.name,
                "season": league.season or "",
            }
        )
    pred_df = pd.concat([hist_df, pd.DataFrame(pad_rows)], ignore_index=True)
    raw = model.predict(pred_df)
    preds = raw["predictions"]
    team_lambda = {team: float(preds[-len(teams) + i]) for i, team in enumerate(teams)}

    # 模拟单淘汰赛(Monte Carlo baseline——审查 §26:定位为基线,不暗示完整赛事模型)
    rng = np.random.default_rng(seed)  # seed=None → 系统熵
    champion_counts = {t: 0 for t in teams}
    for _ in range(max(1, int(num_simulations))):
        remaining = teams.copy()
        rng.shuffle(remaining)
        # 补齐到 2 的幂(轮空处理:直接晋级)
        while len(remaining) & (len(remaining) - 1) != 0:
            remaining.append(None)
        while len(remaining) > 1:
            next_round = []
            for i in range(0, len(remaining), 2):
                a, b = remaining[i], remaining[i + 1]
                if a is None:
                    next_round.append(b)
                    continue
                if b is None:
                    next_round.append(a)
                    continue
                ga = rng.poisson(team_lambda[a])
                gb = rng.poisson(team_lambda[b])
                next_round.append(a if ga >= gb else b)
            remaining = next_round
        if remaining and remaining[0]:
            champion_counts[remaining[0]] += 1

    total = sum(champion_counts.values()) or 1
    champion_probs = sorted(
        [
            {"team": t, "probability": round(c / total, 4)}
            for t, c in champion_counts.items()
            if c > 0
        ],
        key=lambda x: -x["probability"],
    )
    most_likely = champion_probs[0]["team"] if champion_probs else None

    return {
        "league_type": league_type.value,
        "num_simulations": num_simulations,
        "champion_probabilities": champion_probs,
        "most_likely_champion": most_likely,
        "team_strengths": {t: round(v, 3) for t, v in team_lambda.items()},
        # 审查 §26:诚实定位 —— Monte Carlo baseline,未建模主客场/阵容/抽签约束/加时点球
        "model": "monte_carlo_baseline",
        "limitations": "baseline:未建模主客场/中立场/阵容/晋级阶段/抽签约束/加时点球",
        "timestamp": utcnow().isoformat(),
    }
