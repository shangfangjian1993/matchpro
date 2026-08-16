# 足球概率预测引擎(matchpro)

> 版本:v5.0 | 数据:27,825 场真实历史 | API:FastAPI 统一 /api

数据驱动 + 多层融合的足球概率预测引擎(五大联赛 + 欧冠 + 世界杯 + 欧洲杯)。
科研级可复现:每次预测自动落 Snapshot(最终输出 + 输入 + 全版本哈希),可 100% 重放。

## 定位

- 从"比分预测"升级为"概率预测引擎":胜平负 + Top5 比分 + Over/Under + BTTS + xG
- 严格防泄漏(时间重放/赛前特征),数据真实(无源不编造,缺失 NaN)
- 所有改动用 2,000 场防泄漏回测验证(有提升才上线)

## 架构(两层 Engine)

```
数据(sources→raw→canonical)→ Feature Factory(6 家族)
  → Goal Engine(HGBR+DC+NB+ELO)→ 比分矩阵(1X2/O-U/BTTS/xG)
  → Outcome Engine(GBM 分类器)→ 1X2 融合
  → Calibration(β/Platt/Isotonic 择优)→ Snapshot → API
```

## 目录

```
app/
├── api/          # FastAPI(统一 /api;纯 API 层:app/db/auth/helpers/leagues/matches/meta/models/predictions/schemas/security/user)
├── core/         # config/constants/timeutil/paths/cache(logging/exceptions 统一)
├── data/         # 数据引擎(sources/canonical/adapters/pipeline/raw;运行数据在根 data/)
├── features/     # Feature Factory(factory/rolling/strength/attack_defense/form/h2h/registry,4 真实族)
├── models/       # 模型引擎(poisson/dixon_coles/negbin/elo_goal/ensemble/distributions/utils/registry/loader/integrity)
├── prediction/   # 预测服务(service 编排 + goal_engine/outcome_engine/calibration/snapshot/tournament/info_fusion)
├── calibration/  # 概率校准(β/Platt/Isotonic + 失败保护)
├── replay/       # 快照回放/评估
└── services/     # 业务服务(按域分组:data/training/model/system)
artifacts/        # 模型资产(models/ 按联赛分版本 + ensemble/ + experiments/)
data/             # 运行数据(football.db)
runtime/          # active_models.json(动态指针)
pipelines/        # 五件套(ingest/feature/train/predict/replay)
configs/          # leagues.yaml(联赛元信息)/ models.yaml(模型参数+特征开关)/ system.yaml
migrations/       # Alembic(0001→0012)
tests/            # 最小测试集(11 测试)
```

## 安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # 生产依赖(17 项)
# 开发: pip install ruff pytest
```

## 数据初始化

```bash
# 迁移建表
DATABASE_URL=sqlite:///app/data/football.db alembic -c alembic.ini upgrade head
# 一键初始化(爬历史+xG+伤停 → 训练 → 验证)
python -m app.services.data.init_production
# 种子中文映射 + 球队实体回填
python -m app.services.data.seed_team_names && python -m app.services.data.backfill_teams
```

## 训练

```bash
# 全联赛训练(超参 lr=0.03/depth=6,Experiment 自动记录)
python -m app.services.training.train_all_leagues
# Ensemble 权重滚动学习(5 成员,最近 500 场窗口)
python -m app.services.training.learn_ensemble_weights
# 自动模型选择(active_models.json,只选文件存在的最优版本)
python -m app.services.model.auto_select_model
# GBM 分类成员(Outcome Engine)
python -m app.services.training.gbm_member
# 超参搜索 / 特征重要性
python -m app.services.training.hyperopt && python -m app.services.model.feature_importance
```

## 预测

```bash
python pipelines/predict.py --league premier_league --home 阿森纳 --away 切尔西
# 输出:胜平负 + Top5 + Over/Under + BTTS + xG + prediction_status
```

## 回放与评估

```bash
python pipelines/replay.py      # 快照回放 + §6 指标(Log-Loss/Brier/RPS/ECE),汇入 experiments
python -m app.services.data.backfill_snapshots
python -m app.services.training.fit_calibration   # 校准拟合(快照 ≥150/联赛,β/Platt/Isotonic 择优)
```

## API

统一前缀 **/api**(无 v1/v2),FastAPI + Pydantic + JWT cookie + CSRF。
交互文档:`GET /docs`(OpenAPI)。端点详见 docs/PROJECT_OVERVIEW.md 第十章。

```bash
uvicorn app.api.app:app --host 0.0.0.0 --port 8000 --workers 2
```

## 配置

| 文件 | 职责 |
|---|---|
| configs/models.yaml | **模型参数单一源**(leagues 段)+ Ensemble/校准/训练 + features.h2h 开关 |
| configs/leagues.yaml | 联赛元信息(country/competition_type) |
| configs/system.yaml | 数据库/API/调度时间窗/数据源 |

## 模型版本管理

- 版本文件:`app/models/<league>_v<version>.pkl`(递增,无 latest 复制)
- active 指针:`app/models/active_models.json`(自动选择写入,prune 保护)
- 特征版本:feature_store(公式哈希,规格驱动)

## 开发规范

1. 严格防泄漏:特征只用赛前信息;reference_date 用训练集内最大日期
2. 数据真实:无源不编造
3. 回测验证:所有改动 2,000 场防泄漏 A/B
4. 单一实现:概率基元在 distributions.py;特征在 features/(滚动族 factory 调度)
5. 分级异常:可降级(GBM/校准)warning + prediction_status=degraded;核心失败 raise
6. 测试:改动后 `python -m pytest`(11 测试)

## 历史演进

见 docs/CHANGELOG.md(V1 → V2 → P0/P1/P2 演进记录)。
