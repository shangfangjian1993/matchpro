# 足球概率预测引擎 · 项目文档

> 版本:v6.0 | 更新:2026-08-16(结构收口:一套目录/一套入口/一套模型路径)
> 定位:数据驱动 + 多层融合的足球概率预测引擎(五大联赛 + 欧冠 + 世界杯 + 欧洲杯)

---

## 一、项目概述

基于 **27,825 场真实历史数据**的足球概率预测引擎:

- **预测**:胜平负 + Top5 比分 + Over/Under + BTTS + xG(两层 Engine)
- **可复现**:每次预测自动落 Snapshot(最终输出 + 输入 + 全版本哈希)
- **评估闭环**:Log-Loss/Brier/RPS/ECE,回放汇入实验,AUTO LEARN 自动重学
- **诚实原则**:无源不编造;所有改动回测验证;Monte Carlo baseline 明确标注

## 二、目录结构(结构收口后)

```
matchpro/
├── app/
│   ├── api/          # FastAPI(纯 API 层,12 文件)
│   ├── core/         # config/constants/timeutil/paths/cache/logging/exceptions
│   ├── data/         # 数据引擎(sources/canonical/adapters/pipeline/raw)
│   ├── features/     # Feature Factory(factory/rolling/strength/attack_defense/form/h2h/registry)
│   ├── models/       # 模型引擎(poisson/dixon_coles/negbin/elo_goal/ensemble/
│   │                 #   distributions/utils/registry/loader/integrity)
│   ├── prediction/   # 预测服务(service/goal_engine/outcome_engine/calibration/snapshot/tournament/info_fusion)
│   ├── calibration/  # 概率校准(β/Platt/Isotonic 择优 + 失败保护)
│   ├── replay/       # 快照回放/评估
│   └── services/     # 业务服务(data/training/model/system 四组)
├── artifacts/        # 模型资产(models/ + ensemble/ + experiments/)
├── data/             # 运行数据(football.db)
├── runtime/          # active_models.json(动态指针)
├── configs/          # leagues.yaml / models.yaml / system.yaml
├── pipelines/        # 五件套
├── migrations/       # Alembic(0001→0012)
├── tests/            # 最小测试集
└── docs/ frontend/ notebooks/
```

## 三、架构(两层 Engine + 分层可解释)

```
数据 → Feature Factory(4 家族)
  → Goal Engine(HGBR/DC/NB/ELO)→ λ、比分矩阵(Top5/O-U/BTTS/xG)
  → Outcome Engine(GBM)→ 1X2 融合
  → Calibration(β/Platt/Isotonic)
  → Snapshot(冻结最终输出)→ API
每层可关闭:GBM/校准失败 → prediction_status=degraded(warning);模型哈希失败 → raise
```

## 四、模型与资产

- 版本文件:`artifacts/models/<league>/<version>.pkl`(递增,无 latest 复制)
- active 指针:`runtime/active_models.json`(自动选择,prune 保护 active)
- 校准器:`app/models/<league>_model.cal`(三方法择优)
- Ensemble 权重:`artifacts/ensemble/`(学习产物)
- 调参实验:`artifacts/experiments/`(hyperopt 等)

## 五、测试

```bash
python -m pytest        # 11 测试(预测/Ensemble/API/概率;慢测试默认跳过)
```

## 六、API

统一 **/api**(FastAPI + Pydantic + JWT cookie + CSRF);`GET /docs` 交互文档。
端点清单见 docs/api 说明(认证/联赛/比赛/预测/模型/训练/用户/通知/元数据)。

## 七、开发规范

1. 严格防泄漏;2. 数据真实;3. 回测验证;4. 单一实现(distributions/factory);
5. 分级异常;6. 结构收口(不新增平行目录/入口);7. 文档与代码同步。
