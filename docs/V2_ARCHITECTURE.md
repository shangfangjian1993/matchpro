# 足球概率预测引擎 · 架构文档(V2 落地后现状)

> 版本:v6.0 | 2026-08-16 结构收口完成

## 架构决策记录

| 决策 | 现状 |
|---|---|
| API | FastAPI 统一 /api(无 v1/v2);api/ 为纯 API 层 |
| 预测服务 | app/prediction/(service 编排 + goal/outcome/calibration/snapshot/tournament) |
| 模型管理 | app/models/(registry 版本管理 / loader 加载 / integrity 校验 / engine 家族) |
| Feature Factory | 4 真实家族(strength/attack_defense/form/h2h)+ factory 调度 + registry(公式哈希含实现) |
| 数据 | 引擎 app/data/ + 运行数据 data/football.db(路径由 core/paths.py 单一来源) |
| 资产 | artifacts/(models 分版本 + ensemble + experiments)+ runtime/active_models.json |
| 缓存 | core/cache/(ArtifactCache + PredictionCache,键含模型/权重/校准 mtime) |
| 配置 | configs/models.yaml 模型参数单一源 |
| 校准 | β/Platt/Isotonic 三方法,fit_best 联赛择优(ECE),失败保护 |
| 评估 | Log-Loss/Brier/RPS/ECE;experiments 追踪;回放汇入;AUTO LEARN |
| 测试 | tests/(11 测试,DB 经 core.paths) |
| 已移除 | Flask/scripts/blueprints/v1/model_service 兼容层/latest.pkl 复制 |

## 演进史

见 CHANGELOG.md。
