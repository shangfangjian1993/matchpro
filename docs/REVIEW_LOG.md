# 审查历史记录

## 2026-08 第一轮审查 (A70A601)
- P0-1: reconcile _same_score 重复定义 → 已修
- P0-2: source_consensus 字段级假共识 → 已修
- P0-3: upsert vs reconciliation 双轨 → 已修
- P0-4: Ablation future-trained leakage → 已修

## 2026-08 第二轮审查 (f01d7e4)
- P0-1: as_utc_naive lazy import → 已修
- P0-2: db.py timeout comment → 已修

## 2026-08 第三轮审查 (909c44e)
- P1-7: learn_weights present all → 已修
- P1-8: Ablation xG-MAE 共用 → 已修

## 2026-08 第四轮审查
- P0-1: timeutil pandas 依赖 → 已修
- P0-2: ingest vs reconciliation 双轨 → 已修

## 2026-08 第五轮审查 (b2ab244)
- P1-7: 清理审查编号注释 → 已修
- P1-2: merge_league 双 ingestion → 已修

## 2026-08 第六轮审查
- P0-1: maybe_update 闭包作用域 → 已修
- P0-2: bzzoiro cutoff 时区 → 已修

## 2026-08 第七轮审查
- P0-1: LayeredPipeline 完整收口 → 已修
- P0-2: ABLATION_MASKS G/H 修复 → 已修

## 2026-08 第八轮审查
- P0-1: τ 拟合方向修复 → 已修
- P0-2: G/H Ablation 相同 → 已修
- P0-3: GBM 权重固定 0 → 已修
- P0-4: τ/φ 使用 fused λ → 已修

## 2026-08 第九轮审查
- P0-1: DAG 训练 → 已修
- P0-2: NB φ 公式 → 已修
- P0-3: Layer-3 权重 → 已修
- P0-4: EnsembleTrainingResult → 已修

## 2026-08 第十轮审查
- P0-1: ProductionArtifact 接管主链 → 已修
- P0-2: from_dict 丢 calibration → 已修
- P0-3: Typed Schema → 已修
- P0-4: Calibration params → 已修

## 2026-08 第十一轮审查
- P0-1: 多联赛 Artifact 覆盖 → 已修
- P0-2: Typed Schema dict → 已修
- P0-3: GBM learned state → 已修
- P0-4: Artifact league 字段 → 已修
- P1-1: training_cutoff 主链 → 已修
- P1-2: oof_segments K_SEG → 已修
- P1-4: Split optimizer → 已修
- P1-5: Bounded simplex → 已修
- P1-6: 全精度序列化 → 已修

## 2026-08 第十二轮审查
- P0-1: Production Loader League Scoped → 已修
- P0-2: ProductionArtifact 唯一输入 → 已修
- P0-4: Artifact 权重严格验证 → 已修
- P0.5: GBM hash 验证 → 已修
- opt-2: DC/NB 指标重命名 → 已修
- opt-4: diagnostics degraded 传播 → 已修
- opt-5: Weight Stability JSON → 已修
- opt-6: learn_weights deprecation → 已修

## 2026-08 第十三轮审查 (外部)
- P0-1: 概率不变量测试 → 已修
- P0-2: 清理审查标记 → 已修
- P1-1: Walk-forward 回测门禁 → 已修
- P1-2: 特征 NaN + Bayes 先验 → 已修
- P2-1: Lineup + Uncertainty → 已修
- P2-2: MAX_GOALS 可配置 → 已修
