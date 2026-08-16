# CHANGELOG

## 2026-08-16 V2 全面落地
- Flask → FastAPI 统一 /api;SQLAlchemy 2.x;目录对齐 §1.1
- Ensemble 四成员 + GBM 五成员,两层 Engine(Goal/Outcome)
- Snapshot/Calibration/Experiment/Feature Store/AUTO LEARN
- 删除部署/测试(后恢复最小集)/临时文件;代码审查 + 去重 + 性能优化(热缓存 18ms)

## 演进(历史)
- V1:Flask + HGBR 泊松 + 45 特征(回测 51.3%)
- V2-P0:快照/校准/实验/伤停位置加权
- V2-P1:Feature Factory/多维 ELO/H2H 降权/Ensemble
- V2-P2:FastAPI/configs YAML/pipelines/全球分层/自动选择
- 2026-08-16 审查落地:model_service 拆分、超参、滚动修复、校准三方法、GBM 成员、
  时间衰减(实测无增益回滚)、P0 十项修复、P1 配置统一/快照补全/Feature Factory 真实现/公式哈希

## 2026-08 第五轮审查落地(GitHub master 为审查对象:训练/评估/集成闭环)
- P0-1 GBM 时间 holdout 泄漏修复:先切前 80% 训练 → 后 20% 评估,再全量重训(不再全量 fit 后切段评估)
- P0-2 主模型生命周期修正:评估(80% 日期)后 100% 数据重训保存 —— 生产模型不再只吃过 80% 历史
- P0-3 Ensemble 权重学习重写为时间分段 OOF:段前训练临时 HGBR/GBM → 段内预测 →
  收集 OOF 概率 → 拟合 τ/φ → SLSQP;杜绝"最终模型回看历史学权重"
- P0-4 权重学习路径修复:att_diff 读 prepare 后主队行(原恒跳过联赛);GBM 统一 artifacts/<league>/gbm.pkl
- P0-5 Snapshot 修复:主/客特征分行(原只存客队行);核心字段/写库失败 raise —— 预测+快照原子事务
- P1-1 GBM 预测仅用主队行(原取两行只用一个)
- P1-2 ELO 同日时间穿越修复:按日初快照计算同日全部特征,日内场次互不影响,当日结束后统一更新
- P1-3 训练/评估/CV/概率 holdout 全部改为按日期分组切分(同一比赛日不跨 train/test)
- P1-4 fused λ 归一化:GBM 权重不再稀释 λ;λ 与 score matrix 融合一致
- P1-5 Calibration 严格 ORDER BY kickoff;calibration_version = artifact sha256
- P1-6 Auto Select 多指标门禁:0.40·poisson + 0.25·logloss + 0.20·brier + 0.15·rps,
  任一指标缺失/0 占位即拒绝(旧实验 0 值虚低 bug 已修);审计写 runtime/active_meta.json
- P1-7 权重学习成员动态化:GBM 不可用 → 从优化中移除,而非 [0,0,0] 假装存在
- 黄金测试 9 个:同日无泄漏/未来不改过去/快照特征=预测输入/GBM 不改 λ/权重归一化/
  Ensemble 一致性×2/预测确定性/权重学习动态成员(pytest 20 passed)
- 清理:calibration 统一 artifacts/calibration/;hyperopt_best.json → artifacts/experiments/hyperopt/best.json;
  README 路径修正(artifacts/<league>/<version>.pkl、runtime/active_models.json)、宣称降级
  ("100% 重放""2,000 场验证"改为目标描述)、Feature 6 家族标注为 4 实现 + 2 保留
