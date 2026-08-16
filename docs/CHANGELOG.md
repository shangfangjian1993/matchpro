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
