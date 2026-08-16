# raw(§2.2 数据分层:不可变原则)

原始数据按源分目录(fdco/fdo/understat/injuries/news/lineups),
永不修改;canonical 清洗后实体在数据库;features 特征表在 feature_store;snapshots 预测快照在 prediction_snapshots。
