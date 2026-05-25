# STRICT TDC RULES

1. 只用 train/valid 做训练、选超参、选 seed、选融合权重
2. test 只用于最终一次评估，不参与任何选择
3. 所有候选比较按 best_valid_primary 或 valid 指标进行
4. 每个正式结果至少 5 个独立 seed
5. exploratory 和 strict 结果严格分开存放
