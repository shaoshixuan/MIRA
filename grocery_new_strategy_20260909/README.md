# Grocery 新策略复现实验

本目录重建 Grocery 数据处理、候选级探索路由、动态权重选择、最终重排与 A0–A4 消融评估。

## 实际执行规则

- 低交互用户：历史中至少一个商品具有 `title/description` 时，对每个具有文本的候选商品使用 LLM；无文本候选逐项回退 KG。
- 活跃用户：默认使用 KG；若 KG 信号不足，则仅对弱 KG 且具有文本的候选使用 LLM 增强，其余候选继续使用 KG。
- 本次 Grocery 数据按原始弱 KG 判定阈值没有产生 `MIXED_ENHANCEMENT` 用户，因此该分支已实现但没有观测样本。
- 动态探索权重在 80% 路由分层调参用户上搜索，随机种子为 2026，目标为 `0.6*NDCG@15 + 0.4*Recall@15`；并列时选择更小的探索权重。

## 主要命令

```powershell
python run_grocery_pipeline.py prepare
python merge_llm_shards.py
python run_grocery_pipeline.py evaluate --llm-result llm_output/llm_exploration_scores_complete.csv
```

## 主要产物

- `processed/preprocessing_report.json`：数据覆盖率与路由统计。
- `processed/user_route_audit.csv`：用户级路由审计。
- `llm_output/llm_exploration_scores_complete.csv`：完整 LLM 候选分数。
- `results/selected_weights_and_protocol.json`：权重和验证协议。
- `results/ablation_and_final_metrics.csv`：全体、留出集和用户子组的 A0–A4 指标。
- `results/final_ranked_recommendations.csv`：A4 最终排序。
- `RESULTS_SUMMARY.md`：结果摘要与论文解释边界。

