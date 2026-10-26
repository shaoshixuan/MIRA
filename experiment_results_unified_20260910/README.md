# 三数据集补充实验结果

本目录统一保存 Grocery 新结果以及 Automotive、Toys 的既有结果。

- `01_ablation/core_ablation_metrics.csv`：三个数据集的核心消融结果。
- `02_parameter/grocery_route_weight_sensitivity_metrics.csv`：Grocery 新结果的参数实验。
- `02_parameter/route_weight_sensitivity_metrics.csv`：三个数据集的统一参数结果。
- `02_parameter/selected_route_weights.csv`：各数据集、各路由的最终权重。
- `03_user_subgroup/grocery_user_subgroup_metrics.csv`：Grocery 的全体、低交互和 KG-only 用户结果。
- `03_user_subgroup/three_dataset_user_subgroup_comparison.csv`：三个数据集的目标子组对比数据。
- `03_user_subgroup/three_dataset_user_subgroup_comparison_k10.png`：用户子组可视化。
- `03_user_subgroup/USER_SUBGROUP_ANALYSIS_CN.md`：简短文字分析。

注意：Grocery 使用 2026-09-09 重新处理后的候选级 LLM/KG 路由结果；Automotive 和 Toys 使用 2026-08-27 的既有结果。Grocery 没有触发 `MIXED_ENHANCEMENT` 路由。
