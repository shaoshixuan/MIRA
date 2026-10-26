# Toys 目标系数敏感性实验

在原实验的路由分层用户划分上，将验证目标设为 `β × NDCG@15 + (1−β) × Recall@15`，β 取 0.4、0.5、0.6、0.7、0.8。每条路由独立从探索权重 `{0.0, 0.1, ..., 1.0}` 中选择目标函数最高的值，再在相同的留出用户上重新排序和计算指标。验证集有 3122 名用户，留出集有 781 名用户；主要报告 K=10，同时提供 K=5、15 的结果。

候选评分文件位于 `D:\graduate_recsys\datasets_result_forevaluate\toys_scores.csv`；原实验使用的训练历史已在 `E:\Toys_and_Games-20260920T121442Z-1-001\Toys_and_Games\minimal\rec_train.txt` 找到。两份文件的 SHA-256 均与冻结协议一致，见 `audit.json`。本实验复用相同原始验证网格的 Recall/NDCG，按新系数重新排序；脚本用原始候选分数逐路由复核了网格记录，并重新计算全部留出集指标。β=0.6 的 Recall、NDCG、CC、ILD、Novelty 在 K=5、10、15 下均与原实验留出结果一致。

结果文件：`validation_grid.csv` 为全部候选目标值，`selected_route_weights.csv` 为每组系数下的路由权重，`holdout_metrics.csv` 为 K=5、10、15 的留出集指标，`toys_objective_sensitivity.tex` 为正文可用的 K=10 三线表。主文档导言区加载 `\usepackage{booktabs}` 后，用 `\input{...}` 引入表格文件。

## 主要观察

- β=0.4、0.5、0.6 均选择 KG_ONLY=0.4、MIXED_ENHANCEMENT=0.8、LOW_INTERACTION=0.1，三组留出集指标完全相同。β=0.7、0.8 则均选择 0.3、0.8、0.9，得到另一组相同的留出结果。
- 在 K=10 时，β=0.7/0.8 的 Recall=0.3566、NDCG=0.1900、Novelty=11.3997，高于 β=0.4–0.6 的 0.3560、0.1787、11.2362；但其 CC 从 0.6342 降至 0.6116，ILD 从 0.9769 降至 0.9735。
- 在调权使用的 K=15 下，β=0.4–0.6 的 Recall=0.4629、CC=0.6935、ILD=0.9774 较高；β=0.7/0.8 的 NDCG=0.2170、Novelty=11.5170 较高。因此 Toys 不能独立证明 0.6 是唯一最优系数；它对应的是较高的 K=15 召回和覆盖率，而更大的 NDCG 系数对应更高的 NDCG。跨数据集固定系数的判断应结合 Automotive 与待完成的 Grocery，并预先明确评价准则。

复现命令：在本目录运行 `python -B automotive_coefficient_sensitivity.py --dataset toys`。脚本会将重新计算的文件写入本目录下的 `toys_coefficient_results` 子目录。
