# Grocery 目标系数敏感性实验

使用 `D:\graduate_recsys\datasets_result_forevaluate\grocery_scores.csv` 和 `E:\Grocery_and_Gourmet_Food-20260920T121440Z-1-001\Grocery_and_Gourmet_Food\minimal\rec_train.txt`。验证集共 3093 名用户，留出集共 774 名用户；固定随机种子 2026，按用户路由分层划分。分别令 NDCG 系数 β 为 0.4、0.5、0.6、0.7、0.8，在验证集上逐路由搜索探索权重 `{0.0, 0.1, ..., 1.0}`，最大化 `β × NDCG@15 + (1−β) × Recall@15`。将各组选出的权重用于同一留出集，计算 K=5、10、15 下的 Recall、NDCG、类别覆盖率（CC）、ILD、Novelty。

这份新 Grocery 候选文件与冻结实验使用的 Grocery 文件拥有相同的用户、候选键和标签，但探索分数来源发生变化。因此本次重新计算了每个验证集网格点，没有使用旧版 Grocery 的网格结果，也不将旧版留出结果作为数值基准。新文件存在 `LOW_INTERACTION` 用户的候选级 KG 回退分数；用户路由以文件中的 `applied_route` 为准，不能仅从候选级 `explore_score_source` 推断。来源文件 SHA-256、划分规模见 `audit.json`。

## 结果

五个 β 值全部选出相同权重：KG_ONLY=0.0，LOW_INTERACTION=0.1。Grocery 没有 MIXED_ENHANCEMENT 用户，故该路由不参与调权；LaTeX 表格用 `--` 表示。所有 β 对留出用户生成完全相同的推荐排序，因而全部指标相同。K=10 时，Recall=0.3525、NDCG=0.2133、CC=0.6581、ILD=0.9958、Novelty=10.0888；K=15 时，Recall=0.4113、NDCG=0.2290、CC=0.7278、ILD=0.9958、Novelty=10.3464。

因此 Grocery 不支持“β=0.7 比 β=0.6 更好”。按照预设的条件，本次不将固定 NDCG 系数改为 0.7，也不重算其他三个数据集的动态权重或后续消融、参数、用户子组结果。此结论只说明在当前 Grocery 验证集和 0.1 间隔权重网格下，两组目标系数给出了相同的权重与排序；不能推出目标系数在更细网格或其他数据上没有影响。

结果文件：`validation_grid.csv` 为所有验证候选，`selected_route_weights.csv` 为每组系数的最优权重，`holdout_metrics.csv` 为 K=5、10、15 的留出集指标，`grocery_objective_sensitivity.tex` 为 K=10 三线表。主文档导言区需加载 `\usepackage{booktabs}`。

在包含本目录代码和 `frozen_evaluation.py` 的环境中复现：`python -B automotive_coefficient_sensitivity.py --dataset grocery`。
