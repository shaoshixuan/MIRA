# MIRA

## 研究简介

MIRA 面向推荐系统中的准确性与多样性平衡，融合点击率预测（pCTR）、知识图谱（KG）兴趣信号与大语言模型（LLM）探索信号，对候选商品进行重排。

研究根据用户历史交互和 KG 信号质量区分用户路由，通过 LLM 补充低交互用户及弱 KG 候选的探索信息，并按路由选择融合权重。代码涵盖 Automotive、Toys 和 Grocery 三个数据集的数据处理、推荐评估、消融实验与参数敏感性分析。

## 代码目录

| 目录 | 说明 |
| --- | --- |
| `data/KG-related_Files/` | 知识图谱预处理、转移矩阵与个性化重启向量构建、KG 兴趣打分，以及 LLM 探索打分和断点恢复。 |
| `data/KG-related_Files/model_cache/` | 文本向量模型的说明文档。 |
| `data_deal_afterpctr_forllm/` | 为 pCTR 候选预测补全商品 ASIN、标题和描述，默认配置对应 Toys。 |
| `llm_datadeal_for_historyitem_descri/` | 商品元数据映射、用户历史文本整理与 LLM 输入准备，包含 Automotive 候选处理入口。 |
| `data_deal_afterexplore_forevaluate/` | 合并 KG 与 LLM 探索分数，并补充商品类别信息。 |
| `new_evaluate/` | 分数校准、加权及 MMR 重排、自动调参和准确性、多样性、新颖性等指标评估。 |
| `supplementary_experiments_dataset_adaptive_20260827/` | 多数据集 A0–A4 消融实验、路由权重搜索、敏感性分析及论文结果整理。 |
| `grocery_new_strategy_20260909/` | Grocery 数据准备、候选级路由、LLM 任务分片与结果合并，以及重排和消融评估。 |
| `experiment_results_unified_20260910/` | 汇总三个数据集的消融、参数和用户子组实验结果。 |
| `objective_coefficient_sensitivity_grocery/` | 目标函数系数敏感性实验，包含 Grocery 显式用户路由的处理。 |
| `objective_coefficient_sensitivity_toys/` | Toys 等数据集的目标函数系数敏感性实验与冻结评估模块。 |
