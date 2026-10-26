# evaluate_merged_scores.py
"""
合并分数评估代码
根据实际数据列进行评估
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import argparse
from datetime import datetime

# ==================== 1. 读取数据 ====================
print("=" * 60)
print("加载数据")
print("=" * 60)

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument(
    "--scores-file",
    default="D:/graduation_recsys/data_deal_afterexplore_forevaluate/merged_scores_with_categories_20260811_164542.csv",
)
parser.add_argument(
    "--history-file",
    default="D:/graduation_recsys/data_deal_afterexplore_forevaluate/kg_train_triples_forjiaohu.txt",
)
parser.add_argument("--k", type=int, default=15)
parser.add_argument("--output-dir", default=None)
ARGS = parser.parse_args()

# 待评估的合并分数文件
scores_file = ARGS.scores_file

# 历史交互数据
history_file = ARGS.history_file

# 读取评分数据
print(f"\n读取评分数据: {scores_file}")
encodings = ['utf-8-sig', 'utf-8', 'gb18030', 'gbk']
df = None
for enc in encodings:
    try:
        df = pd.read_csv(scores_file, encoding=enc)
        print(f"  ✅ 使用 {enc} 编码")
        break
    except:
        continue

if df is None:
    raise ValueError("无法读取评分文件")

# 清理列名（去除可能的空格）
df.columns = df.columns.str.strip()

print(f"  数据: {df.shape[0]} 行, {df.shape[1]} 列")
print(f"  实际列名: {list(df.columns)}")

# ==================== 列名映射 ====================
# 根据实际列名确定使用的列
COLUMN_MAP = {
    'user_id': 'user_id',
    'item_id': 'item_id',
    'is_positive': 'is_positive',
    'final_score': 'final_combined_score',  # 用于排序的最终分数
    'explore_score': 'explore_score',  # 探索分数
    'category': 'category',  # 类别
    'description': 'description',  # 描述
    'title': 'title',  # 标题
    'pctr_score': 'original_pctr_score',  # pCTR分数
    'kg_score': 'kg_interest_score',  # KG分数
    'llm_score': 'llm_exploration_score',  # LLM分数
    'score_source': 'explore_score_source',  # 分数来源
}

# 检查必要列是否存在
for col_name, col_key in COLUMN_MAP.items():
    if col_key not in df.columns:
        print(f"  ⚠️ 列 '{col_key}' 不存在，相关指标可能无法计算")

# 统一数据类型
df['user_id'] = df['user_id'].astype(str).str.strip()
df['item_id'] = df['item_id'].astype(str).str.strip()
df['is_positive'] = df['is_positive'].astype(int)

# 创建统一的 final_score 列
df['final_score'] = df[COLUMN_MAP['final_score']]

# 处理缺失值
if COLUMN_MAP['category'] in df.columns:
    df['category'] = df[COLUMN_MAP['category']].fillna('')
else:
    df['category'] = ''

if COLUMN_MAP['description'] in df.columns:
    df['description'] = df[COLUMN_MAP['description']].fillna('')
else:
    df['description'] = ''

print(f"\n数据概览:")
print(f"  总记录数: {len(df)}")
print(f"  用户数: {df['user_id'].nunique()}")
print(f"  商品数: {df['item_id'].nunique()}")
print(f"  正样本数: {df['is_positive'].sum()}")
print(f"  负样本数: {len(df) - df['is_positive'].sum()}")

# 按评分来源统计
if COLUMN_MAP['score_source'] in df.columns:
    print(f"\n评分来源分布:")
    source_col = COLUMN_MAP['score_source']
    source_counts = df[source_col].value_counts()
    for source, count in source_counts.items():
        print(f"  {source}: {count} ({count / len(df) * 100:.1f}%)")

# 分数统计
print(f"\n分数统计:")
print(f"  final_score 范围: [{df['final_score'].min():.6f}, {df['final_score'].max():.6f}]")
print(f"  final_score 均值: {df['final_score'].mean():.6f}")
print(f"  final_score 中位数: {df['final_score'].median():.6f}")

if COLUMN_MAP['explore_score'] in df.columns:
    explore_col = COLUMN_MAP['explore_score']
    explore_scores = df[explore_col].dropna()
    if len(explore_scores) > 0:
        print(f"  explore_score 范围: [{explore_scores.min():.6f}, {explore_scores.max():.6f}]")
        print(f"  explore_score 均值: {explore_scores.mean():.6f}")

if COLUMN_MAP['pctr_score'] in df.columns:
    pctr_col = COLUMN_MAP['pctr_score']
    print(f"  pctr_score 范围: [{df[pctr_col].min():.4f}, {df[pctr_col].max():.4f}]")
    print(f"  pctr_score 均值: {df[pctr_col].mean():.4f}")

if COLUMN_MAP['kg_score'] in df.columns:
    kg_col = COLUMN_MAP['kg_score']
    kg_valid = df[(df[kg_col] != -10) & (df[kg_col] != -5)][kg_col]
    if len(kg_valid) > 0:
        print(f"  kg_score (有效值) 范围: [{kg_valid.min():.8f}, {kg_valid.max():.8f}]")
        print(f"  kg_score (有效值) 均值: {kg_valid.mean():.8f}")

if COLUMN_MAP['llm_score'] in df.columns:
    llm_col = COLUMN_MAP['llm_score']
    llm_valid = df[llm_col].dropna()
    if len(llm_valid) > 0:
        print(f"  llm_score 范围: [{llm_valid.min():.8f}, {llm_valid.max():.8f}]")
        print(f"  llm_score 均值: {llm_valid.mean():.8f}")

# ==================== 2. 读取历史交互数据 ====================
print(f"\n读取历史交互数据: {history_file}")

if os.path.exists(history_file):
    history_df = pd.read_csv(
        history_file,
        sep="\t",
        names=["user_id", "item_id", "relation"]
    )
    # Keep historical IDs in the same type as candidate IDs. Without this,
    # string candidate item IDs never match integer historical item IDs.
    history_df["user_id"] = history_df["user_id"].astype(str).str.strip()
    history_df["item_id"] = history_df["item_id"].astype(str).str.strip()
    print(f"  历史交互: {history_df.shape[0]} 行")

    # 计算item全局流行度
    item_popularity = history_df["item_id"].value_counts()
    total_interactions = item_popularity.sum()
    item_popularity_prob = item_popularity / total_interactions

    print(f"  总交互数: {total_interactions}")
    print(f"  唯一商品数: {len(item_popularity)}")
else:
    print(f"  ⚠️ 历史交互文件不存在，新颖性指标将使用默认值")
    item_popularity = pd.Series()
    item_popularity_prob = pd.Series()
    total_interactions = 0


# ==================== 3. 工具函数 ====================

def get_topk(data, k=15):
    """获取每个用户的Top-K推荐"""
    return (
        data.sort_values(["user_id", "final_score"], ascending=[True, False])
        .groupby("user_id")
        .head(k)
    )


# ==================== 4. 准确度指标 ====================

def precision_at_k(data, k):
    """Precision@K"""
    topk = get_topk(data, k)
    if len(topk) == 0:
        return 0.0
    return topk["is_positive"].mean()


def recall_at_k(data, k):
    """Recall@K"""
    recalls = []
    for uid, group in data.groupby("user_id"):
        pos_total = group["is_positive"].sum()
        if pos_total == 0:
            continue
        topk = get_topk(group, k)
        if len(topk) == 0:
            continue
        topk_pos = topk["is_positive"].sum()
        recalls.append(topk_pos / pos_total)
    return np.mean(recalls) if recalls else 0.0


def ndcg_at_k(data, k):
    """NDCG@K"""

    def dcg(rels):
        return np.sum([
            rel / np.log2(idx + 2) for idx, rel in enumerate(rels)
        ])

    ndcgs = []
    for _, group in data.groupby("user_id"):
        ranked = group.sort_values("final_score", ascending=False)
        rels = ranked["is_positive"].values[:k]
        ideal_rels = np.sort(ranked["is_positive"].values)[::-1][:k]

        if len(ideal_rels) == 0 or ideal_rels.sum() == 0:
            continue

        ndcgs.append(dcg(rels) / dcg(ideal_rels))
    return np.mean(ndcgs) if ndcgs else 0.0


def hr_at_k(data, k):
    """Hit Rate@K"""
    topk = get_topk(data, k)
    if len(topk) == 0:
        return 0.0
    hr = topk.groupby("user_id")["is_positive"].max()
    return hr.mean() if len(hr) > 0 else 0.0


def mrr_at_k(data, k):
    """MRR@K (Mean Reciprocal Rank)"""
    mrrs = []
    for _, group in data.groupby("user_id"):
        ranked = group.sort_values("final_score", ascending=False)
        found = False
        for rank, (_, row) in enumerate(ranked.iterrows()):
            if row["is_positive"] == 1 and rank < k:
                mrrs.append(1.0 / (rank + 1))
                found = True
                break
        if not found:
            mrrs.append(0.0)
    return np.mean(mrrs) if mrrs else 0.0


# ==================== 5. 多样性指标 ====================

def parse_categories(category_str):
    """解析类别字符串（可能用 | 分隔）"""
    if pd.isna(category_str) or not category_str:
        return []
    return [c.strip() for c in str(category_str).split('|') if c.strip()]


def category_coverage_at_k(data, k):
    """类别覆盖率@K"""
    topk = get_topk(data, k)

    # Top-K中出现的类别
    topk_categories = set()
    for cats in topk["category"]:
        topk_categories.update(parse_categories(cats))

    # 全局类别
    global_categories = set()
    for cats in data["category"]:
        global_categories.update(parse_categories(cats))

    if len(global_categories) == 0:
        return 0.0

    return len(topk_categories) / len(global_categories)


def ild_category_at_k(data, k):
    """类别ILD@K (Intra-List Diversity)"""
    ilds = []
    for _, group in data.groupby("user_id"):
        topk = get_topk(group, k)
        cats_list = []
        for cats in topk["category"]:
            parsed = parse_categories(cats)
            if parsed:
                # 用第一个类别代表该商品（或所有类别的组合）
                cats_list.append('|'.join(sorted(parsed)))

        if len(cats_list) < 2:
            continue

        pairs = [
            1 if cats_list[i] != cats_list[j] else 0
            for i in range(len(cats_list))
            for j in range(i + 1, len(cats_list))
        ]
        ilds.append(np.mean(pairs))

    return np.mean(ilds) if ilds else 0.0


def ild_description_at_k(data, k):
    """描述ILD@K (基于TF-IDF的语义多样性)"""
    if data["description"].isna().all() or (data["description"] == '').all():
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    ilds = []

    for _, group in data.groupby("user_id"):
        topk = get_topk(group, k)
        texts = topk["description"].fillna('').tolist()
        texts = [t for t in texts if t and str(t).strip()]

        if len(texts) < 2:
            continue

        try:
            X = vectorizer.fit_transform(texts)
            sim = cosine_similarity(X)
            # 上三角矩阵的均值
            triu_indices = np.triu_indices(len(texts), 1)
            if len(triu_indices[0]) > 0:
                ild = 1 - sim[triu_indices].mean()
                ilds.append(ild)
        except:
            continue

    return np.mean(ilds) if ilds else 0.0


def top1_category_ratio(data, k):
    """Top-1类别占比（类别集中度）"""
    ratios = []
    for _, group in data.groupby("user_id"):
        topk = get_topk(group, k)
        cats = []
        for c in topk["category"]:
            parsed = parse_categories(c)
            if parsed:
                cats.append(parsed[0])  # 取第一个类别

        if len(cats) == 0:
            continue

        top1_count = Counter(cats).most_common(1)[0][1]
        ratios.append(top1_count / len(cats))

    return np.mean(ratios) if ratios else 0.0


def avg_categories_per_user(data, k):
    """每用户平均唯一类别数@K"""
    counts = []
    for _, group in data.groupby("user_id"):
        topk = get_topk(group, k)
        unique_cats = set()
        for cats in topk["category"]:
            unique_cats.update(parse_categories(cats))
        counts.append(len(unique_cats))

    return np.mean(counts) if counts else 0.0


# ==================== 6. 新颖性指标 ====================

def novelty_at_k(data, k, pop_prob):
    """新颖性@K (基于流行度的self-information)"""
    if len(pop_prob) == 0:
        return 0.0

    novelties = []
    for _, group in data.groupby("user_id"):
        topk = get_topk(group, k)
        items = topk["item_id"].tolist()
        scores = []
        for item in items:
            if item in pop_prob.index:
                p = pop_prob[item]
                scores.append(-np.log(max(p, 1e-10)))
            else:
                scores.append(-np.log(1e-10))
        if scores:
            novelties.append(np.mean(scores))

    return np.mean(novelties) if novelties else 0.0


def unexpectedness_at_k(data, k, pop_prob):
    """意外性@K"""
    if len(pop_prob) == 0:
        return 0.0

    median_pop = np.median(pop_prob.values)

    unexpected = []
    for _, group in data.groupby("user_id"):
        topk = get_topk(group, k)
        items = topk["item_id"].tolist()
        scores = []
        for item in items:
            if item in pop_prob.index:
                scores.append(1.0 if pop_prob[item] < median_pop else 0.0)
            else:
                scores.append(1.0)
        if scores:
            unexpected.append(np.mean(scores))

    return np.mean(unexpected) if unexpected else 0.0


# ==================== 7. 覆盖率指标 ====================
def gini_at_k(data, k):
    """
    Gini-Index@K
    基于Top-K推荐列表的item曝光公平性

    输入:
        data:
            必须包含:
            user_id
            item_id
            final_score

        k:
            Top-K推荐数量

    输出:
        Gini值
        越低表示曝光越公平
    """

    # ==========================
    # 1. 获取每个用户Top-K推荐
    # ==========================
    topk = (
        data.sort_values(
            ["user_id", "final_score"],
            ascending=[True, False]
        )
        .groupby("user_id")
        .head(k)
    )


    if len(topk) == 0:
        return 0.0


    # ==========================
    # 2. 统计item曝光次数
    # ==========================
    #
    # item被多少用户推荐到Top-K
    #
    item_exposure = (
        topk["item_id"]
        .value_counts()
    )


    # ==========================
    # 3. 补充未曝光item
    # ==========================
    #
    # 没进入任何用户Top-K的item
    # 曝光次数应该为0
    #
    all_items = data["item_id"].unique()

    item_exposure = (
        item_exposure
        .reindex(all_items, fill_value=0)
    )


    # ==========================
    # 4. 排序
    # ==========================
    exposure = np.sort(
        item_exposure.values.astype(float)
    )


    n = len(exposure)

    total_exposure = exposure.sum()


    if n == 0 or total_exposure == 0:
        return 0.0


    # ==========================
    # 5. Gini公式
    # ==========================
    index = np.arange(1, n + 1)


    gini = (
        np.sum(
            (2 * index - n - 1) * exposure
        )
        /
        (n * total_exposure)
    )


    return gini

def item_coverage_at_k(data, k):
    """商品覆盖率@K"""
    topk = get_topk(data, k)
    recommended_items = set(topk["item_id"].unique())
    all_items = set(data["item_id"].unique())
    return len(recommended_items) / len(all_items) if all_items else 0.0


def user_coverage_at_k(data, k):
    """用户覆盖率（有至少一个正样本在Top-K中的用户比例）"""
    topk = get_topk(data, k)
    if len(topk) == 0:
        return 0.0
    users_with_hit = topk.groupby("user_id")["is_positive"].max()
    return (users_with_hit > 0).mean()


# ==================== 8. 按评分来源分组评估 ====================

def evaluate_by_source(data, k, pop_prob):
    """按评分来源分组评估"""
    source_col = COLUMN_MAP['score_source']

    if source_col not in data.columns:
        print("  ⚠️ 未找到评分来源列，跳过分组评估")
        return {}

    sources = data[source_col].dropna().unique()
    results_by_source = {}

    for source in sources:
        subset = data[data[source_col] == source]
        if len(subset) == 0:
            continue

        n_users = subset['user_id'].nunique()
        if n_users < 5:
            continue

        metrics = {
            '记录数': len(subset),
            '用户数': n_users,
            '商品数': subset['item_id'].nunique(),
            '正样本数': subset['is_positive'].sum(),
            'Precision': precision_at_k(subset, k),
            'Recall': recall_at_k(subset, k),
            'NDCG': ndcg_at_k(subset, k),
            'HR': hr_at_k(subset, k),
            'MRR': mrr_at_k(subset, k),
            'ItemCoverage': item_coverage_at_k(subset, k),
        }

        # 新颖性
        if len(pop_prob) > 0:
            metrics['Novelty'] = novelty_at_k(subset, k, pop_prob)

        # 类别相关指标
        if subset["category"].notna().sum() > 0 and (subset["category"] != '').sum() > 0:
            metrics['ILD_Category'] = ild_category_at_k(subset, k)
            metrics['CategoryCoverage'] = category_coverage_at_k(subset, k)
            metrics['AvgCategoriesPerUser'] = avg_categories_per_user(subset, k)

        # 描述多样性
        if (subset["description"].notna().sum() > 0 and
                (subset["description"] != '').sum() > 0):
            metrics['ILD_Description'] = ild_description_at_k(subset, k)

        results_by_source[str(source)] = metrics

    return results_by_source


# ==================== 9. 执行评估 ====================

K = ARGS.k

print(f"\n{'=' * 60}")
print(f"执行评估 (K={K})")
print(f"{'=' * 60}")

# 准确度指标
print("\n[1/4] 计算准确度指标...")
accuracy_metrics = {
    'Precision': precision_at_k(df, K),
    'Recall': recall_at_k(df, K),
    'NDCG': ndcg_at_k(df, K),
    'HR': hr_at_k(df, K),
    'MRR': mrr_at_k(df, K),
}

# 多样性指标
print("[2/4] 计算多样性指标...")
diversity_metrics = {}

has_category = (df["category"].notna().sum() > 0 and (df["category"] != '').sum() > 0)
has_description = (df["description"].notna().sum() > 0 and (df["description"] != '').sum() > 0)

if has_category:
    diversity_metrics.update({
        'ILD_Category': ild_category_at_k(df, K),
        'CategoryCoverage': category_coverage_at_k(df, K),
        'Top1_CategoryRatio': top1_category_ratio(df, K),
        'AvgCategoriesPerUser': avg_categories_per_user(df, K),
    })

if has_description:
    diversity_metrics['ILD_Description'] = ild_description_at_k(df, K)

# 新颖性指标
print("[3/4] 计算新颖性指标...")
novelty_metrics = {}
if len(item_popularity_prob) > 0:
    novelty_metrics = {
        'Novelty': novelty_at_k(df, K, item_popularity_prob),
        'Unexpectedness': unexpectedness_at_k(df, K, item_popularity_prob),
    }
else:
    novelty_metrics = {
        'Novelty': 0.0,
        'Unexpectedness': 0.0,
    }

# 覆盖率指标
print("[4/4] 计算覆盖率指标...")
coverage_metrics = {
    'ItemCoverage': item_coverage_at_k(df, K),
    'UserCoverage': user_coverage_at_k(df, K),
    'Gini': gini_at_k(df, K)
}

# 合并所有指标
all_results = {}
all_results.update(accuracy_metrics)
all_results.update(diversity_metrics)
all_results.update(novelty_metrics)
all_results.update(coverage_metrics)

# 按评分来源评估
print("\n按评分来源分组评估...")
source_results = evaluate_by_source(df, K, item_popularity_prob)

# ==================== 10. 输出结果 ====================

print(f"\n{'=' * 60}")
print(f"📊 评估结果 (K={K})")
print(f"{'=' * 60}")

# 整体结果
print(f"\n{'=' * 50}")
print("一、整体指标")
print(f"{'=' * 50}")
print(f"{'指标':<30} {'数值':>12}")
print("-" * 45)
for metric, value in all_results.items():
    print(f"{metric:<30} {value:>12.4f}")

# 按来源结果
if source_results:
    print(f"\n{'=' * 50}")
    print("二、按评分来源分组")
    print(f"{'=' * 50}")
    for source, metrics in source_results.items():
        print(f"\n  [{source}]")
        print(f"  记录数: {metrics['记录数']}, 用户数: {metrics['用户数']}, 商品数: {metrics['商品数']}")
        print(f"  正样本数: {metrics['正样本数']}")
        print(f"  {'指标':<28} {'数值':>12}")
        print(f"  " + "-" * 42)
        for metric, value in metrics.items():
            if metric not in ['记录数', '用户数', '商品数', '正样本数']:
                print(f"  {metric:<28} {value:>12.4f}")

# 正负样本分数对比
print(f"\n{'=' * 50}")
print("三、正负样本分数对比")
print(f"{'=' * 50}")
positive = df[df['is_positive'] == 1]
negative = df[df['is_positive'] == 0]
print(f"  正样本数: {len(positive)}, 负样本数: {len(negative)}")
print(f"  正样本 final_score 均值: {positive['final_score'].mean():.6f}")
print(f"  负样本 final_score 均值: {negative['final_score'].mean():.6f}")
print(f"  正负样本分数差: {(positive['final_score'].mean() - negative['final_score'].mean()):.6f}")

if COLUMN_MAP['explore_score'] in df.columns:
    explore_col = COLUMN_MAP['explore_score']
    pos_explore = positive[explore_col].dropna()
    neg_explore = negative[explore_col].dropna()
    if len(pos_explore) > 0:
        print(f"  正样本 explore_score 均值: {pos_explore.mean():.6f}")
    if len(neg_explore) > 0:
        print(f"  负样本 explore_score 均值: {neg_explore.mean():.6f}")

# Top-K推荐中的正样本分布
print(f"\n{'=' * 50}")
print("四、Top-K推荐统计")
print(f"{'=' * 50}")
topk = get_topk(df, K)
topk_pos = topk[topk['is_positive'] == 1]
print(f"  Top-{K}总记录数: {len(topk)}")
print(f"  Top-{K}正样本数: {len(topk_pos)}")
print(f"  Top-{K}正样本比例: {len(topk_pos) / len(topk) * 100:.2f}%")

# 每用户Top-K中正样本数分布
user_pos_in_topk = topk.groupby("user_id")["is_positive"].sum()
print(f"  每用户Top-{K}平均正样本数: {user_pos_in_topk.mean():.2f}")
print(f"  每用户Top-{K}正样本数分布:")
for i in range(K + 1):
    count = (user_pos_in_topk == i).sum()
    if count > 0:
        print(f"    {i}个正样本: {count} 用户")

# ==================== 11. 保存报告 ====================

output_dir = ARGS.output_dir or os.path.dirname(scores_file)
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
report_file = os.path.join(output_dir, f'evaluation_report_{timestamp}.txt')

with open(report_file, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("合并分数评估报告\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"数据文件: {scores_file}\n")
    f.write(f"K值: {K}\n\n")

    f.write(f"数据概览:\n")
    f.write(f"  总记录数: {len(df)}\n")
    f.write(f"  用户数: {df['user_id'].nunique()}\n")
    f.write(f"  商品数: {df['item_id'].nunique()}\n")
    f.write(f"  正样本数: {df['is_positive'].sum()}\n")
    f.write(f"  负样本数: {len(df) - df['is_positive'].sum()}\n\n")

    f.write("一、整体指标:\n")
    for metric, value in all_results.items():
        f.write(f"  {metric}: {value:.4f}\n")

    if source_results:
        f.write("\n二、按评分来源分组:\n")
        for source, metrics in source_results.items():
            f.write(f"\n  [{source}]\n")
            f.write(f"  记录数: {metrics['记录数']}, 用户数: {metrics['用户数']}\n")
            for metric, value in metrics.items():
                if metric not in ['记录数', '用户数', '商品数', '正样本数']:
                    f.write(f"    {metric}: {value:.4f}\n")

    f.write(f"\n三、正负样本:\n")
    f.write(f"  正样本 final_score 均值: {positive['final_score'].mean():.6f}\n")
    f.write(f"  负样本 final_score 均值: {negative['final_score'].mean():.6f}\n")

    f.write(f"\n四、Top-{K}推荐:\n")
    f.write(f"  正样本比例: {len(topk_pos) / len(topk) * 100:.2f}%\n")
    f.write(f"  每用户平均正样本数: {user_pos_in_topk.mean():.2f}\n")

print(f"\n✅ 评估报告已保存: {report_file}")

print(f"\n{'=' * 60}")
print("评估完成！")
print(f"{'=' * 60}")
