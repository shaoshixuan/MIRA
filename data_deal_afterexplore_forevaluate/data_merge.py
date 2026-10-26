# merge_scores.py
"""
合并KG分数和LLM分数结果
保留所有原始分数，新增 explore_score 列
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


def find_latest_llm_file(directory):
    """查找最新的LLM探索分数文件"""
    llm_files = []
    for f in os.listdir(directory):
        if f.startswith('llm_exploration_scores_') and f.endswith('.csv'):
            llm_files.append(f)

    if not llm_files:
        return None

    # 按时间戳排序，取最新的
    llm_files.sort(reverse=True)
    return os.path.join(directory, llm_files[0])


def merge_scores(kg_file, llm_file, output_dir):
    """
    合并KG分数和LLM分数

    策略：保留所有原始分数，新增 explore_score 列
    - KG中 kg_score != -10 且 != -5：explore_score = kg_score
    - KG中 kg_score == -10 或 == -5：explore_score = llm_score
    """
    print("=" * 60)
    print("合并KG分数和LLM分数（保留所有原始分数）")
    print("=" * 60)

    # 1. 加载KG分数
    print(f"\n1. 加载KG分数: {kg_file}")
    encodings = ['utf-8-sig', 'utf-8', 'gb18030', 'gbk']
    df_kg = None
    kg_sep = ','

    for enc in encodings:
        try:
            # 检测分隔符
            with open(kg_file, 'r', encoding=enc) as f:
                first_line = f.readline()
            if '\t' in first_line:
                kg_sep = '\t'

            df_kg = pd.read_csv(kg_file, encoding=enc, sep=kg_sep)
            print(f"  ✅ {enc} 编码, 分隔符: {'制表符' if kg_sep == '\t' else '逗号'}")
            break
        except Exception as e:
            continue

    if df_kg is None:
        raise ValueError("无法读取KG分数文件")

    # 清理列名（去除可能的空格）
    df_kg.columns = df_kg.columns.str.strip()

    print(f"  KG数据: {df_kg.shape[0]} 行, {df_kg.shape[1]} 列")
    print(f"  KG列名: {list(df_kg.columns)}")

    # 统计
    kg_neg10 = (df_kg['kg_interest_score'] == -10).sum()
    kg_neg5 = (df_kg['kg_interest_score'] == -5).sum()
    kg_valid = len(df_kg) - kg_neg10 - kg_neg5
    print(f"  KG分数分布:")
    print(f"    KG有效 ({kg_valid}): 使用KG分数")
    print(f"    需LLM (-10, 低交互): {kg_neg10}")
    print(f"    需LLM (-5, 增强): {kg_neg5}")

    # 2. 加载LLM分数
    print(f"\n2. 加载LLM分数: {llm_file}")
    df_llm = None
    for enc in encodings:
        try:
            df_llm = pd.read_csv(llm_file, encoding=enc)
            print(f"  ✅ {enc} 编码")
            break
        except:
            continue

    if df_llm is None:
        raise ValueError("无法读取LLM分数文件")

    # 清理列名
    df_llm.columns = df_llm.columns.str.strip()

    print(f"  LLM数据: {df_llm.shape[0]} 行, {df_llm.shape[1]} 列")
    print(f"  LLM列名: {list(df_llm.columns)}")

    # 统计LLM数据
    llm_scoring_methods = df_llm['scoring_method'].value_counts() if 'scoring_method' in df_llm.columns else {}
    print(f"  LLM评分方法分布:")
    for method, count in llm_scoring_methods.items():
        print(f"    {method}: {count}")

    # 3. 统一数据类型
    print("\n3. 统一数据格式...")

    df_kg['user_id'] = df_kg['user_id'].astype(str).str.strip()
    df_kg['item_id'] = df_kg['item_id'].astype(str).str.strip()
    df_llm['user_id'] = df_llm['user_id'].astype(str).str.strip()
    df_llm['item_id'] = df_llm['item_id'].astype(str).str.strip()

    # 创建合并键
    df_kg['merge_key'] = df_kg['user_id'] + '_' + df_kg['item_id']
    df_llm['merge_key'] = df_llm['user_id'] + '_' + df_llm['item_id']

    print(f"  KG唯一记录: {df_kg['merge_key'].nunique()}")
    print(f"  LLM唯一记录: {df_llm['merge_key'].nunique()}")

    # 检查重叠
    kg_keys = set(df_kg['merge_key'])
    llm_keys = set(df_llm['merge_key'])
    overlap = kg_keys & llm_keys
    print(f"  KG和LLM重叠记录: {len(overlap)}")

    # 4. 准备LLM数据用于合并（只保留需要的列）
    print("\n4. 准备LLM数据...")

    llm_merge_cols = ['merge_key', 'llm_exploration_score', 'scoring_method']
    available_llm_cols = [col for col in llm_merge_cols if col in df_llm.columns]
    df_llm_for_merge = df_llm[available_llm_cols].copy()

    # 重命名
    rename_dict = {}
    if 'scoring_method' in df_llm_for_merge.columns:
        rename_dict['scoring_method'] = 'llm_scoring_method'
    if 'llm_exploration_score' in df_llm_for_merge.columns:
        rename_dict['llm_exploration_score'] = 'llm_exploration_score'

    df_llm_for_merge = df_llm_for_merge.rename(columns=rename_dict)

    # 5. 左连接：KG为主表，LLM补充
    print("\n5. 合并数据...")

    df_merged = df_kg.merge(df_llm_for_merge, on='merge_key', how='left')

    print(f"  合并后记录数: {len(df_merged)}")
    print(f"  有LLM分数的记录: {df_merged['llm_exploration_score'].notna().sum()}")
    print(f"  无LLM分数的记录: {df_merged['llm_exploration_score'].isna().sum()}")

    # 6. 生成 explore_score 列
    print("\n6. 生成 explore_score 列...")

    def determine_explore_score(row):
        """
        确定探索分数：
        - kg_score 有效 (!= -10 且 != -5)：使用 kg_score
        - kg_score 无效 (== -10 或 == -5)：使用 llm_score
        """
        kg_score = row['kg_interest_score']

        if kg_score == -10 or kg_score == -5:
            # 需要LLM的情况
            llm_score = row.get('llm_exploration_score', np.nan)
            return llm_score if pd.notna(llm_score) else np.nan
        else:
            # KG有效的情况
            return kg_score

    def determine_score_source(row):
        """确定分数来源"""
        kg_score = row['kg_interest_score']

        if kg_score == -10:
            return 'LLM_low_interaction'
        elif kg_score == -5:
            return 'LLM_enhancement'
        elif pd.notna(kg_score):
            return 'KG'
        else:
            return 'Unknown'

    df_merged['explore_score'] = df_merged.apply(determine_explore_score, axis=1)
    df_merged['explore_score_source'] = df_merged.apply(determine_score_source, axis=1)

    # 7. 重新计算 final_combined_score
    print("\n7. 重新计算 final_combined_score...")

    def compute_final_score(row):
        """
        计算最终综合分数
        - KG有效：pctr * 0.7 + kg * 0.3
        - LLM(低交互)：pctr * 0.3 + llm * 0.7
        - LLM(增强)：pctr * 0.5 + llm * 0.5
        """
        pctr = row['original_pctr_score']
        kg_score = row['kg_interest_score']
        llm_score = row.get('llm_exploration_score', np.nan)

        if kg_score == -10:
            if pd.notna(llm_score):
                return pctr * 0.3 + llm_score * 0.7
            else:
                return pctr
        elif kg_score == -5:
            if pd.notna(llm_score):
                return pctr * 0.5 + llm_score * 0.5
            else:
                return pctr * 0.85
        elif pd.notna(kg_score):
            return pctr * 0.7 + kg_score * 0.3
        else:
            return pctr

    # 保留原始 final_combined_score
    df_merged['original_final_score'] = df_merged['final_combined_score']
    # 重新计算
    df_merged['final_combined_score'] = df_merged.apply(compute_final_score, axis=1)

    # 8. 清理列
    print("\n8. 整理输出列...")

    # 删除临时列
    df_merged = df_merged.drop(columns=['merge_key'])

    # 确定输出列顺序
    output_columns = [
        'user_id', 'item_id', 'asin', 'title', 'description',
        'is_positive', 'original_pctr_score',
        'kg_interest_score',
        'llm_exploration_score',
        'explore_score',
        'explore_score_source',
        'original_final_score',
        'final_combined_score',
        'llm_scoring_method'
    ]

    # 只保留存在的列
    available_columns = [col for col in output_columns if col in df_merged.columns]
    df_merged = df_merged[available_columns]

    # 9. 排序
    print("\n9. 排序...")
    df_merged = df_merged.sort_values(
        ['user_id', 'final_combined_score'],
        ascending=[True, False]
    )

    # 10. 保存结果
    print("\n10. 保存合并结果...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    output_file = os.path.join(output_dir, f'merged_scores_final_{timestamp}.csv')
    df_merged.to_csv(output_file, index=False, encoding='utf-8-sig')

    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✅ 合并结果已保存: {output_file}")
    print(f"   文件大小: {file_size:.2f} MB")
    print(f"   总记录数: {len(df_merged)}")
    print(f"   用户数: {df_merged['user_id'].nunique()}")
    print(f"   列数: {len(df_merged.columns)}")

    # 11. 统计报告
    print("\n" + "=" * 60)
    print("📊 合并统计报告")
    print("=" * 60)

    print(f"\n数据概览:")
    print(f"  总记录数: {len(df_merged)}")
    print(f"  唯一用户数: {df_merged['user_id'].nunique()}")
    print(f"  唯一商品数: {df_merged['item_id'].nunique()}")

    # 探索分数来源分布
    print(f"\n探索分数来源分布:")
    source_counts = df_merged['explore_score_source'].value_counts()
    for source, count in source_counts.items():
        print(f"  {source}: {count} ({count / len(df_merged) * 100:.1f}%)")

    # 探索分数统计
    explore_scores = df_merged['explore_score'].dropna()
    print(f"\n探索分数统计 (explore_score):")
    print(f"  有效值: {len(explore_scores)}")
    print(f"  缺失值: {df_merged['explore_score'].isna().sum()}")
    if len(explore_scores) > 0:
        print(f"  范围: [{explore_scores.min():.6f}, {explore_scores.max():.6f}]")
        print(f"  均值: {explore_scores.mean():.6f}")
        print(f"  中位数: {explore_scores.median():.6f}")
        print(f"  标准差: {explore_scores.std():.6f}")

    # 按来源分别统计
    for source in source_counts.index:
        subset = df_merged[df_merged['explore_score_source'] == source]
        scores = subset['explore_score'].dropna()
        if len(scores) > 0:
            print(f"\n  {source}:")
            print(f"    数量: {len(scores)}")
            print(f"    范围: [{scores.min():.6f}, {scores.max():.6f}]")
            print(f"    均值: {scores.mean():.6f}")
            print(f"    中位数: {scores.median():.6f}")

    # 最终分数统计
    final_scores = df_merged['final_combined_score'].dropna()
    print(f"\n最终分数统计 (final_combined_score):")
    print(f"  范围: [{final_scores.min():.4f}, {final_scores.max():.4f}]")
    print(f"  均值: {final_scores.mean():.4f}")
    print(f"  中位数: {final_scores.median():.4f}")
    print(f"  标准差: {final_scores.std():.4f}")

    # 正负样本分析
    positive = df_merged[df_merged['is_positive'] == 1]
    negative = df_merged[df_merged['is_positive'] == 0]
    print(f"\n正负样本分析:")
    print(f"  正样本: {len(positive)} ({len(positive) / len(df_merged) * 100:.1f}%)")
    print(f"  负样本: {len(negative)} ({len(negative) / len(df_merged) * 100:.1f}%)")

    if len(positive) > 0:
        pos_explore = positive['explore_score'].dropna()
        print(f"  正样本 explore_score 均值: {pos_explore.mean():.6f}" if len(
            pos_explore) > 0 else "  正样本 explore_score: 无有效值")
        print(f"  正样本 final_score 均值: {positive['final_combined_score'].mean():.4f}")

    if len(negative) > 0:
        neg_explore = negative['explore_score'].dropna()
        print(f"  负样本 explore_score 均值: {neg_explore.mean():.6f}" if len(
            neg_explore) > 0 else "  负样本 explore_score: 无有效值")
        print(f"  负样本 final_score 均值: {negative['final_combined_score'].mean():.4f}")

    # 每用户记录数
    user_counts = df_merged.groupby('user_id').size()
    print(f"\n每用户记录数:")
    print(f"  均值: {user_counts.mean():.1f}")
    print(f"  中位数: {user_counts.median():.0f}")
    print(f"  范围: [{user_counts.min()}, {user_counts.max()}]")

    # 显示前5行
    print(f"\n前5行示例:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 300)
    pd.set_option('display.max_colwidth', 30)
    print(df_merged.head(5).to_string())

    # 保存统计报告
    report_file = os.path.join(output_dir, f'merge_report_{timestamp}.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("KG+LLM分数合并报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"KG文件: {kg_file}\n")
        f.write(f"LLM文件: {llm_file}\n")
        f.write(f"输出文件: {output_file}\n\n")
        f.write(f"总记录数: {len(df_merged)}\n")
        f.write(f"唯一用户数: {df_merged['user_id'].nunique()}\n")
        f.write(f"唯一商品数: {df_merged['item_id'].nunique()}\n\n")

        f.write("探索分数来源分布:\n")
        for source, count in source_counts.items():
            f.write(f"  {source}: {count} ({count / len(df_merged) * 100:.1f}%)\n")

        f.write(f"\n分数统计:\n")
        f.write(
            f"  explore_score: [{explore_scores.min():.6f}, {explore_scores.max():.6f}], mean={explore_scores.mean():.6f}\n")
        f.write(
            f"  final_score: [{final_scores.min():.4f}, {final_scores.max():.4f}], mean={final_scores.mean():.4f}\n")

        f.write(f"\n正负样本:\n")
        f.write(
            f"  正样本: {len(positive)}, explore_mean={positive['explore_score'].mean():.6f}, final_mean={positive['final_combined_score'].mean():.4f}\n")
        f.write(
            f"  负样本: {len(negative)}, explore_mean={negative['explore_score'].mean():.6f}, final_mean={negative['final_combined_score'].mean():.4f}\n")

    print(f"\n✅ 统计报告已保存: {report_file}")

    return output_file


def main():
    """主函数"""
    # ==================== 文件路径配置 ====================
    # KG分数文件
    kg_file = 'D:/graduation_recsys/data/KG-related_Files/kg_interest_scores/reranked_predictions_with_kg.csv'

    # LLM分数目录
    llm_dir = 'D:/graduation_recsys/data/KG-related_Files/llm_interest_scores'

    # 输出目录（放在KG目录下）
    output_dir = 'D:/graduation_recsys/data_deal_afterexplore_forevaluate'

    # ==================== 检查文件 ====================
    print("=" * 60)
    print("检查文件路径")
    print("=" * 60)

    if not os.path.exists(kg_file):
        print(f"❌ KG文件不存在: {kg_file}")
        return

    print(f"✅ KG文件: {kg_file}")

    if not os.path.exists(llm_dir):
        print(f"❌ LLM目录不存在: {llm_dir}")
        return

    # 查找LLM文件
    llm_file = find_latest_llm_file(llm_dir)

    if llm_file is None:
        # 如果自动查找失败，尝试手动指定
        manual_llm_file = os.path.join(llm_dir, 'llm_exploration_scores_20260728_043301.csv')
        if os.path.exists(manual_llm_file):
            llm_file = manual_llm_file
            print(f"✅ LLM文件 (手动指定): {llm_file}")
        else:
            print(f"❌ 未找到LLM分数文件")
            print(f"   请确认目录 {llm_dir} 下有 llm_exploration_scores_*.csv 文件")
            return
    else:
        print(f"✅ LLM文件 (自动查找): {llm_file}")

    print(f"✅ 输出目录: {output_dir}")

    # ==================== 合并 ====================
    merged_file = merge_scores(kg_file, llm_file, output_dir)

    if merged_file:
        print(f"\n{'=' * 60}")
        print(f"✅ 合并完成！")
        print(f"   输出文件: {merged_file}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()