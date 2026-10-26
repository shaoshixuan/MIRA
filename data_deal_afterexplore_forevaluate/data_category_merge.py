# add_category_to_scores.py
"""
为合并分数表添加商品类别信息
通过kg_other_triples.txt和category.txt的映射关系
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict


def load_category_mapping(category_file):
    """
    加载类别映射文件
    格式: category::Category Name\tentity_id

    Returns:
        dict: {entity_id: category_name}
    """
    print(f"\n加载类别文件: {category_file}")

    entity_to_category = {}
    line_count = 0

    with open(category_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="加载类别映射"):
            line = line.strip()
            if not line:
                continue

            line_count += 1

            # 格式: category::Category Name\tentity_id
            parts = line.split('\t')
            if len(parts) == 2:
                category_part = parts[0]  # "category::Category Name"
                entity_id_str = parts[1]  # "212405"

                # 提取类别名称
                if '::' in category_part:
                    category_name = category_part.split('::', 1)[1]
                else:
                    category_name = category_part

                # 提取entity_id
                try:
                    entity_id = str(int(entity_id_str.strip()))
                except:
                    entity_id = entity_id_str.strip()

                # 清理类别名称
                category_name = category_name.strip()

                entity_to_category[entity_id] = category_name

    print(f"  加载了 {len(entity_to_category)} 个类别映射 (共{line_count}行)")

    # 显示示例
    sample_items = list(entity_to_category.items())[:5]
    print(f"  示例:")
    for entity_id, category in sample_items:
        print(f"    entity_id={entity_id} -> {category}")

    return entity_to_category


def load_item_category_relations(triples_file, target_relation=8):
    """
    从三元组文件中加载商品-类别关系
    筛选关系ID为target_relation的数据

    格式: item_id\tentity_id\trelation_id

    Args:
        triples_file: 三元组文件路径
        target_relation: 目标关系ID（默认为8）

    Returns:
        dict: {item_id: [entity_id1, entity_id2, ...]}
    """
    print(f"\n加载三元组文件: {triples_file}")
    print(f"  筛选关系ID: {target_relation}")

    item_to_entities = defaultdict(list)
    line_count = 0
    matched_count = 0

    with open(triples_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="加载三元组"):
            line = line.strip()
            if not line:
                continue

            line_count += 1

            # 格式: item_id\tentity_id\trelation_id
            parts = line.split('\t')
            if len(parts) == 3:
                item_id = parts[0].strip()
                entity_id = parts[1].strip()
                relation_id = parts[2].strip()

                # 筛选关系ID
                try:
                    relation_id_int = int(relation_id)
                except:
                    continue

                if relation_id_int == target_relation:
                    item_to_entities[item_id].append(entity_id)
                    matched_count += 1

    print(f"  总行数: {line_count}")
    print(f"  关系ID={target_relation}的记录: {matched_count}")
    print(f"  涉及商品数: {len(item_to_entities)}")

    # 统计每个商品的类别数
    category_counts = [len(entities) for entities in item_to_entities.values()]
    if category_counts:
        print(f"  每商品平均类别数: {np.mean(category_counts):.2f}")
        print(f"  每商品类别数范围: [{min(category_counts)}, {max(category_counts)}]")

    return item_to_entities


def add_categories_to_scores(scores_file, item_to_entities, entity_to_category, output_dir):
    """
    将类别信息添加到合并分数表中

    Args:
        scores_file: 合并分数CSV文件
        item_to_entities: 商品到实体ID的映射
        entity_to_category: 实体ID到类别名称的映射
        output_dir: 输出目录

    Returns:
        str: 输出文件路径
    """
    print(f"\n加载合并分数文件: {scores_file}")

    # 加载分数数据
    encodings = ['utf-8-sig', 'utf-8', 'gb18030', 'gbk']
    df_scores = None

    for enc in encodings:
        try:
            df_scores = pd.read_csv(scores_file, encoding=enc)
            print(f"  ✅ 使用 {enc} 编码")
            break
        except:
            continue

    if df_scores is None:
        raise ValueError("无法读取合并分数文件")

    # 清理列名
    df_scores.columns = df_scores.columns.str.strip()

    print(f"  数据: {df_scores.shape[0]} 行, {df_scores.shape[1]} 列")
    print(f"  列名: {list(df_scores.columns)}")

    # 统一item_id格式
    df_scores['item_id'] = df_scores['item_id'].astype(str).str.strip()

    # 添加类别信息
    print("\n添加类别信息...")

    categories_list = []
    category_entity_ids_list = []
    matched_count = 0

    for idx, row in tqdm(df_scores.iterrows(), total=len(df_scores), desc="匹配类别"):
        item_id = row['item_id']

        # 获取该商品对应的类别实体ID
        entity_ids = item_to_entities.get(item_id, [])

        if entity_ids:
            # 将实体ID转换为类别名称
            category_names = []
            for entity_id in entity_ids:
                category_name = entity_to_category.get(entity_id, '')
                if category_name:
                    category_names.append(category_name)

            if category_names:
                # 多个类别用 | 分隔
                categories_list.append(' | '.join(category_names))
                category_entity_ids_list.append(' | '.join(entity_ids))
                matched_count += 1
            else:
                categories_list.append('')
                category_entity_ids_list.append('')
        else:
            categories_list.append('')
            category_entity_ids_list.append('')

    df_scores['category'] = categories_list
    df_scores['category_entity_ids'] = category_entity_ids_list

    # 统计
    total = len(df_scores)
    print(f"\n类别匹配统计:")
    print(f"  有类别的记录: {matched_count} ({matched_count / total * 100:.1f}%)")
    print(f"  无类别的记录: {total - matched_count} ({(total - matched_count) / total * 100:.1f}%)")

    # 类别统计
    all_categories = []
    for cats in categories_list:
        if cats:
            all_categories.extend([c.strip() for c in cats.split('|')])

    from collections import Counter
    category_counts = Counter(all_categories)

    print(f"\n唯一类别数: {len(category_counts)}")
    print(f"出现最多的10个类别:")
    for cat, count in category_counts.most_common(10):
        print(f"  {cat}: {count}")

    # 保存结果
    print(f"\n保存结果...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'merged_scores_with_categories_{timestamp}.csv')

    df_scores.to_csv(output_file, index=False, encoding='utf-8-sig')

    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✅ 结果已保存: {output_file}")
    print(f"   文件大小: {file_size:.2f} MB")
    print(f"   记录数: {len(df_scores)}")
    print(f"   列数: {len(df_scores.columns)}")

    # 显示示例
    print(f"\n前5行示例:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 300)
    pd.set_option('display.max_colwidth', 30)
    display_cols = ['user_id', 'item_id', 'title', 'explore_score_source',
                    'explore_score', 'category']
    available_cols = [col for col in display_cols if col in df_scores.columns]
    print(df_scores[available_cols].head(5).to_string())

    # 保存统计报告
    report_file = os.path.join(output_dir, f'category_report_{timestamp}.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("类别添加报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"总记录数: {len(df_scores)}\n")
        f.write(f"有类别的记录: {matched_count} ({matched_count / total * 100:.1f}%)\n")
        f.write(f"唯一类别数: {len(category_counts)}\n\n")
        f.write("出现最多的20个类别:\n")
        for cat, count in category_counts.most_common(20):
            f.write(f"  {cat}: {count}\n")

    print(f"\n✅ 统计报告已保存: {report_file}")

    return output_file


def main():
    """主函数"""
    # ==================== 文件路径配置 ====================
    data_dir = 'D:/graduation_recsys/data_deal_afterexplore_forevaluate'

    triples_file = os.path.join(data_dir, 'kg_other_triples.txt')
    category_file = os.path.join(data_dir, 'category.txt')
    scores_file = os.path.join(data_dir, 'merged_scores_final_20260811_164422.csv')
    output_dir = data_dir

    # ==================== 检查文件 ====================
    print("=" * 60)
    print("检查文件路径")
    print("=" * 60)

    for file_path, file_name in [
        (triples_file, 'kg_other_triples.txt'),
        (category_file, 'category.txt'),
        (scores_file, 'merged_scores_final_20260811_164422.csv')
    ]:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✅ {file_name}: {file_path} ({file_size:.2f} MB)")
        else:
            print(f"❌ {file_name}: {file_path} (文件不存在!)")
            return

    # ==================== 处理步骤 ====================
    print("\n" + "=" * 60)
    print("步骤1: 加载类别映射 (entity_id -> category)")
    print("=" * 60)
    entity_to_category = load_category_mapping(category_file)

    print("\n" + "=" * 60)
    print("步骤2: 加载商品-类别关系 (item_id -> entity_id)")
    print("=" * 60)
    # 关系ID=8 表示商品属于某个类别
    item_to_entities = load_item_category_relations(triples_file, target_relation=8)

    print("\n" + "=" * 60)
    print("步骤3: 为合并分数表添加类别信息")
    print("=" * 60)
    output_file = add_categories_to_scores(
        scores_file,
        item_to_entities,
        entity_to_category,
        output_dir
    )

    if output_file:
        print(f"\n{'=' * 60}")
        print(f"✅ 处理完成！")
        print(f"   输出文件: {output_file}")
        print(f"{'=' * 60}")

        # 验证：显示一个商品的完整信息
        print(f"\n验证 - 显示有类别的商品示例:")
        df = pd.read_csv(output_file, encoding='utf-8-sig')
        df_with_cat = df[df['category'].notna() & (df['category'] != '')]
        if len(df_with_cat) > 0:
            sample = df_with_cat.iloc[0]
            print(f"  user_id: {sample['user_id']}")
            print(f"  item_id: {sample['item_id']}")
            print(f"  title: {sample['title'][:80]}...")
            print(f"  category: {sample['category']}")
            print(f"  explore_score: {sample['explore_score']}")


if __name__ == "__main__":
    main()