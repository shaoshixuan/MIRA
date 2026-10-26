# import pandas as pd
# import json
# import os
# from tqdm import tqdm
#
#
# def main():
#     """
#     主函数：完成数据拼接任务
#     步骤：
#     1. 构建 item_id -> asin 映射（从 kg_items_Toys_and_Games.txt）
#     2. 构建 asin -> {title, description} 映射（从 meta_Toys_and_Games.jsonl）
#     3. 将映射应用到 val_predictions_4000_for_reranking.csv
#     """
#
#     # ==================== 文件路径配置 ====================
#     base_dir = "D:/test"  # 根据实际情况修改
#
#     kg_file = os.path.join(base_dir, "kg_items_Toys_and_Games.txt")
#     jsonl_file = os.path.join(base_dir, "meta_Toys_and_Games.jsonl")
#     csv_file = os.path.join(base_dir, "val_predictions_4000_for_reranking.csv")
#     output_file = os.path.join(base_dir, "val_predictions_with_metadata.csv")
#
#     # ==================== 步骤1: 构建 item_id -> asin 映射 ====================
#     print("=" * 80)
#     print("步骤1: 构建 item_id -> asin 映射")
#     print("=" * 80)
#
#     item_id_to_asin = {}
#
#     with open(kg_file, 'r', encoding='utf-8') as f:
#         for line in tqdm(f, desc="读取kg_items文件", total=98734):  # 总共98734行
#             line = line.strip()
#             if not line:
#                 continue
#
#             # 格式: product::B010VYNN9Y\t98734
#             parts = line.split('\t')
#             if len(parts) == 2:
#                 product_part = parts[0]  # "product::B010VYNN9Y"
#                 item_id_str = parts[1]  # "98734"
#
#                 # 提取asin
#                 if '::' in product_part:
#                     asin = product_part.split('::')[1]
#                 else:
#                     asin = product_part
#
#                 # 提取item_id
#                 item_id = int(item_id_str.strip())
#
#                 # 建立映射
#                 item_id_to_asin[item_id] = asin
#
#     print(f"✓ 完成！共加载 {len(item_id_to_asin)} 个映射")
#     print(f"  示例: item_id 98743 -> asin {item_id_to_asin.get(98743, 'N/A')}")
#     print(f"  示例: item_id 98734 -> asin {item_id_to_asin.get(98734, 'N/A')}")
#
#     # ==================== 步骤2: 构建 asin -> metadata 映射 ====================
#     print("\n" + "=" * 80)
#     print("步骤2: 构建 asin -> {title, description} 映射")
#     print("=" * 80)
#
#     asin_to_metadata = {}
#     line_count = 0
#     error_count = 0
#
#     with open(jsonl_file, 'r', encoding='utf-8') as f:
#         for line in tqdm(f, desc="读取jsonl文件"):
#             line = line.strip()
#             if not line:
#                 continue
#
#             line_count += 1
#
#             try:
#                 data = json.loads(line)
#
#                 # 获取 parent_asin 字段
#                 parent_asin = data.get('parent_asin', None)
#
#                 if parent_asin:
#                     # 提取 title
#                     title = data.get('title', '')
#
#                     # 提取 description（可能是列表，需要转换为字符串）
#                     description = data.get('description', '')
#                     if isinstance(description, list):
#                         description = ' '.join(description)
#
#                     # 存储
#                     asin_to_metadata[parent_asin] = {
#                         'title': title,
#                         'description': description
#                     }
#
#             except json.JSONDecodeError:
#                 error_count += 1
#             except Exception:
#                 error_count += 1
#
#     print(f"✓ 完成！共处理 {line_count} 行")
#     print(f"  成功索引: {len(asin_to_metadata)} 个asin")
#     print(f"  解析错误: {error_count} 行")
#
#     # 显示示例
#     sample_asins = ['B00E6GNPNS', 'B010VYNN9Y']  # 你之前的示例
#     for asin in sample_asins:
#         if asin in asin_to_metadata:
#             meta = asin_to_metadata[asin]
#             print(f"  示例 asin={asin}: title='{meta['title'][:80]}...'")
#         else:
#             print(f"  示例 asin={asin}: 未在jsonl中找到")
#
#     # ==================== 步骤3: 处理CSV文件 ====================
#     print("\n" + "=" * 80)
#     print("步骤3: 处理CSV文件，添加asin、title、description")
#     print("=" * 80)
#
#     # 读取CSV（自动检测分隔符）
#     print("正在读取CSV文件...")
#
#     # 先检测分隔符
#     with open(csv_file, 'r', encoding='utf-8') as f:
#         first_line = f.readline()
#
#     if '\t' in first_line:
#         separator = '\t'
#         print("检测到分隔符: 制表符")
#     else:
#         separator = ','
#         print("检测到分隔符: 逗号")
#
#     # 读取数据
#     df = pd.read_csv(csv_file, sep=separator)
#
#     # 清理列名（去除空格）
#     df.columns = df.columns.str.strip()
#
#     print(f"原始数据: {df.shape[0]} 行, {df.shape[1]} 列")
#     print(f"列名: {list(df.columns)}")
#
#     # --- 添加 asin 列 ---
#     print("\n添加 asin 列...")
#     df['asin'] = df['item_id'].map(item_id_to_asin)
#
#     matched_asin_count = df['asin'].notna().sum()
#     total_count = len(df)
#     print(f"  item_id -> asin 匹配: {matched_asin_count}/{total_count} ({matched_asin_count / total_count * 100:.2f}%)")
#
#     # --- 添加 title 和 description 列 ---
#     print("添加 title 和 description 列...")
#
#     # 使用向量化操作提高效率
#     df['title'] = df['asin'].map(
#         lambda x: asin_to_metadata.get(x, {}).get('title', '') if pd.notna(x) else ''
#     )
#     df['description'] = df['asin'].map(
#         lambda x: asin_to_metadata.get(x, {}).get('description', '') if pd.notna(x) else ''
#     )
#
#     # 统计
#     matched_title_count = (df['title'] != '').sum()
#     matched_desc_count = (df['description'] != '').sum()
#     print(f"  asin -> title 匹配: {matched_title_count}/{total_count} ({matched_title_count / total_count * 100:.2f}%)")
#     print(
#         f"  asin -> description 匹配: {matched_desc_count}/{total_count} ({matched_desc_count / total_count * 100:.2f}%)")
#
#     # ==================== 步骤4: 保存结果 ====================
#     print("\n" + "=" * 80)
#     print("步骤4: 保存结果")
#     print("=" * 80)
#
#     df.to_csv(output_file, index=False, sep='\t')
#     print(f"✓ 结果已保存到: {output_file}")
#     print(f"  行数: {len(df)}")
#     print(f"  列数: {len(df.columns)}")
#     print(f"  新增列: asin, title, description")
#
#     # ==================== 步骤5: 结果验证 ====================
#     print("\n" + "=" * 80)
#     print("步骤5: 结果验证")
#     print("=" * 80)
#
#     # 展示前10行
#     print("\n前10行数据预览:")
#     display_cols = ['user_id', 'item_id', 'asin', 'title', 'is_positive', 'pctr_score']
#     available_cols = [col for col in display_cols if col in df.columns]
#     print(df[available_cols].head(10).to_string())
#
#     # 展示一个正样本的完整信息
#     print("\n正样本示例 (is_positive=1):")
#     positive_samples = df[df['is_positive'] == 1]
#     if len(positive_samples) > 0:
#         sample = positive_samples.iloc[0]
#         print(f"  user_id: {sample['user_id']}")
#         print(f"  item_id: {sample['item_id']}")
#         print(f"  asin: {sample['asin']}")
#         print(f"  title: {sample['title']}")
#         print(f"  description: {str(sample['description'])[:200]}...")
#         print(f"  pctr_score: {sample['pctr_score']}")
#
#     # 统计信息
#     print("\n匹配统计:")
#     print(f"  总行数: {len(df)}")
#     print(f"  唯一用户数: {df['user_id'].nunique()}")
#     print(f"  唯一item数: {df['item_id'].nunique()}")
#     print(f"  正样本数: {(df['is_positive'] == 1).sum()}")
#     print(f"  负样本数: {(df['is_positive'] == 0).sum()}")
#     print(f"  完整匹配(有asin+title)的行数: {((df['asin'].notna()) & (df['title'] != '')).sum()}")
#
#     # 未匹配分析
#     no_asin = df['asin'].isna().sum()
#     no_title = ((df['asin'].notna()) & (df['title'] == '')).sum()
#
#     if no_asin > 0:
#         print(f"\n  未匹配到asin的item_id示例:")
#         no_asin_items = df[df['asin'].isna()]['item_id'].value_counts().head(5)
#         for item_id, count in no_asin_items.items():
#             print(f"    item_id={item_id}: 出现{count}次")
#
#     if no_title > 0:
#         print(f"\n  有asin但无title的asin示例:")
#         no_title_asins = df[(df['asin'].notna()) & (df['title'] == '')]['asin'].value_counts().head(5)
#         for asin, count in no_title_asins.items():
#             print(f"    asin={asin}: 出现{count}次")
#
#     print("\n" + "=" * 80)
#     print("全部完成！")
#     print("=" * 80)
#
#     return df
#
#
# if __name__ == "__main__":
#     df = main()

import pandas as pd
import json
import os
from tqdm import tqdm


def main():
    """
    主函数：完成数据拼接任务
    步骤：
    1. 构建 item_id -> asin 映射（从 kg_items_Toys_and_Games.txt）
    2. 构建 asin -> {title, description} 映射（从 meta_Toys_and_Games.jsonl）
    3. 将映射应用到 val_predictions_4000_for_reranking.csv
    """

    # ==================== 文件路径配置 ====================
    base_dir = "D:/graduation_recsys/data_deal_afterpctr_forllm"  # 根据实际情况修改

    kg_file = os.path.join(base_dir, "kg_items_Automotive.txt")# 更换其他数据集时需要更改文件名
    jsonl_file = os.path.join(base_dir, "meta_Automotive.jsonl")#更换其他数据集时需要更改文件名
    csv_file = os.path.join(base_dir, "val_predictions_4000_for_reranking.csv")
    output_file = os.path.join(base_dir, "val_predictions_with_metadata.csv")

    # ==================== 步骤1: 构建 item_id -> asin 映射 ====================
    print("=" * 80)
    print("步骤1: 构建 item_id -> asin 映射")
    print("=" * 80)

    item_id_to_asin = {}

    with open(kg_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取kg_items文件"):
            line = line.strip()
            if not line:
                continue

            # 格式: product::B010VYNN9Y\t98734
            parts = line.split('\t')
            if len(parts) == 2:
                product_part = parts[0]  # "product::B010VYNN9Y"
                item_id_str = parts[1]  # "98734"

                # 提取asin
                if '::' in product_part:
                    asin = product_part.split('::')[1]
                else:
                    asin = product_part

                # 提取item_id
                item_id = int(item_id_str.strip())

                # 建立映射
                item_id_to_asin[item_id] = asin

    print(f"✓ 完成！共加载 {len(item_id_to_asin)} 个映射")
    print(f"  示例: item_id 98743 -> asin {item_id_to_asin.get(98743, 'N/A')}")
    print(f"  示例: item_id 98734 -> asin {item_id_to_asin.get(98734, 'N/A')}")

    # ==================== 步骤2: 构建 asin -> metadata 映射 ====================
    print("\n" + "=" * 80)
    print("步骤2: 构建 asin -> {title, description} 映射")
    print("=" * 80)

    asin_to_metadata = {}
    line_count = 0
    error_count = 0

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取jsonl文件"):
            line = line.strip()
            if not line:
                continue

            line_count += 1

            try:
                data = json.loads(line)

                # 获取 parent_asin 字段
                parent_asin = data.get('parent_asin', None)

                if parent_asin:
                    # 提取 title，清理其中的换行符和制表符
                    title = data.get('title', '')
                    if isinstance(title, str):
                        # 清理标题中的特殊字符，避免破坏CSV格式
                        title = title.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

                    # 提取 description（可能是列表，需要转换为字符串）
                    description = data.get('description', '')
                    if isinstance(description, list):
                        description = ' '.join(description)
                    if isinstance(description, str):
                        # 清理描述中的特殊字符
                        description = description.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

                    # 存储
                    asin_to_metadata[parent_asin] = {
                        'title': title,
                        'description': description
                    }

            except json.JSONDecodeError:
                error_count += 1
            except Exception:
                error_count += 1

    print(f"✓ 完成！共处理 {line_count} 行")
    print(f"  成功索引: {len(asin_to_metadata)} 个asin")
    print(f"  解析错误: {error_count} 行")

    # 显示示例
    sample_asins = ['B00E6GNPNS', 'B010VYNN9Y']
    for asin in sample_asins:
        if asin in asin_to_metadata:
            meta = asin_to_metadata[asin]
            print(f"  示例 asin={asin}: title='{meta['title'][:80]}...'")
        else:
            print(f"  示例 asin={asin}: 未在jsonl中找到")

    # ==================== 步骤3: 处理CSV文件 ====================
    print("\n" + "=" * 80)
    print("步骤3: 处理CSV文件，添加asin、title、description")
    print("=" * 80)

    # 读取CSV（自动检测分隔符）
    print("正在读取CSV文件...")

    # 先检测分隔符
    with open(csv_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()

    if '\t' in first_line:
        separator = '\t'
        print("检测到分隔符: 制表符")
    else:
        separator = ','
        print("检测到分隔符: 逗号")

    # 读取数据
    df = pd.read_csv(csv_file, sep=separator)

    # 清理列名（去除空格）
    df.columns = df.columns.str.strip()

    print(f"原始数据: {df.shape[0]} 行, {df.shape[1]} 列")
    print(f"列名: {list(df.columns)}")

    # --- 添加 asin 列 ---
    print("\n添加 asin 列...")
    df['asin'] = df['item_id'].map(item_id_to_asin)

    matched_asin_count = df['asin'].notna().sum()
    total_count = len(df)
    print(f"  item_id -> asin 匹配: {matched_asin_count}/{total_count} ({matched_asin_count / total_count * 100:.2f}%)")

    # --- 添加 title 和 description 列 ---
    print("添加 title 和 description 列...")

    # 使用向量化操作提高效率
    df['title'] = df['asin'].map(
        lambda x: asin_to_metadata.get(x, {}).get('title', '') if pd.notna(x) else ''
    )
    df['description'] = df['asin'].map(
        lambda x: asin_to_metadata.get(x, {}).get('description', '') if pd.notna(x) else ''
    )

    # 清理新增列中的特殊字符（再次确保）
    df['title'] = df['title'].astype(str).str.replace('\n', ' ', regex=False).str.replace('\r', ' ',
                                                                                          regex=False).str.replace('\t',
                                                                                                                   ' ',
                                                                                                                   regex=False)
    df['description'] = df['description'].astype(str).str.replace('\n', ' ', regex=False).str.replace('\r', ' ',
                                                                                                      regex=False).str.replace(
        '\t', ' ', regex=False)

    # 统计
    matched_title_count = (df['title'] != '').sum()
    matched_desc_count = (df['description'] != '').sum()
    print(f"  asin -> title 匹配: {matched_title_count}/{total_count} ({matched_title_count / total_count * 100:.2f}%)")
    print(
        f"  asin -> description 匹配: {matched_desc_count}/{total_count} ({matched_desc_count / total_count * 100:.2f}%)")

    # ==================== 步骤4: 保存结果 ====================
    print("\n" + "=" * 80)
    print("步骤4: 保存结果")
    print("=" * 80)

    # 使用逗号分隔保存，并确保字段正确分列
    # 使用quoting参数确保包含特殊字符的字段被正确引用
    df.to_csv(
        output_file,
        index=False,
        sep=',',  # 使用逗号分隔
        quoting=1,  # csv.QUOTE_ALL - 所有字段都加引号，确保正确分列
        encoding='utf-8',
        escapechar='\\'  # 转义字符
    )

    print(f"✓ 结果已保存到: {output_file}")
    print(f"  格式: CSV (逗号分隔，所有字段加引号)")
    print(f"  行数: {len(df)}")
    print(f"  列数: {len(df.columns)}")
    print(f"  列名: {list(df.columns)}")

    # ==================== 步骤5: 验证保存的文件 ====================
    print("\n" + "=" * 80)
    print("步骤5: 验证保存的文件")
    print("=" * 80)

    # 重新读取保存的文件进行验证
    try:
        df_verify = pd.read_csv(output_file, sep=',')
        print(f"✓ 文件验证成功！")
        print(f"  读取行数: {len(df_verify)}")
        print(f"  读取列数: {len(df_verify.columns)}")
        print(f"  列名: {list(df_verify.columns)}")

        # 检查是否有错位
        if len(df_verify.columns) == len(df.columns):
            print("✓ 列数正确，数据未错位")
        else:
            print(f"⚠ 警告: 列数不匹配 (期望{len(df.columns)}，实际{len(df_verify.columns)})")

        # 展示前几行
        print("\n前5行数据预览:")
        print(df_verify.head(5).to_string())

    except Exception as e:
        print(f"✗ 验证失败: {e}")

    # ==================== 步骤6: 结果展示 ====================
    print("\n" + "=" * 80)
    print("步骤6: 结果统计")
    print("=" * 80)

    # 展示前10行（原始df）
    print("\n前10行数据预览:")
    display_cols = ['user_id', 'item_id', 'asin', 'title', 'is_positive', 'pctr_score']
    available_cols = [col for col in display_cols if col in df.columns]
    pd.set_option('display.max_colwidth', 50)  # 限制列宽
    pd.set_option('display.width', 200)  # 设置总宽度
    print(df[available_cols].head(10).to_string())

    # 展示一个正样本的完整信息
    print("\n正样本示例 (is_positive=1):")
    positive_samples = df[df['is_positive'] == 1]
    if len(positive_samples) > 0:
        sample = positive_samples.iloc[0]
        print(f"  user_id: {sample['user_id']}")
        print(f"  item_id: {sample['item_id']}")
        print(f"  asin: {sample['asin']}")
        print(f"  title: {sample['title']}")
        desc = str(sample['description'])
        if len(desc) > 200:
            desc = desc[:200] + "..."
        print(f"  description: {desc}")
        print(f"  pctr_score: {sample['pctr_score']}")

    # 展示一个负样本的完整信息
    print("\n负样本示例 (is_positive=0):")
    negative_samples = df[df['is_positive'] == 0]
    if len(negative_samples) > 0:
        sample = negative_samples.iloc[0]
        print(f"  user_id: {sample['user_id']}")
        print(f"  item_id: {sample['item_id']}")
        print(f"  asin: {sample['asin']}")
        print(f"  title: {sample['title']}")
        desc = str(sample['description'])
        if len(desc) > 200:
            desc = desc[:200] + "..."
        print(f"  description: {desc}")
        print(f"  pctr_score: {sample['pctr_score']}")

    # 统计信息
    print("\n整体统计:")
    print(f"  总行数: {len(df)}")
    print(f"  唯一用户数: {df['user_id'].nunique()}")
    print(f"  唯一item数: {df['item_id'].nunique()}")
    print(f"  正样本数: {(df['is_positive'] == 1).sum()}")
    print(f"  负样本数: {(df['is_positive'] == 0).sum()}")
    print(f"  完整匹配(有asin+title)的行数: {((df['asin'].notna()) & (df['title'] != '')).sum()}")

    # 未匹配分析
    no_asin = df['asin'].isna().sum()
    no_title = ((df['asin'].notna()) & (df['title'] == '')).sum()

    if no_asin > 0:
        print(f"\n  未匹配到asin的item_id (共{no_asin}行):")
        no_asin_items = df[df['asin'].isna()]['item_id'].value_counts().head(10)
        for item_id, count in no_asin_items.items():
            print(f"    item_id={item_id}: 出现{count}次")

    if no_title > 0:
        print(f"\n  有asin但无title的asin (共{no_title}行):")
        no_title_asins = df[(df['asin'].notna()) & (df['title'] == '')]['asin'].value_counts().head(10)
        for asin, count in no_title_asins.items():
            print(f"    asin={asin}: 出现{count}次")

    print("\n" + "=" * 80)
    print("全部完成！")
    print("=" * 80)
    print(f"\n输出文件: {output_file}")
    print("文件格式: CSV (逗号分隔，所有字段用引号包围)")
    print("可以用Excel、文本编辑器或pandas正常打开")

    return df


if __name__ == "__main__":
    df = main()