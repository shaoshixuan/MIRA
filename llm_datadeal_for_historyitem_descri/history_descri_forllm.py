# # llm_data_preparation.py
# """
# LLM增强前置数据处理模块
# 功能：
# 1. 从kg_train_triples.txt获取用户历史交互商品
# 2. 通过kg_items.txt映射得到商品asin
# 3. 从meta_Toys_and_Games.jsonl获取商品详细信息
# 4. 为每个需要LLM增强的用户准备历史交互数据
# 5. 输出格式与KG输出保持一致
# """
#
# import pandas as pd
# import numpy as np
# import json
# import os
# from collections import defaultdict
# from tqdm import tqdm
#
#
# class LLMDataPreparator:
#     """
#     LLM增强数据准备器
#     输出格式与KGInterestScorer保持一致
#     """
#
#     def __init__(self,
#                  kg_train_file='D:/graduation_recsys/llm_datadeal_for_historyitem_descri/kg_train_triples.txt',
#                  kg_items_file='D:/graduation_recsys/llm_datadeal_for_historyitem_descri/kg_items.txt',
#                  meta_jsonl_file='D:/graduation_recsys/llm_datadeal_for_historyitem_descri/meta_Toys_and_Games.jsonl',
#                  kg_output_dir='D:/graduation_recsys/data/KG-related_Files/kg_interest_scores',
#                  output_dir='D:/graduation_recsys/llm_datadeal_for_historyitem_descri/data_for_llm_rec'):
#         """
#         初始化数据准备器
#         """
#         self.kg_train_file = kg_train_file
#         self.kg_items_file = kg_items_file
#         self.meta_jsonl_file = meta_jsonl_file
#         self.kg_output_dir = kg_output_dir
#         self.output_dir = output_dir
#
#         # 数据存储
#         self.item_id_to_asin = {}  # item_id -> asin
#         self.asin_to_metadata = {}  # asin -> {title, description, ...}
#         self.user_history_items = defaultdict(list)  # user_id -> [item_id, ...]
#         self.user_history_details = {}  # user_id -> [{item_id, asin, title, description}, ...]
#
#         # 统计信息
#         self.stats = {
#             'total_users': 0,
#             'total_items_mapped': 0,
#             'total_metadata_found': 0,
#             'users_with_history': 0,
#             'low_interaction_users': 0,
#             'enhancement_users': 0
#         }
#
#     def load_all_data(self):
#         """加载所有必要数据"""
#         print("=" * 60)
#         print("加载LLM增强所需数据")
#         print("=" * 60)
#
#         self._check_files()
#
#         print("\n1. 加载商品ID到ASIN映射...")
#         self._load_item_asin_mapping()
#
#         print("\n2. 构建ASIN元数据索引...")
#         self._build_asin_metadata_index()
#
#         print("\n3. 加载用户历史交互数据...")
#         self._load_user_history()
#
#         print("\n4. 匹配用户历史商品详细信息...")
#         self._match_user_history_details()
#
#         print("\n✅ LLM增强数据准备完成！")
#         self._print_stats()
#
#     def _check_files(self):
#         """检查文件是否存在"""
#         files_to_check = {
#             'kg_items.txt': self.kg_items_file,
#             'kg_train_triples.txt': self.kg_train_file,
#             'meta_Toys_and_Games.jsonl': self.meta_jsonl_file
#         }
#
#         print("检查文件...")
#         for name, path in files_to_check.items():
#             if os.path.exists(path):
#                 file_size = os.path.getsize(path) / (1024 * 1024)
#                 print(f"  ✅ {name}: {path} ({file_size:.2f} MB)")
#             else:
#                 print(f"  ❌ {name}: {path} (文件不存在!)")
#                 raise FileNotFoundError(f"文件不存在: {path}")
#
#         os.makedirs(self.output_dir, exist_ok=True)
#         print(f"  输出目录: {self.output_dir}")
#
#     def _load_item_asin_mapping(self):
#         """加载kg_items文件"""
#         print(f"  读取: {self.kg_items_file}")
#
#         line_count = 0
#         with open(self.kg_items_file, 'r', encoding='utf-8') as f:
#             for line in tqdm(f, desc="  加载item-asin映射"):
#                 line = line.strip()
#                 if not line:
#                     continue
#                 line_count += 1
#
#                 parts = line.split('\t')
#                 if len(parts) == 2:
#                     product_part = parts[0]
#                     item_id_str = parts[1]
#
#                     if '::' in product_part:
#                         asin = product_part.split('::')[1]
#                     else:
#                         asin = product_part
#
#                     try:
#                         item_id = str(int(float(item_id_str.strip())))
#                     except:
#                         item_id = item_id_str.strip()
#
#                     self.item_id_to_asin[item_id] = asin
#
#         print(f"  加载了 {len(self.item_id_to_asin)} 个映射")
#
#     def _build_asin_metadata_index(self):
#         """构建asin元数据索引"""
#         print(f"  读取: {self.meta_jsonl_file}")
#
#         line_count = 0
#         error_count = 0
#
#         with open(self.meta_jsonl_file, 'r', encoding='utf-8') as f:
#             for line in tqdm(f, desc="  构建元数据索引"):
#                 line = line.strip()
#                 if not line:
#                     continue
#                 line_count += 1
#
#                 try:
#                     data = json.loads(line)
#                     asin = data.get('parent_asin', None)
#
#                     if asin:
#                         title = data.get('title', '')
#                         description = data.get('description', [])
#
#                         if isinstance(description, list):
#                             description = ' '.join([str(d) for d in description])
#                         elif not isinstance(description, str):
#                             description = str(description)
#
#                         title = title.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
#                         description = description.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
#
#                         categories = data.get('categories', [])
#                         if isinstance(categories, str):
#                             categories = [categories]
#                         elif not isinstance(categories, list):
#                             categories = []
#
#                         features = data.get('features', [])
#                         if isinstance(features, str):
#                             features = [features]
#                         elif not isinstance(features, list):
#                             features = []
#
#                         self.asin_to_metadata[asin] = {
#                             'title': title,
#                             'description': description,
#                             'main_category': str(data.get('main_category', '')),
#                             'brand': str(data.get('brand', '')),
#                             'categories': categories,
#                             'average_rating': data.get('average_rating', None),
#                             'rating_number': data.get('rating_number', None),
#                             'price': data.get('price', None),
#                             'features': features,
#                             'details': data.get('details', {}),
#                             'store': str(data.get('store', '')),
#                             'parent_asin': asin
#                         }
#                 except json.JSONDecodeError:
#                     error_count += 1
#                 except Exception:
#                     error_count += 1
#
#         print(f"  处理了 {line_count} 行，索引了 {len(self.asin_to_metadata)} 个商品")
#
#     def _load_user_history(self):
#         """加载用户历史交互数据"""
#         print(f"  读取: {self.kg_train_file}")
#
#         try:
#             train_triples = pd.read_csv(self.kg_train_file, sep='\t', header=None,
#                                         names=['head', 'tail', 'relation'])
#             user_col, item_col = 'head', 'tail'
#         except:
#             train_triples = pd.read_csv(self.kg_train_file)
#             user_col = train_triples.columns[0]
#             item_col = train_triples.columns[1]
#
#         print(f"  训练数据形状: {train_triples.shape}")
#
#         def format_id(x):
#             if pd.isna(x):
#                 return ""
#             x_str = str(x).strip()
#             if '.' in x_str:
#                 try:
#                     x_float = float(x_str)
#                     if abs(x_float - round(x_float)) < 0.001:
#                         return str(int(round(x_float)))
#                 except:
#                     pass
#             return x_str
#
#         train_triples[user_col] = train_triples[user_col].apply(format_id)
#         train_triples[item_col] = train_triples[item_col].apply(format_id)
#
#         for _, row in tqdm(train_triples.iterrows(), total=len(train_triples), desc="  分组用户历史"):
#             user_id = row[user_col]
#             item_id = row[item_col]
#             if user_id and item_id and user_id != 'nan' and item_id != 'nan':
#                 self.user_history_items[user_id].append(item_id)
#
#         for user_id in self.user_history_items:
#             self.user_history_items[user_id] = list(set(self.user_history_items[user_id]))
#
#         print(f"  加载了 {len(self.user_history_items)} 个用户的历史交互")
#
#     def _match_user_history_details(self):
#         """为用户历史商品匹配详细信息"""
#         print("  匹配用户历史商品的详细信息...")
#
#         matched_users = 0
#         total_items_matched = 0
#         total_items_not_matched = 0
#
#         for user_id, item_ids in tqdm(self.user_history_items.items(), desc="  匹配历史详情"):
#             user_details = []
#
#             for item_id in item_ids:
#                 asin = self.item_id_to_asin.get(item_id, None)
#                 if asin is None:
#                     total_items_not_matched += 1
#                     continue
#
#                 metadata = self.asin_to_metadata.get(asin, None)
#                 if metadata is None:
#                     total_items_not_matched += 1
#                     continue
#
#                 user_details.append({
#                     'item_id': item_id,
#                     'asin': asin,
#                     'title': metadata['title'],
#                     'description': metadata['description'],
#                     'brand': metadata.get('brand', ''),
#                     'categories': metadata.get('categories', []),
#                     'average_rating': metadata.get('average_rating', None),
#                     'price': metadata.get('price', None),
#                 })
#                 total_items_matched += 1
#
#             if user_details:
#                 self.user_history_details[user_id] = user_details
#                 matched_users += 1
#
#         self.stats['total_users'] = len(self.user_history_items)
#         self.stats['total_items_mapped'] = total_items_matched
#         self.stats['total_metadata_found'] = total_items_matched
#         self.stats['users_with_history'] = matched_users
#
#         print(f"  匹配结果: 用户={matched_users}, 商品={total_items_matched}")
#
#     def prepare_llm_input_data(self):
#         """
#         准备LLM输入数据
#         输出格式与KG输出保持一致：user_id, item_id, asin, title, description, is_positive,
#                                original_pctr_score, kg_interest_score, final_combined_score
#
#         Returns:
#             pd.DataFrame: 需要LLM处理的完整数据（结构与KG输出一致）
#             dict: 用户历史交互数据（用于构建LLM prompt）
#         """
#         print("\n" + "=" * 60)
#         print("准备LLM输入数据")
#         print("=" * 60)
#
#         # 1. 读取需要LLM处理的用户列表
#         llm_users_file = os.path.join(self.kg_output_dir, 'users_needing_llm.txt')
#
#         if not os.path.exists(llm_users_file):
#             print(f"  ⚠️ 未找到: {llm_users_file}")
#             return None, None
#
#         users_needing_llm = {}
#         with open(llm_users_file, 'r') as f:
#             for line in f:
#                 parts = line.strip().split('\t')
#                 if len(parts) >= 3:
#                     user_id = parts[0]
#                     interaction_count = int(parts[1])
#                     user_type = parts[2]
#                     users_needing_llm[user_id] = {
#                         'interaction_count': interaction_count,
#                         'user_type': user_type
#                     }
#
#         low_interaction = {k: v for k, v in users_needing_llm.items() if v['user_type'] == 'LOW_INTERACTION'}
#         enhancement = {k: v for k, v in users_needing_llm.items() if v['user_type'] == 'NEEDS_ENHANCEMENT'}
#
#         print(f"  低交互用户(<5次): {len(low_interaction)}")
#         print(f"  需要增强用户(≥5次): {len(enhancement)}")
#         print(f"  总计: {len(users_needing_llm)}")
#
#         self.stats['low_interaction_users'] = len(low_interaction)
#         self.stats['enhancement_users'] = len(enhancement)
#
#         # 2. 读取KG输出CSV，筛选需要LLM处理的记录
#         kg_csv_file = os.path.join(self.kg_output_dir, 'reranked_predictions_with_kg.csv')
#
#         if not os.path.exists(kg_csv_file):
#             print(f"  ⚠️ 未找到: {kg_csv_file}")
#             return None, None
#
#         # 读取CSV
#         encodings = ['utf-8-sig', 'utf-8', 'gb18030', 'gbk']
#         df = None
#         for enc in encodings:
#             try:
#                 df = pd.read_csv(kg_csv_file, encoding=enc)
#                 print(f"  使用 {enc} 编码读取CSV成功")
#                 break
#             except:
#                 continue
#
#         if df is None:
#             print(f"  ❌ 无法读取CSV文件")
#             return None, None
#
#         print(f"  CSV数据: {df.shape[0]} 行, {df.shape[1]} 列")
#
#         # 3. 筛选需要LLM处理的记录
#         # kg_interest_score == -10 或 kg_interest_score == -5
#         llm_mask = (df['kg_interest_score'] == -10) | (df['kg_interest_score'] == -5)
#         llm_df = df[llm_mask].copy()
#
#         print(f"  需要LLM处理的记录: {len(llm_df)}")
#         print(f"    占位符(-10): {sum(df['kg_interest_score'] == -10)}")
#         print(f"    增强标记(-5): {sum(df['kg_interest_score'] == -5)}")
#
#         # 4. 按用户分组，准备LLM prompt所需的历史数据
#         llm_history_data = {}
#
#         for user_id in users_needing_llm.keys():
#             # 获取用户历史交互详情
#             history = self.user_history_details.get(user_id, [])
#
#             # 构建历史商品摘要
#             history_items = []
#             for item in history:
#                 history_items.append({
#                     'item_id': item['item_id'],
#                     'asin': item['asin'],
#                     'title': item['title'],
#                     'description': item['description'],
#                     'brand': item.get('brand', ''),
#                     'categories': item.get('categories', [])
#                 })
#
#             # 统计偏好
#             all_categories = []
#             all_brands = []
#             for item in history:
#                 if item.get('brand'):
#                     all_brands.append(item['brand'])
#                 cats = item.get('categories', [])
#                 if isinstance(cats, list):
#                     all_categories.extend(cats)
#
#             from collections import Counter
#             top_categories = [cat for cat, _ in Counter(all_categories).most_common(10)]
#             top_brands = [brand for brand, _ in Counter(all_brands).most_common(5)]
#
#             llm_history_data[user_id] = {
#                 'interaction_count': len(history),
#                 'user_type': users_needing_llm[user_id]['user_type'],
#                 'history_items': history_items,
#                 'history_titles': [item['title'] for item in history],
#                 'history_descriptions': [item['description'] for item in history],
#                 'preferred_categories': top_categories,
#                 'preferred_brands': top_brands
#             }
#
#         # 5. 为llm_df中的每条记录补充asin（如果为空的话）
#         # 通过item_id查找asin
#         missing_asin_mask = llm_df['asin'].isna() | (llm_df['asin'] == '')
#         if missing_asin_mask.any():
#             print(f"  补充缺失的asin: {missing_asin_mask.sum()} 条")
#             for idx in llm_df[missing_asin_mask].index:
#                 item_id = str(llm_df.loc[idx, 'item_id'])
#                 asin = self.item_id_to_asin.get(item_id, '')
#                 if asin:
#                     llm_df.loc[idx, 'asin'] = asin
#
#                     # 同时补充title和description
#                     metadata = self.asin_to_metadata.get(asin, {})
#                     if metadata:
#                         if not llm_df.loc[idx, 'title'] or pd.isna(llm_df.loc[idx, 'title']):
#                             llm_df.loc[idx, 'title'] = metadata.get('title', '')
#                         if not llm_df.loc[idx, 'description'] or pd.isna(llm_df.loc[idx, 'description']):
#                             llm_df.loc[idx, 'description'] = metadata.get('description', '')
#
#         return llm_df, llm_history_data
#
#     def save_llm_input_data(self):
#         """
#         保存LLM输入数据
#         输出：
#         1. llm_input_data.csv - 需要LLM处理的候选商品（格式与KG输出一致）
#         2. llm_user_history.json - 用户历史交互数据（用于构建prompt）
#         """
#         llm_df, llm_history_data = self.prepare_llm_input_data()
#
#         if llm_df is None:
#             print("❌ 数据准备失败")
#             return None, None
#
#         # 保存CSV（与KG输出格式完全一致）
#         csv_output = os.path.join(self.output_dir, 'llm_input_data.csv')
#         # 保持与KG输出相同的列顺序
#         output_columns = ['user_id', 'item_id', 'asin', 'title', 'description',
#                           'is_positive', 'original_pctr_score', 'kg_interest_score',
#                           'final_combined_score']
#         available_columns = [col for col in output_columns if col in llm_df.columns]
#         llm_df[available_columns].to_csv(csv_output, index=False, encoding='utf-8-sig')
#
#         print(f"\n✅ LLM输入数据(CSV)已保存: {csv_output}")
#         print(f"   记录数: {len(llm_df)}")
#         print(f"   列: {available_columns}")
#
#         # 按用户类型统计
#         low_interaction_records = len(llm_df[llm_df['kg_interest_score'] == -10])
#         enhancement_records = len(llm_df[llm_df['kg_interest_score'] == -5])
#         print(f"   低交互用户记录(-10): {low_interaction_records}")
#         print(f"   增强用户记录(-5): {enhancement_records}")
#
#         # 保存用户历史数据（JSON）
#         json_output = os.path.join(self.output_dir, 'llm_user_history.json')
#
#         def convert_to_native(obj):
#             if isinstance(obj, (np.integer,)):
#                 return int(obj)
#             elif isinstance(obj, (np.floating,)):
#                 return float(obj)
#             elif isinstance(obj, np.ndarray):
#                 return obj.tolist()
#             elif isinstance(obj, dict):
#                 return {str(k): convert_to_native(v) for k, v in obj.items()}
#             elif isinstance(obj, list):
#                 return [convert_to_native(item) for item in obj]
#             return obj
#
#         llm_history_data = convert_to_native(llm_history_data)
#
#         with open(json_output, 'w', encoding='utf-8') as f:
#             json.dump(llm_history_data, f, ensure_ascii=False, indent=2)
#
#         file_size = os.path.getsize(json_output) / (1024 * 1024)
#         print(f"\n✅ 用户历史数据(JSON)已保存: {json_output}")
#         print(f"   文件大小: {file_size:.2f} MB")
#         print(f"   用户数: {len(llm_history_data)}")
#
#         # 保存按用户分组的候选商品（便于批量LLM处理）
#         user_candidates = defaultdict(list)
#         for _, row in llm_df.iterrows():
#             user_id = str(row['user_id'])
#             user_candidates[user_id].append({
#                 'item_id': str(row['item_id']),
#                 'asin': str(row.get('asin', '')),
#                 'title': str(row.get('title', '')),
#                 'description': str(row.get('description', '')),
#                 'is_positive': int(row.get('is_positive', 0)),
#                 'original_pctr_score': float(row.get('original_pctr_score', 0)),
#                 'kg_interest_score': float(row.get('kg_interest_score', 0)),
#                 'final_combined_score': float(row.get('final_combined_score', 0))
#             })
#
#         # 合并用户历史数据和候选商品
#         user_full_data = {}
#         for user_id in user_candidates:
#             user_full_data[user_id] = {
#                 'history': llm_history_data.get(user_id, {}),
#                 'candidates': user_candidates[user_id]
#             }
#
#         user_full_data = convert_to_native(user_full_data)
#
#         full_json_output = os.path.join(self.output_dir, 'llm_user_full_data.json')
#         with open(full_json_output, 'w', encoding='utf-8') as f:
#             json.dump(user_full_data, f, ensure_ascii=False, indent=2)
#
#         file_size = os.path.getsize(full_json_output) / (1024 * 1024)
#         print(f"\n✅ 用户完整数据(JSON)已保存: {full_json_output}")
#         print(f"   文件大小: {file_size:.2f} MB")
#         print(f"   用户数: {len(user_full_data)}")
#
#         return csv_output, json_output
#
#     def get_user_prompt_data(self, user_id):
#         """
#         获取单个用户的LLM prompt数据
#
#         Returns:
#             dict: {
#                 'history_text': 历史商品文本,
#                 'candidates': 候选商品列表,
#                 'preference': 用户偏好
#             }
#         """
#         user_id = str(user_id)
#         history = self.user_history_details.get(user_id, [])
#
#         # 构建历史文本
#         history_text_parts = []
#         for i, item in enumerate(history[:20]):  # 最多20个历史商品
#             parts = [f"商品{i + 1}: {item['title']}"]
#             if item.get('description'):
#                 parts.append(f"  描述: {item['description'][:200]}")
#             if item.get('brand'):
#                 parts.append(f"  品牌: {item['brand']}")
#             if item.get('categories'):
#                 cats = item['categories'][:3] if isinstance(item['categories'], list) else []
#                 if cats:
#                     parts.append(f"  类别: {', '.join(cats)}")
#             history_text_parts.append('\n'.join(parts))
#
#         history_text = '\n\n'.join(history_text_parts)
#
#         # 统计偏好
#         all_categories = []
#         all_brands = []
#         for item in history:
#             if item.get('brand'):
#                 all_brands.append(item['brand'])
#             cats = item.get('categories', [])
#             if isinstance(cats, list):
#                 all_categories.extend(cats)
#
#         from collections import Counter
#         preference = {
#             'interaction_count': len(history),
#             'top_categories': [cat for cat, _ in Counter(all_categories).most_common(5)],
#             'top_brands': [brand for brand, _ in Counter(all_brands).most_common(3)]
#         }
#
#         return {
#             'history_text': history_text,
#             'preference': preference
#         }
#
#     def _print_stats(self):
#         """打印统计信息"""
#         print("\n" + "=" * 60)
#         print("📊 数据准备统计")
#         print("=" * 60)
#         print(f"商品ID-ASIN映射: {len(self.item_id_to_asin)}")
#         print(f"ASIN元数据索引: {len(self.asin_to_metadata)}")
#         print(f"用户历史交互: {len(self.user_history_items)}")
#         print(f"有详细信息的用户: {self.stats['users_with_history']}")
#         print(f"匹配到的历史商品: {self.stats['total_items_mapped']}")
#
#
# def main():
#     """主函数"""
#     # 文件路径
#     kg_train_file = 'D:/graduation_recsys/llm_datadeal_for_historyitem_descri/kg_train_triples.txt'
#     kg_items_file = 'D:/graduation_recsys/llm_datadeal_for_historyitem_descri/kg_items.txt'
#     meta_jsonl_file = 'D:/graduation_recsys/llm_datadeal_for_historyitem_descri/meta_Automotive.jsonl'
#     kg_output_dir = 'D:/graduation_recsys/data/KG-related_Files/kg_interest_scores'
#     output_dir = 'D:/graduation_recsys/llm_datadeal_for_historyitem_descri/data_for_llm_rec'
#
#     print("=" * 60)
#     print("LLM增强数据准备")
#     print("=" * 60)
#
#     preparator = LLMDataPreparator(
#         kg_train_file=kg_train_file,
#         kg_items_file=kg_items_file,
#         meta_jsonl_file=meta_jsonl_file,
#         kg_output_dir=kg_output_dir,
#         output_dir= output_dir
#     )
#
#     # 加载数据
#     preparator.load_all_data()
#
#     # 准备并保存LLM输入数据
#     csv_output, json_output = preparator.save_llm_input_data()
#
#     if csv_output:
#         print(f"\n{'=' * 60}")
#         print("输出文件说明:")
#         print(f"{'=' * 60}")
#         print(f"1. llm_input_data.csv - 需要LLM处理的候选商品（格式与KG输出一致）")
#         print(f"   列: user_id, item_id, asin, title, description, is_positive,")
#         print(f"        original_pctr_score, kg_interest_score, final_combined_score")
#         print(f"   kg_interest_score含义:")
#         print(f"     -10: 低交互用户，需要LLM完全重新打分")
#         print(f"     -5:  高交互用户，KG分数区分度不足，需要LLM增强")
#         print(f"")
#         print(f"2. llm_user_history.json - 用户历史交互数据")
#         print(f"3. llm_user_full_data.json - 用户完整数据（历史+候选）")
#
#     print("\n✅ 数据准备完成！")
#
#
# if __name__ == "__main__":
#     main()

import os
import json
import gc
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from tqdm import tqdm


class LLMDataPreparator:
    """
    LLM增强数据准备器（内存优化版）

    优化：
    1. metadata只加载KG涉及的商品
    2. metadata只保存LLM需要字段
    3. pandas apply改为向量化
    4. 主动释放临时变量
    """

    def __init__(
            self,
            kg_train_file,
            kg_items_file,
            meta_jsonl_file,
            kg_output_dir,
            output_dir
    ):

        self.kg_train_file = kg_train_file
        self.kg_items_file = kg_items_file
        self.meta_jsonl_file = meta_jsonl_file

        self.kg_output_dir = kg_output_dir
        self.output_dir = output_dir


        # item -> asin
        self.item_id_to_asin = {}


        # asin -> metadata
        # 只保存:
        # title
        # description
        # brand
        # categories
        self.asin_to_metadata = {}


        # user -> item list
        self.user_history_items = defaultdict(list)


        # user -> detail list
        self.user_history_details = {}


        self.stats = {
            "total_users":0,
            "total_items_mapped":0,
            "total_metadata_found":0
        }



    def load_all_data(self):

        print("="*60)
        print("加载LLM增强数据")
        print("="*60)


        self._check_files()


        print("\n1. 加载item-asin映射")
        self._load_item_asin_mapping()


        print("\n2. 加载商品metadata")
        self._build_asin_metadata_index()


        print("\n3. 加载用户历史")
        self._load_user_history()


        print("\n4. 匹配历史商品详情")
        self._match_user_history_details()


        gc.collect()


        print("\n数据加载完成")



    def _check_files(self):

        files = {
            "kg_items":
                self.kg_items_file,

            "kg_train":
                self.kg_train_file,

            "metadata":
                self.meta_jsonl_file
        }


        for name,path in files.items():

            if not os.path.exists(path):

                raise FileNotFoundError(path)


            size=os.path.getsize(path)/(1024*1024)

            print(
                f"{name}: {path} ({size:.2f}MB)"
            )


        os.makedirs(
            self.output_dir,
            exist_ok=True
        )



    def _load_item_asin_mapping(self):

        """
        加载kg_items

        生成:
            item_id -> asin

        同时记录所有需要加载metadata的asin
        """


        with open(
                self.kg_items_file,
                "r",
                encoding="utf-8"
        ) as f:


            for line in tqdm(
                    f,
                    desc="读取item映射"
            ):

                line=line.strip()

                if not line:
                    continue


                parts=line.split("\t")


                if len(parts)!=2:
                    continue


                product_part,item_id=parts


                if "::" in product_part:

                    asin=product_part.split("::")[1]

                else:

                    asin=product_part



                item_id=(
                    str(item_id)
                    .replace(".0","")
                    .strip()
                )


                self.item_id_to_asin[item_id]=asin



        print(
            "item数量:",
            len(self.item_id_to_asin)
        )



    def _build_asin_metadata_index(self):

        """
        核心优化函数

        原逻辑：
        加载200万商品metadata

        新逻辑：
        只加载KG涉及的商品
        """


        need_asins=set(
            self.item_id_to_asin.values()
        )


        print(
            "需要匹配ASIN:",
            len(need_asins)
        )


        total=0
        matched=0


        with open(
                self.meta_jsonl_file,
                "r",
                encoding="utf-8"
        ) as f:


            for line in tqdm(
                    f,
                    desc="扫描metadata"
            ):


                if not line.strip():
                    continue


                total+=1


                try:

                    data=json.loads(line)


                    asin=data.get(
                        "parent_asin",
                        ""
                    )


                    # 非KG商品直接跳过
                    if asin not in need_asins:
                        continue



                    title=data.get(
                        "title",
                        ""
                    )


                    description=data.get(
                        "description",
                        ""
                    )


                    if isinstance(
                            description,
                            list
                    ):

                        description=" ".join(
                            map(
                                str,
                                description
                            )
                        )


                    categories=data.get(
                        "categories",
                        []
                    )


                    if not isinstance(
                            categories,
                            list
                    ):

                        categories=[]



                    self.asin_to_metadata[asin]={

                        "title":
                            str(title)
                            .replace("\n"," ")
                            .replace("\r"," "),


                        "description":
                            str(description)
                            .replace("\n"," ")
                            .replace("\r"," "),


                        "brand":
                            str(
                                data.get(
                                    "brand",
                                    ""
                                )
                            ),


                        "categories":
                            categories
                    }



                    matched+=1


                except Exception:

                    continue



        print(
            f"metadata扫描:{total}"
        )

        print(
            f"实际加载:{matched}"
        )


        # 释放need_asins
        del need_asins

        gc.collect()



    def _load_user_history(self):

        """
        加载用户历史交互

        优化：
        不使用apply
        """



        try:

            train=pd.read_csv(
                self.kg_train_file,
                sep="\t",
                header=None,
                names=[
                    "head",
                    "tail",
                    "relation"
                ]
            )


            user_col="head"
            item_col="tail"



        except:


            train=pd.read_csv(
                self.kg_train_file
            )


            user_col=train.columns[0]
            item_col=train.columns[1]



        print(
            "训练数据:",
            train.shape
        )



        # 向量化替代apply
        train[user_col]=(
            train[user_col]
            .astype(str)
            .str.replace(
                r"\.0$",
                "",
                regex=True
            )
            .str.strip()
        )


        train[item_col]=(
            train[item_col]
            .astype(str)
            .str.replace(
                r"\.0$",
                "",
                regex=True
            )
            .str.strip()
        )



        for row in tqdm(
                train[
                    [
                        user_col,
                        item_col
                    ]
                ].itertuples(
                    index=False,
                    name=None
                ),
                desc="构建用户历史",
                total=len(train)
        ):


            user,item=row


            if user and item:

                self.user_history_items[user].append(
                    item
                )



        for user in self.user_history_items:

            self.user_history_items[user]=list(
                set(
                    self.user_history_items[user]
                )
            )



        print(
            "用户数量:",
            len(self.user_history_items)
        )


        del train

        gc.collect()



    def _match_user_history_details(self):

        """
        根据item->asin->metadata
        匹配历史商品详情
        """


        matched=0


        for user,items in tqdm(
                self.user_history_items.items(),
                desc="匹配历史"
        ):


            details=[]


            for item in items:


                asin=self.item_id_to_asin.get(
                    item
                )


                if not asin:
                    continue



                meta=self.asin_to_metadata.get(
                    asin
                )


                if not meta:
                    continue



                details.append({

                    "item_id":item,

                    "asin":asin,

                    "title":
                        meta["title"],

                    "description":
                        meta["description"],

                    "brand":
                        meta["brand"],

                    "categories":
                        meta["categories"]

                })



            if details:

                self.user_history_details[user]=details

                matched+=1



        self.stats["total_users"]=len(
            self.user_history_items
        )

        self.stats["users_with_history"]=matched


        print(
            "匹配成功用户:",
            matched
        )

    def prepare_llm_input_data(self):
        """
        准备需要LLM处理的候选商品和用户历史数据。

        Returns:
            llm_df: 需要LLM处理的候选商品
            llm_history_data: 用户历史商品及偏好信息
        """

        print("\n" + "=" * 60)
        print("准备LLM输入数据")
        print("=" * 60)

        users_file = os.path.join(
            self.kg_output_dir,
            "users_needing_llm.txt"
        )

        if not os.path.exists(users_file):
            print(f"未找到文件: {users_file}")
            return None, None

        users_needing_llm = {}

        with open(users_file, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("\t")

                if len(parts) < 3:
                    continue

                user_id = str(parts[0]).strip()

                try:
                    interaction_count = int(parts[1])
                except (TypeError, ValueError):
                    interaction_count = 0

                user_type = str(parts[2]).strip()

                users_needing_llm[user_id] = {
                    "interaction_count": interaction_count,
                    "user_type": user_type
                }

        low_interaction_count = sum(
            info["user_type"] == "LOW_INTERACTION"
            for info in users_needing_llm.values()
        )

        enhancement_count = sum(
            info["user_type"] == "NEEDS_ENHANCEMENT"
            for info in users_needing_llm.values()
        )

        self.stats["low_interaction_users"] = low_interaction_count
        self.stats["enhancement_users"] = enhancement_count

        print(f"低交互用户: {low_interaction_count}")
        print(f"需要增强用户: {enhancement_count}")
        print(f"需要LLM处理的用户总数: {len(users_needing_llm)}")

        kg_csv_file = os.path.join(
            self.kg_output_dir,
            "reranked_predictions_with_kg.csv"
        )

        if not os.path.exists(kg_csv_file):
            print(f"未找到文件: {kg_csv_file}")
            return None, None

        encodings = [
            "utf-8-sig",
            "utf-8",
            "gb18030",
            "gbk"
        ]

        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(
                    kg_csv_file,
                    encoding=encoding,
                    low_memory=False
                )

                print(f"使用 {encoding} 编码读取CSV成功")
                break

            except (UnicodeDecodeError, LookupError):
                continue

        if df is None:
            print("无法读取KG输出CSV")
            return None, None

        required_columns = [
            "user_id",
            "item_id",
            "kg_interest_score"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"KG输出CSV缺少必要列: {missing_columns}"
            )

        df["user_id"] = (
            df["user_id"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )

        df["item_id"] = (
            df["item_id"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )

        df["kg_interest_score"] = pd.to_numeric(
            df["kg_interest_score"],
            errors="coerce"
        )

        llm_mask = df["kg_interest_score"].isin([-10, -5])
        llm_df = df.loc[llm_mask].copy()

        # 只保留明确需要LLM处理的用户
        llm_df = llm_df[
            llm_df["user_id"].isin(users_needing_llm)
        ].copy()

        print(f"KG输出总记录数: {len(df)}")
        print(f"需要LLM处理的记录数: {len(llm_df)}")
        print(
            f"低交互记录(-10): "
            f"{int((llm_df['kg_interest_score'] == -10).sum())}"
        )
        print(
            f"增强记录(-5): "
            f"{int((llm_df['kg_interest_score'] == -5).sum())}"
        )

        del df
        gc.collect()

        # 确保候选商品信息列存在
        for column in [
            "asin",
            "title",
            "description"
        ]:
            if column not in llm_df.columns:
                llm_df[column] = ""
            else:
                llm_df[column] = llm_df[column].fillna("").astype(str)

        # 根据item_id补充asin和metadata
        item_asins = llm_df["item_id"].map(
            self.item_id_to_asin
        ).fillna("")

        missing_asin = llm_df["asin"].eq("")
        llm_df.loc[missing_asin, "asin"] = item_asins[missing_asin]

        metadata_title = llm_df["asin"].map(
            lambda asin: self.asin_to_metadata.get(
                asin, {}
            ).get("title", "")
        )

        metadata_description = llm_df["asin"].map(
            lambda asin: self.asin_to_metadata.get(
                asin, {}
            ).get("description", "")
        )

        missing_title = llm_df["title"].eq("")
        missing_description = llm_df["description"].eq("")

        llm_df.loc[missing_title, "title"] = (
            metadata_title[missing_title]
        )

        llm_df.loc[missing_description, "description"] = (
            metadata_description[missing_description]
        )

        # 构建用户历史信息
        llm_history_data = {}

        for user_id, user_info in tqdm(
            users_needing_llm.items(),
            desc="构建用户历史数据"
        ):
            history = self.user_history_details.get(
                user_id,
                []
            )

            history_items = []
            all_categories = []
            all_brands = []

            for item in history:
                categories = item.get(
                    "categories",
                    []
                )

                if not isinstance(categories, list):
                    categories = []

                brand = str(
                    item.get("brand", "")
                ).strip()

                history_items.append({
                    "item_id": str(
                        item.get("item_id", "")
                    ),
                    "asin": str(
                        item.get("asin", "")
                    ),
                    "title": str(
                        item.get("title", "")
                    ),
                    "description": str(
                        item.get("description", "")
                    ),
                    "brand": brand,
                    "categories": categories
                })

                if brand:
                    all_brands.append(brand)

                all_categories.extend(
                    str(category)
                    for category in categories
                    if str(category).strip()
                )

            preferred_categories = [
                category
                for category, _ in Counter(
                    all_categories
                ).most_common(10)
            ]

            preferred_brands = [
                brand
                for brand, _ in Counter(
                    all_brands
                ).most_common(5)
            ]

            llm_history_data[user_id] = {
                "interaction_count": user_info[
                    "interaction_count"
                ],
                "matched_history_count": len(
                    history_items
                ),
                "user_type": user_info[
                    "user_type"
                ],
                "history_items": history_items,
                "history_titles": [
                    item["title"]
                    for item in history_items
                ],
                "history_descriptions": [
                    item["description"]
                    for item in history_items
                ],
                "preferred_categories": preferred_categories,
                "preferred_brands": preferred_brands
            }

        return llm_df, llm_history_data


    @staticmethod
    def _convert_to_native(obj):
        """将NumPy类型转换为JSON可序列化的Python类型。"""

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, dict):
            return {
                str(key): LLMDataPreparator._convert_to_native(value)
                for key, value in obj.items()
            }

        if isinstance(obj, list):
            return [
                LLMDataPreparator._convert_to_native(value)
                for value in obj
            ]

        if pd.isna(obj):
            return None

        return obj


    @staticmethod
    def _safe_float(value, default=0.0):
        """安全转换为float。"""

        try:
            if pd.isna(value):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default


    @staticmethod
    def _safe_int(value, default=0):
        """安全转换为int。"""

        try:
            if pd.isna(value):
                return default
            return int(value)
        except (TypeError, ValueError):
            return default


    def save_llm_input_data(self):
        """
        保存：
        1. llm_input_data.csv
        2. llm_user_history.json
        3. llm_user_full_data.json
        """

        llm_df, llm_history_data = (
            self.prepare_llm_input_data()
        )

        if llm_df is None:
            print("数据准备失败")
            return None, None

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        csv_output = os.path.join(
            self.output_dir,
            "llm_input_data.csv"
        )

        output_columns = [
            "user_id",
            "item_id",
            "asin",
            "title",
            "description",
            "is_positive",
            "original_pctr_score",
            "kg_interest_score",
            "final_combined_score"
        ]

        available_columns = [
            column
            for column in output_columns
            if column in llm_df.columns
        ]

        llm_df[available_columns].to_csv(
            csv_output,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"\nLLM候选数据已保存: {csv_output}")
        print(f"记录数: {len(llm_df)}")

        history_json_output = os.path.join(
            self.output_dir,
            "llm_user_history.json"
        )

        native_history_data = self._convert_to_native(
            llm_history_data
        )

        with open(
            history_json_output,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                native_history_data,
                file,
                ensure_ascii=False,
                indent=2
            )

        print(
            f"用户历史数据已保存: "
            f"{history_json_output}"
        )

        # 使用groupby代替逐行构建全部用户候选
        user_candidates = defaultdict(list)

        candidate_columns = [
            "user_id",
            "item_id",
            "asin",
            "title",
            "description",
            "is_positive",
            "original_pctr_score",
            "kg_interest_score",
            "final_combined_score"
        ]

        existing_candidate_columns = [
            column
            for column in candidate_columns
            if column in llm_df.columns
        ]

        for row in tqdm(
            llm_df[
                existing_candidate_columns
            ].itertuples(
                index=False,
                name=None
            ),
            total=len(llm_df),
            desc="构建候选商品数据"
        ):
            record = dict(
                zip(
                    existing_candidate_columns,
                    row
                )
            )

            user_id = str(
                record.get("user_id", "")
            )

            candidate = {
                "item_id": str(
                    record.get("item_id", "")
                ),
                "asin": str(
                    record.get("asin", "")
                ),
                "title": str(
                    record.get("title", "")
                ),
                "description": str(
                    record.get("description", "")
                ),
                "is_positive": self._safe_int(
                    record.get("is_positive", 0)
                ),
                "original_pctr_score": self._safe_float(
                    record.get(
                        "original_pctr_score",
                        0
                    )
                ),
                "kg_interest_score": self._safe_float(
                    record.get(
                        "kg_interest_score",
                        0
                    )
                ),
                "final_combined_score": self._safe_float(
                    record.get(
                        "final_combined_score",
                        0
                    )
                )
            }

            user_candidates[user_id].append(
                candidate
            )

        user_full_data = {}

        for user_id, candidates in user_candidates.items():
            user_full_data[user_id] = {
                "history": native_history_data.get(
                    user_id,
                    {}
                ),
                "candidates": candidates
            }

        full_json_output = os.path.join(
            self.output_dir,
            "llm_user_full_data.json"
        )

        with open(
            full_json_output,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                self._convert_to_native(
                    user_full_data
                ),
                file,
                ensure_ascii=False,
                indent=2
            )

        print(
            f"用户完整数据已保存: "
            f"{full_json_output}"
        )

        del user_candidates
        del user_full_data
        gc.collect()

        return csv_output, history_json_output


    def get_user_prompt_data(self, user_id):
        """获取单个用户的Prompt数据。"""

        user_id = str(user_id)
        history = self.user_history_details.get(
            user_id,
            []
        )

        history_text_parts = []

        for index, item in enumerate(history[:20]):
            parts = [
                f"商品{index + 1}: "
                f"{item.get('title', '')}"
            ]

            description = str(
                item.get("description", "")
            )

            if description:
                parts.append(
                    f"描述: {description[:200]}"
                )

            brand = str(
                item.get("brand", "")
            )

            if brand:
                parts.append(
                    f"品牌: {brand}"
                )

            categories = item.get(
                "categories",
                []
            )

            if isinstance(categories, list):
                categories = [
                    str(category)
                    for category in categories[:3]
                ]

                if categories:
                    parts.append(
                        f"类别: {', '.join(categories)}"
                    )

            history_text_parts.append(
                "\n".join(parts)
            )

        all_categories = []
        all_brands = []

        for item in history:
            brand = str(
                item.get("brand", "")
            ).strip()

            if brand:
                all_brands.append(brand)

            categories = item.get(
                "categories",
                []
            )

            if isinstance(categories, list):
                all_categories.extend(
                    str(category)
                    for category in categories
                    if str(category).strip()
                )

        preference = {
            "interaction_count": len(history),
            "top_categories": [
                category
                for category, _ in Counter(
                    all_categories
                ).most_common(5)
            ],
            "top_brands": [
                brand
                for brand, _ in Counter(
                    all_brands
                ).most_common(3)
            ]
        }

        return {
            "history_text": "\n\n".join(
                history_text_parts
            ),
            "preference": preference
        }


    def _print_stats(self):
        """打印统计信息。"""

        print("\n" + "=" * 60)
        print("数据准备统计")
        print("=" * 60)

        print(
            f"商品ID-ASIN映射: "
            f"{len(self.item_id_to_asin)}"
        )

        print(
            f"ASIN元数据索引: "
            f"{len(self.asin_to_metadata)}"
        )

        print(
            f"用户历史交互: "
            f"{len(self.user_history_items)}"
        )

        print(
            f"有详细信息的用户: "
            f"{len(self.user_history_details)}"
        )


def main():
    kg_train_file = (
        "D:/graduation_recsys/"
        "llm_datadeal_for_historyitem_descri/"
        "kg_train_triples.txt"
    )

    kg_items_file = (
        "D:/graduation_recsys/"
        "llm_datadeal_for_historyitem_descri/"
        "kg_items.txt"
    )

    meta_jsonl_file = (
        "D:/graduation_recsys/"
        "llm_datadeal_for_historyitem_descri/"
        "meta_Toys_and_Games.jsonl"
    )

    kg_output_dir = (
        "D:/graduation_recsys/data/"
        "KG-related_Files/kg_interest_scores"
    )

    output_dir = (
        "D:/graduation_recsys/"
        "llm_datadeal_for_historyitem_descri/"
        "data_for_llm_rec"
    )

    print("=" * 60)
    print("LLM增强数据准备")
    print("=" * 60)

    preparator = LLMDataPreparator(
        kg_train_file=kg_train_file,
        kg_items_file=kg_items_file,
        meta_jsonl_file=meta_jsonl_file,
        kg_output_dir=kg_output_dir,
        output_dir=output_dir
    )

    preparator.load_all_data()

    csv_output, json_output = (
        preparator.save_llm_input_data()
    )

    if csv_output:
        preparator._print_stats()

        print("\n" + "=" * 60)
        print("输出文件")
        print("=" * 60)
        print("1. llm_input_data.csv")
        print("2. llm_user_history.json")
        print("3. llm_user_full_data.json")

    print("\n数据准备完成")


if __name__ == "__main__":
    main()