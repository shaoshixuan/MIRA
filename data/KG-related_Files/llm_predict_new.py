# # llm_exploration_scoring.py
# """
# LLM探索分数计算系统
# 适配新的混合打分策略：
# 1. 低交互用户(kg_score=-10)：使用LLM进行完整的商品推荐（英文输出）
# 2. 需要增强的高交互用户(kg_score=-5)：使用LLM进行语义相似度增强（英文输出）
# 3. 输出格式与KG输出完全一致，便于后续合并和评估
# """
#
# import pandas as pd
# import numpy as np
# import os
# import warnings
# import json
# import requests
# from sentence_transformers import SentenceTransformer
# import scipy.spatial.distance as distance
# from datetime import datetime
# import time
# import re
# from typing import List, Dict, Tuple, Optional
# from collections import defaultdict
# from tqdm import tqdm
# from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
#
# warnings.filterwarnings('ignore')
#
#
# class LLMExplorationScorer:
#     """
#     LLM探索分数计算器
#     适配混合打分策略：
#     - 低交互用户(kg_score=-10)：LLM直接推荐商品，计算相似度分数
#     - 高交互增强用户(kg_score=-5)：LLM语义增强，提升KG分数区分度
#     """
#
#     def __init__(self,
#                  llm_input_csv: str = 'D:/graduation_recsys/llm_datadeal_for_historyitem_descri/data_for_llm_rec/llm_input_data.csv',
#                  user_history_json: str = 'D:/graduation_recsys/llm_datadeal_for_historyitem_descri/data_for_llm_rec/llm_user_history.json',
#                  llm_api_url: str = 'https://api.siliconflow.cn/v1/chat/completions',
#                  llm_api_key: str = os.getenv('SILICONFLOW_API_KEY', ''),
#                  embedding_model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
#                  output_dir: str = 'D:/graduation_recsys/data/KG-related_Files/llm_interest_scores'):
#         """
#         初始化LLM探索分数计算器
#
#         Args:
#             llm_input_csv: LLM输入CSV文件（从KG scorer输出）
#             user_history_json: 用户历史数据JSON文件
#             llm_api_url: LLM API地址
#             llm_api_key: LLM API密钥
#             embedding_model_name: 语义相似度模型
#             output_dir: 输出目录
#         """
#         self.llm_input_csv = llm_input_csv
#         self.user_history_json = user_history_json
#         self.llm_api_url = llm_api_url
#         self.llm_api_key = llm_api_key
#         self.output_dir = output_dir
#
#         # 加载数据
#         self.llm_input_data = None
#         self.user_history_data = {}
#         self.user_candidates = defaultdict(list)
#
#         # 加载嵌入模型（使用多语言模型以支持英文商品描述匹配）
#         print(f"加载嵌入模型: {embedding_model_name}")
#         self.embedding_model = SentenceTransformer(embedding_model_name)
#         print(f"模型维度: {self.embedding_model.get_sentence_embedding_dimension()}")
#
#         # 缓存
#         self.llm_recommendations_cache = {}  # {user_id: [推荐商品名列表]}
#         self.llm_embeddings_cache = {}  # {user_id: {商品名: 嵌入向量}}
#         self.description_embedding_cache = {}  # {description: 嵌入向量}
#
#         # 结果存储
#         self.exploration_scores = {}  # {(user_id, item_id): llm_exploration_score}
#         self.scoring_methods = {}  # {(user_id, item_id): method_description}
#
#         # 统计信息
#         self.stats = {
#             'total_records': 0,
#             'low_interaction_records': 0,
#             'enhancement_records': 0,
#             'llm_direct_scored': 0,
#             'llm_enhanced_scored': 0,
#             'processing_time': 0,
#             'api_calls': 0
#         }
#
#         os.makedirs(output_dir, exist_ok=True)
#
#     def load_data(self):
#         """加载LLM输入数据和用户历史数据"""
#         print("=" * 60)
#         print("加载LLM处理数据")
#         print("=" * 60)
#
#         # 1. 加载LLM输入CSV
#         print(f"\n1. 加载LLM输入数据: {self.llm_input_csv}")
#
#         if not os.path.exists(self.llm_input_csv):
#             raise FileNotFoundError(f"LLM输入文件不存在: {self.llm_input_csv}")
#
#         encodings = ['utf-8-sig', 'utf-8', 'gb18030', 'gbk']
#         for enc in encodings:
#             try:
#                 self.llm_input_data = pd.read_csv(self.llm_input_csv, encoding=enc)
#                 print(f"  ✅ 使用 {enc} 编码成功")
#                 break
#             except:
#                 continue
#
#         if self.llm_input_data is None:
#             raise ValueError("无法读取LLM输入CSV文件")
#
#         print(f"  数据形状: {self.llm_input_data.shape}")
#         print(f"  列名: {list(self.llm_input_data.columns)}")
#
#         # 统计
#         neg10_count = (self.llm_input_data['kg_interest_score'] == -10).sum()
#         neg5_count = (self.llm_input_data['kg_interest_score'] == -5).sum()
#         print(f"  低交互用户记录(-10): {neg10_count}")
#         print(f"  增强用户记录(-5): {neg5_count}")
#
#         self.stats['total_records'] = len(self.llm_input_data)
#         self.stats['low_interaction_records'] = neg10_count
#         self.stats['enhancement_records'] = neg5_count
#
#         # 按用户分组候选商品
#         print("\n2. 按用户分组候选商品...")
#         for _, row in self.llm_input_data.iterrows():
#             user_id = str(row['user_id'])
#             self.user_candidates[user_id].append({
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
#         print(f"  用户数: {len(self.user_candidates)}")
#
#         # 3. 加载用户历史数据
#         print(f"\n3. 加载用户历史数据: {self.user_history_json}")
#
#         if os.path.exists(self.user_history_json):
#             with open(self.user_history_json, 'r', encoding='utf-8') as f:
#                 self.user_history_data = json.load(f)
#             print(f"  加载了 {len(self.user_history_data)} 个用户的历史数据")
#         else:
#             print(f"  ⚠️ 用户历史文件不存在，将使用空历史")
#             self.user_history_data = {}
#
#         # 验证数据语言
#         print("\n4. 验证数据语言...")
#         sample_descriptions = []
#         for user_cands in list(self.user_candidates.values())[:5]:
#             for cand in user_cands[:3]:
#                 desc = cand.get('description', '')
#                 if desc and desc != 'nan':
#                     sample_descriptions.append(desc[:100])
#                     break
#
#         if sample_descriptions:
#             # 检测是否包含中文
#             has_chinese = any(bool(re.search(r'[\u4e00-\u9fa5]', desc)) for desc in sample_descriptions)
#             print(f"  样本描述数: {len(sample_descriptions)}")
#             print(f"  包含中文: {has_chinese}")
#             print(f"  样本示例: {sample_descriptions[0][:80]}...")
#
#             if has_chinese:
#                 print(f"  ⚠️ 警告：检测到中文描述，但LLM将输出英文推荐")
#
#         print("\n✅ 数据加载完成")
#
#     def get_user_history_text_english(self, user_id: str, max_items: int = 10) -> str:
#         """
#         获取用户历史交互商品的英文文本描述
#
#         Args:
#             user_id: 用户ID
#             max_items: 最多包含的历史商品数
#
#         Returns:
#             str: 格式化的英文历史商品文本
#         """
#         user_history = self.user_history_data.get(user_id, {})
#         history_items = user_history.get('history_items', [])
#
#         if not history_items:
#             return "This user has no purchase history."
#
#         # 限制数量
#         if len(history_items) > max_items:
#             history_items = history_items[:max_items]
#
#         text_parts = [
#             f"The user has {user_history.get('interaction_count', len(history_items))} historical interactions. Here are some items:\n"]
#
#         for i, item in enumerate(history_items):
#             title = item.get('title', 'Unknown Item')
#             text_parts.append(f"Item {i + 1}: {title}")
#
#             desc = item.get('description', '')
#             if desc:
#                 # 取前200个字符
#                 text_parts.append(f"  Description: {desc[:200]}")
#
#             brand = item.get('brand', '')
#             if brand:
#                 text_parts.append(f"  Brand: {brand}")
#
#             text_parts.append("")
#
#         # 添加偏好信息
#         preferred_categories = user_history.get('preferred_categories', [])
#         if preferred_categories:
#             text_parts.append(f"Preferred Categories: {', '.join(preferred_categories[:5])}")
#
#         preferred_brands = user_history.get('preferred_brands', [])
#         if preferred_brands:
#             text_parts.append(f"Preferred Brands: {', '.join(preferred_brands[:3])}")
#
#         return '\n'.join(text_parts)
#
#     @retry(
#         stop=stop_after_attempt(3),
#         wait=wait_random_exponential(min=3, max=30),
#         retry=(retry_if_exception_type(requests.exceptions.Timeout) |
#                retry_if_exception_type(requests.exceptions.ConnectionError))
#     )
#     def call_llm_api(self, prompt: str, max_tokens: int = 4096) -> str:
#         """调用LLM API"""
#         headers = {
#             'Authorization': f'Bearer {self.llm_api_key}',
#             'Content-Type': 'application/json'
#         }
#
#         data = {
#             'model': 'deepseek-ai/DeepSeek-V3.2',
#             'messages': [{'role': 'user', 'content': prompt}],
#             'max_tokens': max_tokens,
#             'temperature': 0.3
#         }
#
#         response = requests.post(self.llm_api_url, headers=headers, json=data, timeout=120)
#
#         if response.status_code == 200:
#             result = response.json()
#             return result['choices'][0]['message']['content']
#         else:
#             raise Exception(f"API call failed, status code: {response.status_code}")
#
#     def get_llm_recommendations_for_low_interaction(self, user_id: str, top_k: int = 10) -> List[str]:
#         """
#         为低交互用户获取LLM直接推荐商品（英文输出）
#
#         Args:
#             user_id: 用户ID
#             top_k: 推荐商品数量
#
#         Returns:
#             List[str]: 推荐商品名称列表（英文）
#         """
#         # 检查缓存
#         if user_id in self.llm_recommendations_cache:
#             return self.llm_recommendations_cache[user_id]
#
#         # 获取用户历史（英文）
#         history_text = self.get_user_history_text_english(user_id)
#
#         # 构建英文prompt - 明确要求英文输出
#         prompt = f"""You are a professional product recommendation expert. Based on the user's purchase history below, recommend {top_k} specific products the user might be interested in.
#
# **User Purchase History:**
# {history_text}
#
# **Recommendation Requirements:**
# 1. IMPORTANT: All recommendations MUST be in English. Output product names and descriptions in English only.
# 2. Recommend REAL, existing consumer products (not fictional or experimental)
# 3. Products should relate to the user's interests but can offer some novelty
# 4. Each recommendation should include: Product Name and a brief description (core features and benefits)
# 5. Be specific and concrete, avoid vague categories
#
# **Output Format (STRICTLY FOLLOW):**
# Output exactly {top_k} lines, one per recommendation, in this format:
# Product Name 1 | Brief description 1
# Product Name 2 | Brief description 2
# ...
# Product Name {top_k} | Brief description {top_k}
#
# IMPORTANT:
# - DO NOT add any introductory or concluding text
# - DO NOT use markdown formatting
# - DO NOT include brand names in product names (describe generically)
# - ALL output MUST be in English
#
# Start your recommendations now:"""
#
#         print(f"  调用LLM为用户 {user_id} 生成英文推荐...")
#         self.stats['api_calls'] += 1
#
#         try:
#             response = self.call_llm_api(prompt)
#
#             # 验证输出是否为英文
#             has_chinese = bool(re.search(r'[\u4e00-\u9fa5]', response))
#             if has_chinese:
#                 print(f"  ⚠️ LLM输出包含中文，尝试重新生成...")
#                 # 添加更强硬的英文要求
#                 prompt_retry = prompt + "\n\nCRITICAL REMINDER: Your previous response contained Chinese characters. You MUST output ONLY in English. No Chinese characters allowed."
#                 response = self.call_llm_api(prompt_retry)
#                 has_chinese = bool(re.search(r'[\u4e00-\u9fa5]', response))
#                 if has_chinese:
#                     print(f"  ⚠️ 重新生成仍包含中文，过滤中文内容")
#                     # 过滤掉中文字符
#                     response = re.sub(r'[\u4e00-\u9fa5]+', '', response)
#
#             # 解析响应
#             recommendations = []
#             for line in response.strip().split('\n'):
#                 line = line.strip()
#                 # 跳过空行、markdown标记、解释性文本
#                 if not line or line.startswith('#') or line.startswith('```') or line.startswith('Here'):
#                     continue
#
#                 # 移除可能的序号前缀（如 "1. ", "- ", "* "）
#                 line = re.sub(r'^[\d]+[\.\)]\s*', '', line)
#                 line = re.sub(r'^[-*]\s*', '', line)
#
#                 if '|' in line:
#                     parts = line.split('|', 1)
#                     if len(parts) == 2:
#                         name = parts[0].strip()
#                         desc = parts[1].strip()
#                         # 清理可能的引号和特殊字符
#                         name = name.strip('"\'[]')
#                         desc = desc.strip('"\'[]')
#                         if name and len(name) > 2:
#                             # 存储格式: "Product Name: Description"
#                             recommendations.append(f"{name}: {desc}")
#                 elif line and len(line) > 3:
#                     # 没有分隔符，整行作为商品名
#                     cleaned = line.strip('"\'[]')
#                     if len(cleaned) > 2:
#                         recommendations.append(cleaned)
#
#             # 再次过滤中文
#             recommendations = [rec for rec in recommendations if not re.search(r'[\u4e00-\u9fa5]', rec)]
#
#             # 限制数量
#             recommendations = recommendations[:top_k]
#
#             # 缓存结果
#             self.llm_recommendations_cache[user_id] = recommendations
#
#             # 预计算嵌入向量
#             if recommendations:
#                 self.llm_embeddings_cache[user_id] = {}
#                 for item_text in recommendations:
#                     embedding = self.embedding_model.encode(item_text)
#                     self.llm_embeddings_cache[user_id][item_text] = embedding
#
#             print(f"  获取到 {len(recommendations)} 个英文推荐")
#             if recommendations:
#                 print(f"  示例: {recommendations[0][:80]}...")
#
#             return recommendations
#
#         except Exception as e:
#             print(f"  ❌ LLM推荐失败: {e}")
#             return []
#
#     def get_llm_enhancement_for_high_interaction(self, user_id: str, candidate_descriptions: List[str]) -> List[str]:
#         """
#         为高交互用户获取LLM增强关键词（英文输出）
#
#         Args:
#             user_id: 用户ID
#             candidate_descriptions: 候选商品描述列表
#
#         Returns:
#             List[str]: LLM增强后的兴趣关键词（英文）
#         """
#         # 检查缓存
#         cache_key = f"{user_id}_enhancement"
#         if cache_key in self.llm_recommendations_cache:
#             return self.llm_recommendations_cache[cache_key]
#
#         # 获取用户历史（英文）
#         history_text = self.get_user_history_text_english(user_id)
#
#         # 取前5个候选商品描述作为上下文
#         sample_descriptions = [d for d in candidate_descriptions[:5] if d and d != 'nan']
#         sample_text = '\n'.join([f"- {desc[:150]}..." for desc in
#                                  sample_descriptions]) if sample_descriptions else "No sample descriptions available."
#
#         # 构建英文prompt
#         prompt = f"""You are a user interest analysis expert. Based on the user's purchase history and candidate product samples, extract the user's core interest keywords and potential interest directions.
#
# **User Purchase History:**
# {history_text}
#
# **Candidate Product Samples (for context):**
# {sample_text}
#
# **Task Requirements:**
# 1. IMPORTANT: ALL output MUST be in English. No Chinese characters allowed.
# 2. Extract 5-10 core interest keywords from the user's history
# 3. Infer 2-3 potential interest directions the user might explore
# 4. Keywords should be specific product features, functions, or use scenarios
# 5. Keep keywords concise (1-5 words each)
#
# **Output Format (STRICTLY FOLLOW):**
# Core Interest Keywords: keyword1, keyword2, keyword3, ...
# Potential Interest Directions: direction description 1, direction description 2, ...
#
# IMPORTANT:
# - DO NOT add any introductory or concluding text beyond the specified format
# - ALL output MUST be in English
# - Keep directions brief (5-15 words each)
#
# Start your analysis now:"""
#
#         print(f"  调用LLM为用户 {user_id} 生成英文兴趣增强...")
#         self.stats['api_calls'] += 1
#
#         try:
#             response = self.call_llm_api(prompt, max_tokens=2048)
#
#             # 验证输出是否为英文
#             has_chinese = bool(re.search(r'[\u4e00-\u9fa5]', response))
#             if has_chinese:
#                 print(f"  ⚠️ LLM输出包含中文，过滤中文内容")
#                 response = re.sub(r'[\u4e00-\u9fa5]+', '', response)
#
#             # 解析响应，提取关键词
#             keywords = []
#
#             for line in response.strip().split('\n'):
#                 line = line.strip()
#
#                 # 解析 "Core Interest Keywords:" 行
#                 if 'core interest keywords' in line.lower():
#                     # 提取冒号后的内容
#                     if '：' in line:
#                         parts = line.split('：', 1)
#                     elif ':' in line:
#                         parts = line.split(':', 1)
#                     else:
#                         continue
#
#                     if len(parts) >= 2:
#                         keywords_str = parts[1].strip()
#                         # 分割关键词
#                         extracted = [k.strip().strip('"\'[]') for k in keywords_str.split(',') if k.strip()]
#                         keywords.extend(extracted)
#
#                 # 解析 "Potential Interest Directions:" 行
#                 elif 'potential interest directions' in line.lower():
#                     if '：' in line:
#                         parts = line.split('：', 1)
#                     elif ':' in line:
#                         parts = line.split(':', 1)
#                     else:
#                         continue
#
#                     if len(parts) >= 2:
#                         directions_str = parts[1].strip()
#                         extracted = [d.strip().strip('"\'[]') for d in directions_str.split(',') if d.strip()]
#                         keywords.extend(extracted)
#
#             # 如果没有解析到关键词，使用整个响应
#             if not keywords:
#                 # 提取所有非空行作为关键词
#                 keywords = [line.strip().strip('"\'[]') for line in response.strip().split('\n')
#                             if line.strip() and len(line.strip()) > 3]
#                 keywords = keywords[:15]  # 限制数量
#
#             # 过滤中文
#             keywords = [kw for kw in keywords if not re.search(r'[\u4e00-\u9fa5]', kw)]
#
#             # 去重并限制数量
#             keywords = list(dict.fromkeys(keywords))[:15]
#
#             # 缓存
#             self.llm_recommendations_cache[cache_key] = keywords
#
#             # 预计算嵌入向量
#             self.llm_embeddings_cache[cache_key] = {}
#             for keyword in keywords:
#                 embedding = self.embedding_model.encode(keyword)
#                 self.llm_embeddings_cache[cache_key][keyword] = embedding
#
#             print(f"  提取到 {len(keywords)} 个英文兴趣关键词")
#             if keywords:
#                 print(f"  示例: {keywords[:3]}")
#
#             return keywords
#
#         except Exception as e:
#             print(f"  ❌ LLM增强失败: {e}")
#             return []
#
#     def calculate_semantic_similarity(self,
#                                       candidate_text: str,
#                                       reference_texts: List[str],
#                                       reference_embeddings: Dict[str, np.ndarray]) -> float:
#         """
#         计算候选文本与参考文本的语义相似度
#         使用多语言模型以支持英文匹配
#
#         Args:
#             candidate_text: 候选商品文本（英文 title + description）
#             reference_texts: 参考文本列表
#             reference_embeddings: 参考文本的嵌入向量字典
#
#         Returns:
#             float: 最高相似度分数 [0, 1]
#         """
#         if not candidate_text or not reference_embeddings:
#             return 0.0
#
#         # 清理文本
#         candidate_text = str(candidate_text).strip()
#         if not candidate_text or candidate_text.lower() == 'nan':
#             return 0.0
#
#         # 获取候选文本的嵌入向量
#         if candidate_text not in self.description_embedding_cache:
#             embedding = self.embedding_model.encode(candidate_text)
#             self.description_embedding_cache[candidate_text] = embedding
#         else:
#             embedding = self.description_embedding_cache[candidate_text]
#
#         # 计算与所有参考文本的相似度
#         similarities = []
#         for ref_text, ref_embedding in reference_embeddings.items():
#             if isinstance(ref_embedding, list):
#                 ref_embedding = np.array(ref_embedding)
#             sim = 1 - distance.cosine(embedding, ref_embedding)
#             similarities.append(sim)
#
#         return max(similarities) if similarities else 0.0
#
#     def score_candidate_for_low_interaction(self,
#                                             user_id: str,
#                                             candidate: Dict) -> Dict:
#         """
#         为低交互用户的候选商品计算LLM探索分数
#
#         Args:
#             user_id: 用户ID
#             candidate: 候选商品信息
#
#         Returns:
#             Dict: 包含探索分数和方法的信息
#         """
#         # 构建候选商品文本（英文）
#         title = candidate.get('title', '')
#         description = candidate.get('description', '')
#         candidate_text = f"{title} {description}".strip()
#
#         if not candidate_text or candidate_text.lower() == 'nan':
#             return {
#                 'llm_exploration_score': 0.0,
#                 'method': 'LLM_direct_no_description'
#             }
#
#         # 获取LLM英文推荐结果
#         recommendations = self.get_llm_recommendations_for_low_interaction(user_id)
#
#         if not recommendations:
#             return {
#                 'llm_exploration_score': 0.0,
#                 'method': 'LLM_recommendation_failed'
#             }
#
#         # 获取嵌入向量
#         if user_id not in self.llm_embeddings_cache:
#             return {
#                 'llm_exploration_score': 0.0,
#                 'method': 'LLM_embedding_cache_missing'
#             }
#
#         # 计算语义相似度
#         raw_score = self.calculate_semantic_similarity(
#             candidate_text,
#             recommendations,
#             self.llm_embeddings_cache[user_id]
#         )
#
#         # 分数调整：低交互用户的分数范围在0.01-0.3之间
#         adjusted_score = raw_score * 0.3
#
#         # 如果候选商品title/description为空，降低分数
#         if not title and not description:
#             adjusted_score *= 0.5
#
#         return {
#             'llm_exploration_score': adjusted_score,
#             'method': f'LLM_direct (raw_sim={raw_score:.4f})'
#         }
#
#     def score_candidate_for_enhancement(self,
#                                         user_id: str,
#                                         candidate: Dict) -> Dict:
#         """
#         为需要增强的高交互用户候选商品计算LLM增强分数
#
#         Args:
#             user_id: 用户ID
#             candidate: 候选商品信息
#
#         Returns:
#             Dict: 包含增强分数和方法的信息
#         """
#         # 获取该用户的所有候选商品描述（用于上下文）
#         user_candidates = self.user_candidates.get(user_id, [])
#         all_descriptions = [c.get('description', '') for c in user_candidates]
#
#         # 获取LLM英文增强关键词
#         cache_key = f"{user_id}_enhancement"
#         enhancement_keywords = self.get_llm_enhancement_for_high_interaction(user_id, all_descriptions)
#
#         if not enhancement_keywords:
#             # 如果LLM增强失败，返回基于pCTR的默认分数
#             pctr = candidate.get('original_pctr_score', 0.1)
#             return {
#                 'llm_exploration_score': pctr * 0.1,
#                 'method': f'LLM_enhance_failed_pCTR (pctr={pctr:.4f})'
#             }
#
#         # 构建候选商品文本（英文）
#         title = candidate.get('title', '')
#         description = candidate.get('description', '')
#         candidate_text = f"{title} {description}".strip()
#
#         if not candidate_text or candidate_text.lower() == 'nan':
#             return {
#                 'llm_exploration_score': 0.0,
#                 'method': 'LLM_enhance_no_description'
#             }
#
#         # 计算语义相似度
#         if cache_key in self.llm_embeddings_cache:
#             raw_score = self.calculate_semantic_similarity(
#                 candidate_text,
#                 enhancement_keywords,
#                 self.llm_embeddings_cache[cache_key]
#             )
#         else:
#             raw_score = 0.1  # 默认值
#
#         # 增强分数调整：范围在0.001-0.1之间
#         adjusted_score = raw_score * 0.1
#
#         return {
#             'llm_exploration_score': adjusted_score,
#             'method': f'LLM_enhance (raw_sim={raw_score:.4f})'
#         }
#
#     def calculate_all_scores(self):
#         """为所有需要LLM处理的候选商品计算探索分数"""
#         print("\n" + "=" * 60)
#         print("计算LLM探索分数 (英文推荐模式)")
#         print("=" * 60)
#
#         start_time = time.time()
#
#         all_users = list(self.user_candidates.keys())
#         print(f"需要处理的用户数: {len(all_users)}")
#
#         for user_id in tqdm(all_users, desc="计算LLM分数"):
#             candidates = self.user_candidates[user_id]
#
#             # 判断用户类型
#             first_candidate = candidates[0] if candidates else {}
#             kg_score = first_candidate.get('kg_interest_score', 0)
#
#             is_low_interaction = (kg_score == -10)
#
#             for candidate in candidates:
#                 item_id = candidate['item_id']
#
#                 if is_low_interaction:
#                     result = self.score_candidate_for_low_interaction(user_id, candidate)
#                     self.stats['llm_direct_scored'] += 1
#                 else:
#                     result = self.score_candidate_for_enhancement(user_id, candidate)
#                     self.stats['llm_enhanced_scored'] += 1
#
#                 self.exploration_scores[(user_id, item_id)] = result['llm_exploration_score']
#                 self.scoring_methods[(user_id, item_id)] = result['method']
#
#         self.stats['processing_time'] = time.time() - start_time
#
#         # 打印统计
#         print(f"\n计算完成！")
#         print(f"  处理时间: {self.stats['processing_time']:.2f} 秒")
#         print(f"  LLM直接推荐: {self.stats['llm_direct_scored']}")
#         print(f"  LLM语义增强: {self.stats['llm_enhanced_scored']}")
#         print(f"  API调用次数: {self.stats['api_calls']}")
#
#         if self.exploration_scores:
#             scores = list(self.exploration_scores.values())
#             print(f"  分数范围: [{min(scores):.6f}, {max(scores):.6f}]")
#             print(f"  平均分数: {np.mean(scores):.6f}")
#             print(f"  分数>0的比例: {sum(1 for s in scores if s > 0) / len(scores) * 100:.1f}%")
#
#     def save_results(self):
#         """
#         保存LLM探索分数结果
#         输出格式与KG输出完全一致
#         """
#         print("\n" + "=" * 60)
#         print("保存LLM探索分数结果")
#         print("=" * 60)
#
#         results = []
#
#         for _, row in self.llm_input_data.iterrows():
#             user_id = str(row['user_id'])
#             item_id = str(row['item_id'])
#
#             llm_score = self.exploration_scores.get((user_id, item_id), 0.0)
#             method = self.scoring_methods.get((user_id, item_id), 'unknown')
#
#             kg_score = float(row.get('kg_interest_score', 0))
#             pctr_score = float(row.get('original_pctr_score', 0))
#
#             # 计算最终分数
#             if kg_score == -10:
#                 # 低交互用户：LLM分数为主
#                 final_score = pctr_score * 0.3 + llm_score * 0.7
#             elif kg_score == -5:
#                 # 增强用户
#                 final_score = pctr_score * 0.5 + llm_score * 0.5
#             else:
#                 final_score = float(row.get('final_combined_score', 0))
#
#             results.append({
#                 'user_id': user_id,
#                 'item_id': item_id,
#                 'asin': str(row.get('asin', '')),
#                 'title': str(row.get('title', '')),
#                 'description': str(row.get('description', '')),
#                 'is_positive': int(row.get('is_positive', 0)),
#                 'original_pctr_score': pctr_score,
#                 'kg_interest_score': kg_score,
#                 'llm_exploration_score': round(llm_score, 8),
#                 'final_combined_score': round(final_score, 8),
#                 'scoring_method': method
#             })
#
#         results_df = pd.DataFrame(results)
#         results_df = results_df.sort_values(
#             ['user_id', 'final_combined_score'],
#             ascending=[True, False]
#         )
#
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         output_file = os.path.join(self.output_dir, f'llm_exploration_scores_{timestamp}.csv')
#
#         output_columns = ['user_id', 'item_id', 'asin', 'title', 'description',
#                           'is_positive', 'original_pctr_score', 'kg_interest_score',
#                           'llm_exploration_score', 'final_combined_score', 'scoring_method']
#         results_df[output_columns].to_csv(output_file, index=False, encoding='utf-8-sig')
#
#         print(f"✅ LLM探索分数已保存: {output_file}")
#         print(f"   记录数: {len(results_df)}")
#
#         # 分数统计
#         print(f"\n分数统计:")
#         print(f"  LLM分数范围: [{results_df['llm_exploration_score'].min():.6f}, "
#               f"{results_df['llm_exploration_score'].max():.6f}]")
#         print(f"  LLM分数均值: {results_df['llm_exploration_score'].mean():.6f}")
#         print(f"  最终分数范围: [{results_df['final_combined_score'].min():.4f}, "
#               f"{results_df['final_combined_score'].max():.4f}]")
#
#         # 按用户类型统计
#         low_mask = results_df['kg_interest_score'] == -10
#         enh_mask = results_df['kg_interest_score'] == -5
#
#         if low_mask.any():
#             low_df = results_df[low_mask]
#             print(f"\n低交互用户(-10): {len(low_df)} 条")
#             print(f"  LLM分数均值: {low_df['llm_exploration_score'].mean():.6f}")
#
#         if enh_mask.any():
#             enh_df = results_df[enh_mask]
#             print(f"\n增强用户(-5): {len(enh_df)} 条")
#             print(f"  LLM分数均值: {enh_df['llm_exploration_score'].mean():.6f}")
#
#         # 保存统计
#         stats_file = os.path.join(self.output_dir, f'llm_scoring_stats_{timestamp}.json')
#         stats_to_save = self.stats.copy()
#         stats_to_save['llm_score_stats'] = {
#             'min': float(results_df['llm_exploration_score'].min()),
#             'max': float(results_df['llm_exploration_score'].max()),
#             'mean': float(results_df['llm_exploration_score'].mean()),
#             'median': float(results_df['llm_exploration_score'].median()),
#             'std': float(results_df['llm_exploration_score'].std())
#         }
#
#         with open(stats_file, 'w', encoding='utf-8') as f:
#             json.dump(stats_to_save, f, ensure_ascii=False, indent=2)
#
#         print(f"✅ 统计信息已保存: {stats_file}")
#
#         return output_file
#
#     def run(self):
#         """运行完整的LLM探索分数计算流程"""
#         print("=" * 60)
#         print("LLM探索分数计算系统 (英文输出模式)")
#         print("=" * 60)
#
#         try:
#             self.load_data()
#             self.calculate_all_scores()
#             output_file = self.save_results()
#
#             print("\n" + "=" * 60)
#             print("✅ LLM探索分数计算完成！")
#             print(f"   输出文件: {output_file}")
#             print("=" * 60)
#
#         except Exception as e:
#             print(f"\n❌ 执行失败: {e}")
#             import traceback
#             traceback.print_exc()
#
#
# if __name__ == "__main__":
#     scorer = LLMExplorationScorer(
#         llm_input_csv='D:/graduation_recsys/data/KG-related_Files/kg_interest_scores/llm_input_data.csv',
#         user_history_json='D:/graduation_recsys/data/KG-related_Files/kg_interest_scores/llm_user_history.json',
#         output_dir='D:/graduation_recsys/data/KG-related_Files/kg_interest_scores'
#     )
#
#     scorer.run()

# llm_exploration_scoring_fixed.py
"""
LLM探索分数计算系统 - 最终修复版
自动处理模型加载问题，多重回退策略
"""

import os
import sys

# ==================== 环境设置 ====================
import os

PROXY = os.getenv("LOCAL_HTTP_PROXY", "http://127.0.0.1:7897")

os.environ.pop("HF_ENDPOINT", None)
os.environ["HTTP_PROXY"] = PROXY
os.environ["HTTPS_PROXY"] = PROXY

import pandas as pd
import numpy as np
import json
import requests
import time
import re
import warnings
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

warnings.filterwarnings('ignore')

# 尝试导入sklearn（必定可用）
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 尝试导入sentence_transformers（可能不可用）
SENTENCE_TRANSFORMER_AVAILABLE = False
EMBEDDING_MODEL = None

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMER_AVAILABLE = True
except Exception as e:
    print(f"⚠️ sentence-transformers不可用: {e}")
    print("将使用TF-IDF作为嵌入方法")


class LLMExplorationScorer:
    """
    LLM探索分数计算器（多重回退版本）
    优先级：sentence-transformers > TF-IDF
    """

    def __init__(self,
                 llm_input_csv: str = 'D:/graduation_recsys/data/KG-related_Files/kg_interest_scores/llm_input_data.csv',
                 user_history_json: str = 'D:/graduation_recsys/data/KG-related_Files/kg_interest_scores/llm_user_history.json',
                 llm_api_url: str = 'https://api.siliconflow.cn/v1/chat/completions',
                 llm_api_key: str = os.getenv('SILICONFLOW_API_KEY', ''),
                 output_dir: str = 'D:/graduation_recsys/data/KG-related_Files/kg_interest_scores'):
        """
        初始化LLM探索分数计算器
        """
        self.api_session = requests.Session()

        # SiliconFlow不使用HTTP_PROXY/HTTPS_PROXY
        self.api_session.trust_env = False
        self.llm_input_csv = llm_input_csv
        self.user_history_json = user_history_json
        self.llm_api_url = llm_api_url
        self.llm_api_key = llm_api_key
        self.output_dir = output_dir

        # 初始化嵌入模型（多重回退）
        self.embedding_model = None
        self.embedding_method = self._init_embedding_model()

        # TF-IDF（备用方案）
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.text_to_idx = {}
        self.all_texts = []

        # 缓存
        self.llm_recommendations_cache = {}
        self.llm_embeddings_cache = {}
        self.text_embedding_cache = {}

        # 数据存储
        self.llm_input_data = None
        self.user_history_data = {}
        self.user_candidates = defaultdict(list)

        # 结果存储
        self.exploration_scores = {}
        self.scoring_methods = {}

        # 统计信息
        self.stats = {
            'total_records': 0,
            'low_interaction_records': 0,
            'enhancement_records': 0,
            'llm_direct_scored': 0,
            'llm_enhanced_scored': 0,
            'processing_time': 0,
            'api_calls': 0,
            'embedding_method': self.embedding_method
        }

        os.makedirs(output_dir, exist_ok=True)

    def _init_embedding_model(self) -> str:
        """
        初始化嵌入模型（多重回退策略）

        Returns:
            str: 使用的嵌入方法名称
        """
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            return 'tfidf'

        # 尝试加载多个模型（按优先级）
        model_candidates = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            "sentence-transformers/distiluse-base-multilingual-cased-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ]

        for model_name in model_candidates:
            try:
                print(f"尝试加载模型: {model_name}")
                self.embedding_model = SentenceTransformer(
                    model_name,
                    cache_folder='./model_cache'
                )
                dim = self.embedding_model.get_sentence_embedding_dimension()
                print(f"✅ 成功加载: {model_name} (维度: {dim})")
                return f'sentence_transformer_{model_name}'
            except Exception as e:
                print(f"❌ 加载失败: {model_name} - {str(e)[:100]}")
                continue

        print("⚠️ 所有sentence-transformer模型加载失败，使用TF-IDF")
        return 'tfidf'

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        获取文本嵌入向量

        Args:
            text: 输入文本

        Returns:
            np.ndarray or None: 嵌入向量
        """
        if not text or str(text).strip().lower() == 'nan':
            return None

        text = str(text).strip()

        # 检查缓存
        if text in self.text_embedding_cache:
            return self.text_embedding_cache[text]

        embedding = None

        if self.embedding_model is not None:
            # 使用sentence-transformer
            try:
                embedding = self.embedding_model.encode(text)
            except Exception:
                pass

        if embedding is None and self.tfidf_vectorizer is not None:
            # 使用TF-IDF
            try:
                if self.tfidf_matrix is not None and text in self.text_to_idx:
                    idx = self.text_to_idx[text]
                    embedding = self.tfidf_matrix[idx].toarray().flatten()
                else:
                    embedding = self.tfidf_vectorizer.transform([text]).toarray().flatten()
            except Exception:
                pass

        if embedding is not None:
            self.text_embedding_cache[text] = embedding

        return embedding

    def _build_tfidf(self):
        """构建全局TF-IDF矩阵"""
        print("构建TF-IDF矩阵...")

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )

        self.all_texts = []
        self.text_to_idx = {}

        # 收集所有文本
        for user_id, candidates in self.user_candidates.items():
            for cand in candidates:
                text = f"{cand.get('title', '')} {cand.get('description', '')}"
                text = text.strip()
                if text and text.lower() != 'nan':
                    self.all_texts.append(text)
                    self.text_to_idx[text] = len(self.all_texts) - 1

        # 也收集用户历史文本
        for user_id, history in self.user_history_data.items():
            for item in history.get('history_items', []):
                text = f"{item.get('title', '')} {item.get('description', '')}"
                text = text.strip()
                if text and text.lower() != 'nan' and text not in self.text_to_idx:
                    self.all_texts.append(text)
                    self.text_to_idx[text] = len(self.all_texts) - 1

        if self.all_texts:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.all_texts)
            print(f"✅ TF-IDF矩阵: {self.tfidf_matrix.shape}")
        else:
            print("⚠️ 没有有效文本用于构建TF-IDF")

    def load_data(self):
        """加载数据"""
        print("=" * 60)
        print("加载LLM处理数据")
        print("=" * 60)

        # 1. 加载CSV
        print(f"\n1. 加载LLM输入数据: {self.llm_input_csv}")

        if not os.path.exists(self.llm_input_csv):
            raise FileNotFoundError(f"文件不存在: {self.llm_input_csv}")

        encodings = ['utf-8-sig', 'utf-8', 'gb18030', 'gbk']
        for enc in encodings:
            try:
                self.llm_input_data = pd.read_csv(self.llm_input_csv, encoding=enc)
                print(f"  ✅ 使用 {enc} 编码")
                break
            except:
                continue

        if self.llm_input_data is None:
            raise ValueError("无法读取CSV文件")

        print(f"  数据: {self.llm_input_data.shape[0]} 行, {self.llm_input_data.shape[1]} 列")

        neg10 = (self.llm_input_data['kg_interest_score'] == -10).sum()
        neg5 = (self.llm_input_data['kg_interest_score'] == -5).sum()
        print(f"  低交互(-10): {neg10}, 增强(-5): {neg5}")

        self.stats['total_records'] = len(self.llm_input_data)
        self.stats['low_interaction_records'] = neg10
        self.stats['enhancement_records'] = neg5

        # 2. 按用户分组
        print("\n2. 按用户分组...")
        for _, row in self.llm_input_data.iterrows():
            user_id = str(row['user_id'])
            self.user_candidates[user_id].append({
                'item_id': str(row['item_id']),
                'asin': str(row.get('asin', '')),
                'title': str(row.get('title', '')),
                'description': str(row.get('description', '')),
                'is_positive': int(row.get('is_positive', 0)),
                'original_pctr_score': float(row.get('original_pctr_score', 0)),
                'kg_interest_score': float(row.get('kg_interest_score', 0)),
                'final_combined_score': float(row.get('final_combined_score', 0))
            })

        print(f"  用户数: {len(self.user_candidates)}")

        # 3. 加载用户历史
        print(f"\n3. 加载用户历史: {self.user_history_json}")
        if os.path.exists(self.user_history_json):
            with open(self.user_history_json, 'r', encoding='utf-8') as f:
                self.user_history_data = json.load(f)
            print(f"  加载了 {len(self.user_history_data)} 个用户")
        else:
            print(f"  ⚠️ 文件不存在，使用空历史")
            self.user_history_data = {}

        # 4. 构建TF-IDF（如果sentence-transformer不可用）
        if self.embedding_model is None:
            print("\n4. 构建TF-IDF矩阵...")
            self._build_tfidf()

        print(f"\n✅ 数据加载完成 (嵌入方法: {self.embedding_method})")

    def get_user_history_text(self, user_id: str, max_items: int = 10) -> str:
        """获取用户历史文本"""
        history = self.user_history_data.get(user_id, {})
        items = history.get('history_items', [])

        if not items:
            return "No purchase history available."

        items = items[:max_items]

        parts = [f"The user has {history.get('interaction_count', len(items))} historical purchases:\n"]

        for i, item in enumerate(items):
            parts.append(f"Item {i + 1}: {item.get('title', 'Unknown')}")
            desc = item.get('description', '')
            if desc:
                parts.append(f"  Description: {desc[:200]}")
            brand = item.get('brand', '')
            if brand:
                parts.append(f"  Brand: {brand}")
            parts.append("")

        return '\n'.join(parts)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=3, max=30),
        retry=(
                retry_if_exception_type(requests.exceptions.Timeout)
                | retry_if_exception_type(requests.exceptions.ConnectionError)
                | retry_if_exception_type(requests.exceptions.SSLError)
        ),
        reraise=True
    )

    def call_llm(self, prompt: str, max_tokens: int = 4096) -> str:
        """调用LLM API"""
        headers = {
            'Authorization': f'Bearer {self.llm_api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'model': 'deepseek-ai/DeepSeek-V3.2',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens,
            'temperature': 0.3
        }

        response = self.api_session.post(
            self.llm_api_url,
            headers=headers,
            json=data,
            timeout=(15, 120)
        )

        if response.status_code == 402:
            raise RuntimeError(
                f"SiliconFlow 计费或额度错误：{response.text[:1000]}"
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"SiliconFlow API {response.status_code}: "
                f"{response.text[:1000]}"
            )

        return response.json()["choices"][0]["message"]["content"]


    def get_llm_recommendations(self, user_id: str, top_k: int = 10) -> List[str]:
        """获取LLM推荐（英文）"""
        if user_id in self.llm_recommendations_cache:
            return self.llm_recommendations_cache[user_id]

        history = self.get_user_history_text(user_id)

        prompt = f"""You are a product recommendation expert. Based on the purchase history, recommend {top_k} specific products.

**Purchase History:**
{history}

**Requirements:**
1. ALL output MUST be in English only
2. Recommend REAL existing products
3. Format: Product Name | Brief description

**Output exactly {top_k} lines:**
Product 1 | Description 1
...
Product {top_k} | Description {top_k}

DO NOT add any other text. Start now:"""



        print(f"  LLM推荐: user={user_id}")
        self.stats['api_calls'] += 1

        try:
            response = self.call_llm(prompt)
            # 过滤中文
            if re.search(r'[\u4e00-\u9fa5]', response):
                response = re.sub(r'[\u4e00-\u9fa5]+', '', response)

            recommendations = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('```'):
                    continue
                line = re.sub(r'^[\d]+[\.\)]\s*', '', line)

                if '|' in line:
                    parts = line.split('|', 1)
                    name = parts[0].strip().strip('"\'')
                    desc = parts[1].strip().strip('"\'') if len(parts) > 1 else ''
                    if name and len(name) > 2:
                        recommendations.append(f"{name}: {desc}" if desc else name)
                elif len(line) > 3:
                    recommendations.append(line.strip('"\''))

            recommendations = [r for r in recommendations if not re.search(r'[\u4e00-\u9fa5]', r)]
            recommendations = recommendations[:top_k]

            self.llm_recommendations_cache[user_id] = recommendations

            # 预计算嵌入
            if recommendations:
                self.llm_embeddings_cache[user_id] = {}
                for text in recommendations:
                    emb = self._get_embedding(text)
                    if emb is not None:
                        self.llm_embeddings_cache[user_id][text] = emb

            print(f"  获得 {len(recommendations)} 个推荐")
            return recommendations

        except Exception as e:
            print(f"  ❌ 失败: {type(e).__name__}: {e}")
            self.llm_recommendations_cache[user_id] = []
            return []

    def get_llm_enhancement(self, user_id: str, candidate_descs: List[str]) -> List[str]:
        """获取LLM增强关键词"""
        cache_key = f"{user_id}_enh"
        if cache_key in self.llm_recommendations_cache:
            return self.llm_recommendations_cache[cache_key]

        history = self.get_user_history_text(user_id)
        samples = [d for d in candidate_descs[:5] if d and d != 'nan']
        sample_text = '\n'.join([f"- {d[:150]}..." for d in samples]) if samples else "N/A"

        prompt = f"""Extract user interest keywords from purchase history.

**History:**
{history}

**Output Format:**
Core Keywords: keyword1, keyword2, ...
Interest Directions: direction1, direction2, ...

ALL output MUST be in English. No Chinese.

Start:"""

        print(f"  LLM增强: user={user_id}")
        self.stats['api_calls'] += 1

        try:
            response = self.call_llm(prompt, max_tokens=2048)
            if re.search(r'[\u4e00-\u9fa5]', response):
                response = re.sub(r'[\u4e00-\u9fa5]+', '', response)

            keywords = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if 'keyword' in line.lower() or 'direction' in line.lower():
                    parts = line.split('：') if '：' in line else line.split(':', 1)
                    if len(parts) >= 2:
                        kws = [k.strip().strip('"\'') for k in parts[1].split(',') if k.strip()]
                        keywords.extend(kws)

            if not keywords:
                keywords = [l.strip().strip('"\'') for l in response.strip().split('\n')
                            if l.strip() and len(l.strip()) > 3][:15]

            keywords = [k for k in keywords if not re.search(r'[\u4e00-\u9fa5]', k)]
            keywords = list(dict.fromkeys(keywords))[:15]

            self.llm_recommendations_cache[cache_key] = keywords

            if keywords:
                self.llm_embeddings_cache[cache_key] = {}
                for kw in keywords:
                    emb = self._get_embedding(kw)
                    if emb is not None:
                        self.llm_embeddings_cache[cache_key][kw] = emb

            print(f"  提取 {len(keywords)} 个关键词")
            return keywords

        except Exception as e:

            print(f"  ❌ 失败: {type(e).__name__}: {e}")

            self.llm_recommendations_cache[cache_key] = []

            return []

    def calc_similarity(self, text: str, ref_embeddings: Dict) -> float:
        """计算余弦相似度"""
        if not text or not ref_embeddings:
            return 0.0

        text = str(text).strip()
        if not text or text.lower() == 'nan':
            return 0.0

        emb = self._get_embedding(text)
        if emb is None:
            return 0.0

        sims = []
        for ref_emb in ref_embeddings.values():
            if ref_emb is not None:
                dot = np.dot(emb, ref_emb)
                norm = np.linalg.norm(emb) * np.linalg.norm(ref_emb)
                if norm > 0:
                    sims.append(max(0.0, float(dot / norm)))

        return max(sims) if sims else 0.0

    def calculate_all_scores(self):
        """计算所有分数"""
        print("\n" + "=" * 60)
        print(f"计算LLM探索分数 ({self.embedding_method})")
        print("=" * 60)

        t0 = time.time()
        users = list(self.user_candidates.keys())
        print(f"用户数: {len(users)}")

        for user_id in tqdm(users, desc="计算分数"):
            candidates = self.user_candidates[user_id]
            kg_score = candidates[0].get('kg_interest_score', 0) if candidates else 0
            is_low = (kg_score == -10)

            for cand in candidates:
                item_id = cand['item_id']

                if is_low:
                    recs = self.get_llm_recommendations(user_id)
                    if recs and user_id in self.llm_embeddings_cache:
                        text = f"{cand.get('title', '')} {cand.get('description', '')}"
                        score = self.calc_similarity(text, self.llm_embeddings_cache[user_id]) * 0.3
                        method = f'LLM_direct_{self.embedding_method}'
                    else:
                        score = 0.0
                        method = 'LLM_failed'
                    self.stats['llm_direct_scored'] += 1
                else:
                    descs = [c.get('description', '') for c in candidates]
                    kws = self.get_llm_enhancement(user_id, descs)
                    cache_key = f"{user_id}_enh"
                    if kws and cache_key in self.llm_embeddings_cache:
                        text = f"{cand.get('title', '')} {cand.get('description', '')}"
                        score = self.calc_similarity(text, self.llm_embeddings_cache[cache_key]) * 0.1
                        method = f'LLM_enhance_{self.embedding_method}'
                    else:
                        score = cand.get('original_pctr_score', 0.1) * 0.1
                        method = 'LLM_fallback'
                    self.stats['llm_enhanced_scored'] += 1

                self.exploration_scores[(user_id, item_id)] = score
                self.scoring_methods[(user_id, item_id)] = method

        self.stats['processing_time'] = time.time() - t0

        print(f"\n完成! 耗时: {self.stats['processing_time']:.1f}s")
        print(f"  直接推荐: {self.stats['llm_direct_scored']}")
        print(f"  语义增强: {self.stats['llm_enhanced_scored']}")
        print(f"  API调用: {self.stats['api_calls']}")

    def save_results(self):
        """保存结果"""
        print("\n" + "=" * 60)
        print("保存结果")
        print("=" * 60)

        results = []
        for _, row in self.llm_input_data.iterrows():
            uid = str(row['user_id'])
            iid = str(row['item_id'])

            llm_score = self.exploration_scores.get((uid, iid), 0.0)
            method = self.scoring_methods.get((uid, iid), 'unknown')

            kg = float(row.get('kg_interest_score', 0))
            pctr = float(row.get('original_pctr_score', 0))

            if kg == -10:
                final = pctr * 0.3 + llm_score * 0.7
            elif kg == -5:
                final = pctr * 0.5 + llm_score * 0.5
            else:
                final = float(row.get('final_combined_score', 0))

            results.append({
                'user_id': uid,
                'item_id': iid,
                'asin': str(row.get('asin', '')),
                'title': str(row.get('title', '')),
                'description': str(row.get('description', '')),
                'is_positive': int(row.get('is_positive', 0)),
                'original_pctr_score': pctr,
                'kg_interest_score': kg,
                'llm_exploration_score': round(llm_score, 8),
                'final_combined_score': round(final, 8),
                'scoring_method': method
            })

        df = pd.DataFrame(results)
        df = df.sort_values(['user_id', 'final_combined_score'], ascending=[True, False])

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output = os.path.join(self.output_dir, f'llm_exploration_scores_{ts}.csv')

        cols = ['user_id', 'item_id', 'asin', 'title', 'description',
                'is_positive', 'original_pctr_score', 'kg_interest_score',
                'llm_exploration_score', 'final_combined_score', 'scoring_method']
        df[cols].to_csv(output, index=False, encoding='utf-8-sig')

        print(f"✅ 保存: {output}")
        print(f"  记录: {len(df)}")
        print(f"  方法: {self.embedding_method}")
        return output

    def run(self):
        """运行"""
        print("=" * 60)
        print(f"LLM探索分数计算 ({self.embedding_method})")
        print("=" * 60)

        try:
            self.load_data()
            self.calculate_all_scores()
            output = self.save_results()
            print(f"\n✅ 完成! 输出: {output}")
        except Exception as e:
            print(f"\n❌ 失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    scorer = LLMExplorationScorer(
        llm_input_csv='D:/graduation_recsys/llm_datadeal_for_historyitem_descri/data_for_llm_rec/llm_input_data.csv',
        user_history_json='D:/graduation_recsys/llm_datadeal_for_historyitem_descri/data_for_llm_rec/llm_user_history.json',
        output_dir='D:/graduation_recsys/data/KG-related_Files/llm_interest_scores'
    )
    scorer.run()
