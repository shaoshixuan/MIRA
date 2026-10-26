# kg_interest_scoring.py
import pickle
import numpy as np
import pandas as pd
import os
import warnings
from collections import defaultdict
from tqdm import tqdm
import scipy.sparse as sp
import time
import json
from gc import collect

warnings.filterwarnings('ignore')


class KGInterestScorer:
    """
    第四阶段：候选池商品潜在兴趣分数计算
    基于知识图谱的个性化随机游走计算候选商品潜在兴趣分数
    """

    def __init__(self,
                 preprocessed_dir='./kg_preprocessed',
                 transfer_matrix_dir='./kg_transfer_matrix',
                 personalized_vectors_dir='./kg_personalized_vectors',
                 candidate_file='new_item_description4llm.csv',
                 output_dir='./kg_interest_scores'):
        """
        初始化兴趣分数计算器

        Args:
            preprocessed_dir: 第一阶段预处理结果目录
            transfer_matrix_dir: 第二阶段转移矩阵目录
            personalized_vectors_dir: 第三阶段个性化向量目录
            candidate_file: 候选池商品数据文件
            output_dir: 输出目录
        """
        self.preprocessed_dir = preprocessed_dir
        self.transfer_matrix_dir = transfer_matrix_dir
        self.personalized_vectors_dir = personalized_vectors_dir
        self.candidate_file = candidate_file
        self.output_dir = output_dir

        # 数据存储
        self.graph = None
        self.transfer_matrix = None
        self.node_index = None
        self.index_node = None
        self.user_restart_vectors = None
        self.entity_id_to_info = None

        # 用户交互次数缓存
        self.user_interaction_counts = {}

        # 候选池数据
        self.candidate_data = None
        self.user_candidates = defaultdict(list)

        # 分数计算结果
        self.interest_scores = {}
        self.final_scores = {}

        # LLM处理标记
        self.users_needing_llm = set()
        self.users_kg_full = set()
        self.users_kg_partial = set()

        # 统计信息
        self.stats = {
            'total_candidates': 0,
            'users_processed': 0,
            'users_kg_scored': 0,
            'users_other_scored': 0,
            'avg_candidates_per_user': 0,
            'computation_time': 0,
            'score_distribution': {}
        }

        # 参数设置
        self.params = {
            'restart_prob': 0.4,
            'max_iterations': 100,
            'convergence_threshold': 1e-6,
            'exploration_bonus': 1.5,
            'kg_weight': 0.3,
            'pctr_weight': 0.7,
            'min_interactions_for_kg': 5
        }

    # ==================== 数据加载方法 ====================

    def load_all_data(self):
        """加载所有必要数据"""
        print("=" * 50)
        print("加载所有必要数据")
        print("=" * 50)

        try:
            # 1. 加载图结构
            print("1. 加载图结构...")
            with open(os.path.join(self.preprocessed_dir, 'graph_adjacency.pkl'), 'rb') as f:
                self.graph = pickle.load(f)
            print(f"   ✅ 图结构加载完成，节点数: {len(self.graph)}")

            # 2. 加载实体映射
            print("2. 加载实体映射...")
            with open(os.path.join(self.preprocessed_dir, 'entity_mappings.pkl'), 'rb') as f:
                mappings = pickle.load(f)
                self.entity_id_to_info = mappings['entity_id_to_info']
            print(f"   ✅ 实体映射加载完成")

            # 3. 加载转移矩阵
            print("3. 加载转移矩阵...")
            matrix_path = os.path.join(self.transfer_matrix_dir, 'transfer_matrix.npz')
            self.transfer_matrix = sp.load_npz(matrix_path)
            print(f"   ✅ 转移矩阵加载完成，形状: {self.transfer_matrix.shape}")

            # 4. 加载节点索引
            print("4. 加载节点索引...")
            with open(os.path.join(self.transfer_matrix_dir, 'node_indices.pkl'), 'rb') as f:
                index_data = pickle.load(f)
                self.node_index = index_data['node_index']
                self.index_node = index_data['index_node']
            print(f"   ✅ 节点索引加载完成，节点数: {len(self.node_index)}")

            # 5. 加载候选池数据
            print("5. 加载候选池数据...")
            self._load_candidate_data()

            if not self.user_candidates:
                print("  ⚠️ 警告：没有候选池数据，无法继续")
                return

            # 6. 加载个性化重启向量
            print("6. 加载个性化重启向量...")
            self._load_restart_vectors()

            # 7. 计算用户交互次数
            print("7. 计算用户交互次数...")
            self._load_user_interaction_counts()
            print(f"   ✅ 用户交互次数计算完成，用户数: {len(self.user_interaction_counts)}")

            print("\n✅ 所有数据加载完成！")

        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _load_restart_vectors(self):
        """加载重启向量"""
        print("  加载个性化重启向量...")

        try:
            batches_info_path = os.path.join(self.personalized_vectors_dir, 'batches_info.pkl')
            if not os.path.exists(batches_info_path):
                print("  ⚠️ 批次信息文件不存在，使用简化加载")
                self._load_restart_vectors_simple()
                return

            with open(batches_info_path, 'rb') as f:
                batches_info = pickle.load(f)

            num_batches = batches_info.get('num_batches', 0)
            total_users = batches_info.get('total_users', 0)
            print(f"  总用户数: {total_users}, 批次: {num_batches}")

            candidate_users = set(self.user_candidates.keys())
            print(f"  候选池用户数: {len(candidate_users)}")

            loaded_users = 0
            self.user_restart_vectors = {}

            for batch_idx in tqdm(range(num_batches), desc="加载重启向量批次"):
                batch_path = os.path.join(self.personalized_vectors_dir, f'restart_vectors_batch_{batch_idx}.pkl')

                if not os.path.exists(batch_path):
                    continue

                try:
                    with open(batch_path, 'rb') as f:
                        batch_data = pickle.load(f)

                    for user_id, user_data in batch_data.items():
                        if user_id in candidate_users:
                            vector = sp.csr_matrix(
                                (user_data['vector_data'],
                                 ([0] * len(user_data['vector_indices']),
                                  user_data['vector_indices'])),
                                shape=user_data['vector_shape']
                            )
                            self.user_restart_vectors[user_id] = {
                                'vector': vector,
                                'strategy': user_data.get('strategy', 'unknown'),
                                'interaction_count': user_data.get('interaction_count', 0)
                            }
                            loaded_users += 1

                    del batch_data
                    collect()

                except Exception as e:
                    print(f"  加载批次 {batch_idx} 失败: {e}")
                    continue

            print(f"  ✅ 重启向量加载完成，加载了 {loaded_users} 个用户的重启向量")

            missing_users = candidate_users - set(self.user_restart_vectors.keys())
            if missing_users:
                print(f"  ⚠️ 有 {len(missing_users)} 个候选用户没有重启向量")

        except Exception as e:
            print(f"  加载重启向量失败: {e}")
            self._load_restart_vectors_simple()

    def _load_restart_vectors_simple(self):
        """简化版重启向量加载"""
        print("  使用简化版重启向量加载...")
        self.user_restart_vectors = {}

        user_history_path = os.path.join(self.personalized_vectors_dir, 'user_history.pkl')
        if os.path.exists(user_history_path):
            try:
                with open(user_history_path, 'rb') as f:
                    user_history = pickle.load(f)

                for user_id, items in user_history.items():
                    if user_id in self.user_candidates:
                        self.user_restart_vectors[user_id] = {
                            'vector': None,
                            'strategy': 'placeholder',
                            'interaction_count': len(items)
                        }
                print(f"  从user_history加载了 {len(self.user_restart_vectors)} 个用户信息")
            except MemoryError:
                print("  ⚠️ user_history文件太大，跳过")

        if not self.user_restart_vectors:
            for user_id in self.user_candidates.keys():
                self.user_restart_vectors[user_id] = {
                    'vector': None,
                    'strategy': 'empty',
                    'interaction_count': 0
                }
        print(f"  ✅ 创建了 {len(self.user_restart_vectors)} 个重启向量占位符")

    def _load_user_interaction_counts(self):
        """计算用户交互次数"""
        print("  从kg_train_triples.txt计算用户交互次数...")

        if not self.user_candidates:
            print("  ⚠️ user_candidates为空，跳过")
            return

        possible_paths = [
            './kg_train_triples.txt',
            '../kg_train_triples.txt',
            'D:/graduation_recsys/data/KG-related_Files/kg_train_triples.txt',
            os.path.join(self.preprocessed_dir, 'train_triples.csv')
        ]

        train_file = None
        for path in possible_paths:
            if os.path.exists(path):
                train_file = path
                print(f"  找到训练文件: {path}")
                break

        if not train_file:
            print("  ⚠️ 未找到训练三元组文件，使用重启向量中的交互次数")
            for user_id in self.user_candidates.keys():
                if user_id in self.user_restart_vectors:
                    self.user_interaction_counts[user_id] = self.user_restart_vectors[user_id].get('interaction_count',
                                                                                                   0)
                else:
                    self.user_interaction_counts[user_id] = 0
            return

        try:
            if train_file.endswith('.csv'):
                train_triples = pd.read_csv(train_file)
                user_col = train_triples.columns[0]
            else:
                train_triples = pd.read_csv(train_file, sep='\t', header=None,
                                            names=['head', 'tail', 'relation'])
                user_col = 'head'

            train_triples[user_col] = train_triples[user_col].astype(str).apply(self._format_id_to_int)
            user_counts = train_triples[user_col].value_counts()

            candidate_users = list(self.user_candidates.keys())
            matched_users = 0

            for user_id in candidate_users:
                if user_id in user_counts:
                    self.user_interaction_counts[user_id] = user_counts[user_id]
                    matched_users += 1
                else:
                    self.user_interaction_counts[user_id] = 0

            print(f"  总用户数: {len(candidate_users)}, 匹配的用户数: {matched_users}")

            interaction_values = list(self.user_interaction_counts.values())
            if interaction_values:
                print(f"  交互≥{self.params['min_interactions_for_kg']}次的用户数: "
                      f"{sum(1 for v in interaction_values if v >= self.params['min_interactions_for_kg'])}")

        except Exception as e:
            print(f"  读取训练三元组文件失败: {e}")
            for user_id in self.user_candidates.keys():
                if user_id in self.user_restart_vectors:
                    self.user_interaction_counts[user_id] = self.user_restart_vectors[user_id].get('interaction_count',
                                                                                                   0)
                else:
                    self.user_interaction_counts[user_id] = 0

    def _load_candidate_data(self):
        """加载候选池数据"""
        print(f"  从文件 {self.candidate_file} 加载候选池数据...")

        possible_paths = [
            self.candidate_file,
            f'./{self.candidate_file}',
            f'../{self.candidate_file}',
            f'D:/graduation_recsys/data/KG-related_Files/{self.candidate_file}',
        ]

        for file_path in possible_paths:
            if os.path.exists(file_path):
                self.candidate_file = file_path
                print(f"  找到文件: {file_path}")
                break

        if not os.path.exists(self.candidate_file):
            raise FileNotFoundError(f"候选池文件不存在: {self.candidate_file}")

        print(f"  正在读取文件: {self.candidate_file}")

        # 尝试多种编码
        encodings_to_try = ['gb18030', 'gbk', 'utf-8', 'utf-8-sig', 'latin-1']
        successful_encoding = None

        for encoding in encodings_to_try:
            try:
                sample_data = pd.read_csv(self.candidate_file, nrows=5, encoding=encoding)
                print(f"  ✅ {encoding} 编码成功！")
                successful_encoding = encoding
                break
            except Exception as e:
                print(f"  ❌ {encoding} 编码失败: {str(e)[:100]}")
                continue

        if successful_encoding is None:
            raise ValueError(f"无法用任何编码读取文件: {self.candidate_file}")

        print(f"  列名: {list(sample_data.columns)}")

        # 确定需要的列
        needed_columns = []
        column_mapping = {}

        for col in sample_data.columns:
            col_lower = col.lower().strip()

            if 'user' in col_lower and 'id' in col_lower:
                column_mapping[col] = 'user_id'
                needed_columns.append(col)
            elif col_lower == 'item_id':
                column_mapping[col] = 'item_id'
                needed_columns.append(col)
            elif 'pctr' in col_lower or 'ctr' in col_lower:
                column_mapping[col] = 'pctr_score'
                needed_columns.append(col)
            elif col_lower == 'asin':
                needed_columns.append(col)
            elif col_lower == 'title':
                needed_columns.append(col)
            elif col_lower == 'description':
                needed_columns.append(col)
            elif col_lower == 'is_positive':
                needed_columns.append(col)
            elif col_lower == 'rank_in_candidates':
                needed_columns.append(col)

        print(f"  需要的列: {needed_columns}")

        # 分批读取
        chunks = []
        chunk_size = 10000

        for chunk in pd.read_csv(
                self.candidate_file,
                usecols=needed_columns,
                chunksize=chunk_size,
                encoding=successful_encoding,
                on_bad_lines='skip'
        ):
            if column_mapping:
                chunk = chunk.rename(columns=column_mapping)

            if 'user_id' in chunk.columns:
                chunk['user_id'] = chunk['user_id'].apply(
                    lambda x: self._format_id_to_int(x) if pd.notna(x) else ''
                )
            if 'item_id' in chunk.columns:
                chunk['item_id'] = chunk['item_id'].apply(
                    lambda x: str(int(float(x))) if pd.notna(x) else ''
                )

            chunks.append(chunk)

        self.candidate_data = pd.concat(chunks, ignore_index=True)
        print(f"  成功读取，总行数: {len(self.candidate_data)}")

        # 确保有需要的列
        required_cols = ['user_id', 'item_id']
        for col in required_cols:
            if col not in self.candidate_data.columns:
                raise ValueError(f"CSV文件缺少必要列: {col}")

        # 按用户分组候选商品
        user_candidates_count = 0
        for idx, row in self.candidate_data.iterrows():
            try:
                user_id = str(row['user_id'])
                item_id = str(row['item_id'])

                if not user_id or user_id == 'nan' or not item_id or item_id == 'nan':
                    continue

                pctr_score = 0.0
                if 'pctr_score' in row and pd.notna(row.get('pctr_score')):
                    try:
                        pctr_score = float(row['pctr_score'])
                    except:
                        pctr_score = 0.0

                asin = str(row.get('asin', '')) if pd.notna(row.get('asin')) else ''
                title = str(row.get('title', '')) if pd.notna(row.get('title')) else ''
                description = str(row.get('description', '')) if pd.notna(row.get('description')) else ''
                is_positive = int(row.get('is_positive', 0)) if pd.notna(row.get('is_positive')) else 0

                self.user_candidates[user_id].append({
                    'item_id': item_id,
                    'pctr_score': pctr_score,
                    'asin': asin,
                    'title': title,
                    'description': description,
                    'is_positive': is_positive
                })
                user_candidates_count += 1

            except Exception as e:
                print(f"  处理行 {idx} 时出错: {e}")
                continue

        print(f"  成功处理了 {user_candidates_count} 条候选记录")
        print(f"  用户数: {len(self.user_candidates)}")

        self.stats['total_candidates'] = user_candidates_count
        self.stats['users_processed'] = len(self.user_candidates)

        if self.user_candidates:
            avg_candidates = user_candidates_count / len(self.user_candidates)
            self.stats['avg_candidates_per_user'] = avg_candidates
            print(f"  平均候选商品数/用户: {avg_candidates:.2f}")

    def _format_id_to_int(self, x):
        """将ID格式化为整数"""
        if isinstance(x, (pd.Series, pd.DataFrame)):
            raise ValueError(f"_format_id_to_int不能直接处理Series/DataFrame")

        if pd.isna(x):
            return ""

        try:
            if isinstance(x, str):
                x = x.strip()
                if x.endswith('.0'):
                    x = x[:-2]
            return str(int(float(x)))
        except:
            return str(x)

    # ==================== 核心计算方法 ====================

    def calculate_interest_scores(self):
        """
        计算候选池商品潜在兴趣分数（混合策略版）

        策略：
        - 交互≥5 + 有重启向量 + 商品在KG中：使用KG PageRank分数
        - 交互≥5 + 商品不在KG中：使用启发式分数
        - 交互≥5 + 分数区分度不足：标记需要LLM增强（kg_score=-5）
        - 交互<5：标记需要LLM处理（kg_score=-10）
        """
        print("\n" + "=" * 60)
        print("计算候选池商品潜在兴趣分数（混合策略）")
        print("=" * 60)

        start_time = time.time()

        # 初始化
        self.users_needing_llm = set()
        self.users_kg_full = set()
        self.users_kg_partial = set()

        all_users = list(self.user_candidates.keys())
        print(f"需要处理 {len(all_users)} 个用户的候选商品")

        # 统计用户交互次数
        user_interaction_counts = {}
        for user_id in all_users:
            user_interaction_counts[user_id] = self.user_interaction_counts.get(user_id, 0)

        high_interaction = [u for u, c in user_interaction_counts.items()
                            if c >= self.params['min_interactions_for_kg']]
        low_interaction = [u for u, c in user_interaction_counts.items()
                           if c < self.params['min_interactions_for_kg']]

        print(f"\n用户分类统计:")
        print(f"  高交互用户(≥{self.params['min_interactions_for_kg']}次): {len(high_interaction)}")
        print(f"  低交互用户(<{self.params['min_interactions_for_kg']}次): {len(low_interaction)}")

        # 统计变量
        stats = {
            'kg_scored_users': 0,
            'llm_marked_users': 0,
            'placeholder_users': 0,
            'items_in_kg': 0,
            'items_not_in_kg': 0,
        }

        # 分批处理
        batch_size = 100
        num_batches = (len(all_users) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(all_users))
            batch_users = all_users[start_idx:end_idx]

            print(f"\n处理批次 {batch_idx + 1}/{num_batches}: {len(batch_users)} 个用户")

            for user_id in tqdm(batch_users, desc="计算兴趣分数"):
                interaction_count = user_interaction_counts[user_id]
                candidates = self.user_candidates[user_id]

                # 策略选择
                if interaction_count >= self.params['min_interactions_for_kg']:
                    user_scores = self._calculate_kg_with_fallback(
                        user_id, candidates, interaction_count
                    )
                    has_llm_mark = any(s == -5 for s in user_scores)
                    if has_llm_mark:
                        self.users_needing_llm.add(user_id)
                        stats['llm_marked_users'] += 1
                        self.users_kg_partial.add(user_id)
                    else:
                        self.users_kg_full.add(user_id)
                        stats['kg_scored_users'] += 1
                else:
                    user_scores = [-10] * len(candidates)
                    self.users_needing_llm.add(user_id)
                    stats['placeholder_users'] += 1

                # 统计商品覆盖
                for item_data in candidates:
                    if item_data['item_id'] in self.node_index:
                        stats['items_in_kg'] += 1
                    else:
                        stats['items_not_in_kg'] += 1

                # 保存分数
                for item_data, kg_score in zip(candidates, user_scores):
                    item_id = item_data['item_id']
                    pctr_score = item_data.get('pctr_score', 0.0)
                    self.interest_scores[(user_id, item_id)] = kg_score
                    self.final_scores[(user_id, item_id)] = self._compute_final_score(pctr_score, kg_score)

            collect()

        # 更新统计
        elapsed_time = time.time() - start_time
        self._update_stats(stats, elapsed_time)
        self._print_scoring_report(stats)
        self._save_llm_user_list()

        return stats

    def _calculate_kg_with_fallback(self, user_id, candidates, interaction_count):
        """
        高交互用户的KG计算（带回退策略）
        """
        # 第一步：尝试KG计算
        kg_scores = None

        if user_id in self.user_restart_vectors:
            vector_data = self.user_restart_vectors[user_id]
            if vector_data.get('vector') is not None and vector_data['vector'].nnz > 0:
                try:
                    kg_scores = self._run_pagerank_for_user(user_id, candidates)
                except Exception as e:
                    kg_scores = None

        # 第二步：KG计算失败，使用纯启发式
        if kg_scores is None:
            return self._calculate_pure_heuristic(candidates, interaction_count)

        # 第三步：混合KG分数和启发式分数
        scores = []
        items_in_kg_scores = []

        for i, (item_data, kg_score) in enumerate(zip(candidates, kg_scores)):
            item_id = item_data['item_id']

            if item_id in self.node_index and kg_score > 0:
                scores.append(kg_score)
                items_in_kg_scores.append(kg_score)
            elif item_id in self.node_index:
                min_score = self._get_min_positive_score(kg_scores)
                s = min_score if min_score > 0 else 1e-8
                scores.append(s)
                items_in_kg_scores.append(s)
            else:
                heuristic_score = self._get_heuristic_score(
                    item_data, interaction_count, i, len(candidates)
                )
                scores.append(heuristic_score)

        # 第四步：检查是否需要LLM增强
        if self._needs_llm_enhancement(scores, items_in_kg_scores, candidates):
            for i in range(len(scores)):
                if self._is_poorly_scored(scores[i], items_in_kg_scores):
                    scores[i] = -5

        return scores

    def _run_pagerank_for_user(self, user_id, candidates):
        """执行个性化PageRank计算"""
        vector_data = self.user_restart_vectors[user_id]
        restart_vector = vector_data['vector']

        pagerank_scores = self._personalized_pagerank(
            restart_vector,
            alpha=self.params['restart_prob'],
            max_iter=min(self.params['max_iterations'], 50),
            epsilon=self.params['convergence_threshold']
        )

        scores = []
        for item_data in candidates:
            item_id = item_data['item_id']
            if item_id in self.node_index:
                node_idx = self.node_index[item_id]
                scores.append(float(pagerank_scores[node_idx]))
            else:
                scores.append(0.0)

        return scores

    def _personalized_pagerank(self, restart_vector, alpha, max_iter, epsilon):
        """个性化PageRank算法"""
        n = self.transfer_matrix.shape[0]
        current_dist = restart_vector.toarray().flatten()

        for iteration in range(max_iter):
            walk_dist = current_dist @ self.transfer_matrix
            new_dist = alpha * restart_vector.toarray().flatten() + (1 - alpha) * walk_dist
            diff = np.linalg.norm(new_dist - current_dist, 1)
            if diff < epsilon:
                break
            current_dist = new_dist

        return current_dist

    def _calculate_pure_heuristic(self, candidates, interaction_count):
        """纯启发式分数计算"""
        scores = []
        for i, item_data in enumerate(candidates):
            score = self._get_heuristic_score(item_data, interaction_count, i, len(candidates))
            scores.append(score)
        return scores

    def _get_base_heuristic_score(self, interaction_count):
        """根据交互次数获取基础启发式分数"""
        if interaction_count >= 30:
            return 0.08
        elif interaction_count >= 20:
            return 0.06
        elif interaction_count >= 10:
            return 0.04
        elif interaction_count >= 5:
            return 0.03
        else:
            return 0.02

    def _get_heuristic_score(self, item_data, interaction_count, position, total):
        """计算单个商品的启发式分数"""
        base = self._get_base_heuristic_score(interaction_count)
        position_factor = 1.0 - (position / total) * 0.1
        in_kg_bonus = 1.3 if item_data['item_id'] in self.node_index else 1.0
        score = base * position_factor * in_kg_bonus
        noise = np.random.uniform(-0.05, 0.05) * base
        score += noise
        return max(0.001, min(0.15, score))

    def _needs_llm_enhancement(self, scores, in_kg_scores, candidates):
        """判断是否需要LLM增强"""
        if not in_kg_scores:
            return True

        max_score = max(in_kg_scores) if in_kg_scores else 0
        min_score = min(in_kg_scores) if in_kg_scores else 0
        score_range = max_score - min_score
        kg_coverage = len(in_kg_scores) / len(candidates) if candidates else 0

        return any([
            max_score < 0.001,
            score_range < 0.0005,
            kg_coverage < 0.3,
            len(in_kg_scores) < 5
        ])

    def _is_poorly_scored(self, score, in_kg_scores):
        """判断单个分数是否质量不佳"""
        if score <= 0:
            return True
        if not in_kg_scores:
            return True

        max_score = max(in_kg_scores)
        if max_score == 0:
            return True

        avg_score = sum(in_kg_scores) / len(in_kg_scores)
        return score < avg_score * 0.1

    def _get_min_positive_score(self, scores):
        """获取最小正分数"""
        positive_scores = [s for s in scores if s > 0]
        return min(positive_scores) if positive_scores else 1e-10

    def _compute_final_score(self, pctr_score, kg_score):
        """计算最终分数（自适应权重）"""
        if kg_score == -10:
            return pctr_score
        if kg_score == -5:
            return pctr_score * 0.85
        if kg_score < 1e-6:
            return pctr_score * 0.98
        if kg_score < 1e-5:
            return pctr_score * 0.95 + kg_score * 0.05
        if kg_score < 1e-4:
            return pctr_score * 0.90 + kg_score * 0.10
        if kg_score < 0.001:
            return pctr_score * 0.85 + kg_score * 0.15
        if kg_score < 0.01:
            return pctr_score * 0.75 + kg_score * 0.25
        return pctr_score * self.params['pctr_weight'] + kg_score * self.params['kg_weight']

    # ==================== 统计和报告方法 ====================

    def _update_stats(self, stats, elapsed_time):
        """更新统计信息"""
        total_items = stats['items_in_kg'] + stats['items_not_in_kg']
        self.stats.update({
            'total_users': len(self.user_candidates),
            'kg_scored_users': stats['kg_scored_users'],
            'llm_marked_users': stats['llm_marked_users'],
            'placeholder_users': stats['placeholder_users'],
            'items_in_kg': stats['items_in_kg'],
            'items_not_in_kg': stats['items_not_in_kg'],
            'kg_coverage': stats['items_in_kg'] / total_items * 100 if total_items > 0 else 0,
            'computation_time': elapsed_time,
            'users_needing_llm_count': len(self.users_needing_llm),
        })

    def _print_scoring_report(self, stats):
        """打印评分报告"""
        print("\n" + "=" * 60)
        print("📊 兴趣分数计算报告")
        print("=" * 60)
        print(f"总用户数: {self.stats['total_users']}")
        print(f"总候选记录: {self.stats.get('total_candidates', 'N/A')}")
        print(f"计算耗时: {self.stats['computation_time']:.2f} 秒")

        print(f"\n用户分类:")
        print(f"  ✅ KG完全计算: {stats['kg_scored_users']}")
        print(f"  ⚠️ KG+LLM增强标记: {stats['llm_marked_users']}")
        print(f"  🔄 需要LLM处理(低交互): {stats['placeholder_users']}")

        print(f"\n商品覆盖:")
        print(f"  在KG中: {stats['items_in_kg']}")
        print(f"  不在KG中: {stats['items_not_in_kg']}")
        print(f"  KG覆盖率: {self.stats['kg_coverage']:.2f}%")

        print(f"\n处理策略分布:")
        total_users = self.stats['total_users']
        if total_users > 0:
            print(f"  完整KG: {stats['kg_scored_users'] / total_users * 100:.1f}%")
            print(f"  KG+LLM增强: {stats['llm_marked_users'] / total_users * 100:.1f}%")
            print(f"  纯LLM(低交互): {stats['placeholder_users'] / total_users * 100:.1f}%")

        print(f"\n需要LLM处理的用户总数: {len(self.users_needing_llm)}")

    def _save_llm_user_list(self):
        """保存需要LLM处理的用户列表"""
        output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if not self.users_needing_llm:
            print("\n✅ 所有用户都获得了足够的KG分数，无需LLM增强")
            return

        # 保存用户列表
        llm_users_file = os.path.join(output_dir, 'users_needing_llm.txt')
        with open(llm_users_file, 'w') as f:
            for user_id in sorted(self.users_needing_llm):
                interaction_count = int(self.user_interaction_counts.get(user_id, 0))  # 转换为Python int
                is_low = interaction_count < self.params['min_interactions_for_kg']
                user_type = "LOW_INTERACTION" if is_low else "NEEDS_ENHANCEMENT"
                f.write(f"{user_id}\t{interaction_count}\t{user_type}\n")

        print(f"\n📝 需要LLM处理的用户列表已保存: {llm_users_file}")
        print(f"   共 {len(self.users_needing_llm)} 个用户")

        low_count = sum(1 for uid in self.users_needing_llm
                        if int(self.user_interaction_counts.get(uid, 0)) < self.params['min_interactions_for_kg'])
        enhancement_count = len(self.users_needing_llm) - low_count
        print(f"   低交互用户: {low_count}")
        print(f"   需要增强的高交互用户: {enhancement_count}")

        # 保存详细数据 - 修复JSON序列化问题
        llm_data = {}
        for user_id in self.users_needing_llm:
            user_llm_items = []
            for item_data in self.user_candidates[user_id]:
                item_id = item_data['item_id']
                kg_score = self.interest_scores.get((user_id, item_id), -10)
                if kg_score == -10 or kg_score == -5:
                    # 确保所有值都是Python原生类型
                    user_llm_items.append({
                        'item_id': str(item_id),  # 转为字符串
                        'asin': str(item_data.get('asin', '')),
                        'title': str(item_data.get('title', '')),
                        'kg_score': int(kg_score) if isinstance(kg_score, (np.integer,)) else float(kg_score),
                        'score_type': 'placeholder' if kg_score == -10 else 'needs_enhancement'
                    })

            if user_llm_items:
                llm_data[str(user_id)] = {  # 确保user_id是字符串
                    'interaction_count': int(self.user_interaction_counts.get(user_id, 0)),  # 转换为Python int
                    'items_needing_llm': user_llm_items
                }

        llm_scores_file = os.path.join(output_dir, 'llm_enhancement_data.json')

        # 自定义JSON编码器处理numpy类型
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(llm_scores_file, 'w', encoding='utf-8') as f:
            json.dump(llm_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        print(f"📝 LLM增强数据已保存: {llm_scores_file}")

    # ==================== 结果保存方法 ====================

    def save_results(self):
        """保存结果到新文件"""
        print("\n" + "=" * 50)
        print("保存结果")
        print("=" * 50)

        os.makedirs(self.output_dir, exist_ok=True)

        try:
            results = []

            for (user_id, item_id), final_score in tqdm(self.final_scores.items(), desc="提取结果"):
                kg_score = self.interest_scores.get((user_id, item_id), 0.0)

                asin = ''
                title = ''
                description = ''
                pctr_score = 0.0
                is_positive = 0

                if user_id in self.user_candidates:
                    for item_data in self.user_candidates[user_id]:
                        if item_data['item_id'] == item_id:
                            pctr_score = item_data.get('pctr_score', 0.0)
                            asin = item_data.get('asin', '')
                            title = item_data.get('title', '')
                            description = item_data.get('description', '')
                            is_positive = item_data.get('is_positive', 0)
                            break

                # 确保所有数值都是Python原生类型
                results.append({
                    'user_id': str(user_id),
                    'item_id': str(item_id),
                    'asin': str(asin),
                    'title': str(title),
                    'description': str(description),
                    'is_positive': int(is_positive) if isinstance(is_positive, (np.integer,)) else is_positive,
                    'original_pctr_score': float(pctr_score) if isinstance(pctr_score,
                                                                           (np.floating, np.integer)) else pctr_score,
                    'kg_interest_score': float(kg_score) if isinstance(kg_score,
                                                                       (np.floating, np.integer)) else kg_score,
                    'final_combined_score': float(final_score) if isinstance(final_score,
                                                                             (np.floating, np.integer)) else final_score
                })

            if not results:
                print("❌ 没有结果数据可以保存")
                return

            results_df = pd.DataFrame(results)
            del results
            collect()

            results_df = results_df.sort_values(
                ['user_id', 'final_combined_score'],
                ascending=[True, False]
            )

            # 清理文本字段
            for col in ['title', 'description']:
                if col in results_df.columns:
                    results_df[col] = results_df[col].astype(str).apply(
                        lambda x: x.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                    )

            output_file = os.path.join(self.output_dir, 'reranked_predictions_with_kg.csv')
            results_df.to_csv(
                output_file,
                index=False,
                encoding='utf-8-sig',
                sep=',',
                quoting=1
            )

            print(f"✅ 结果已保存到: {output_file}")
            print(f"   记录数: {len(results_df)}")
            print(f"   列名: {list(results_df.columns)}")

            # 保存统计信息 - 修复numpy类型问题
            stats_file = os.path.join(self.output_dir, 'scoring_stats.json')

            def convert_to_native(obj):
                """递归转换numpy类型为Python原生类型"""
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(item) for item in obj]
                return obj

            stats_native = convert_to_native(self.stats)

            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_native, f, ensure_ascii=False, indent=2)

            print(f"✅ 统计信息已保存到: {stats_file}")

            # 打印KG分数分布
            print(f"\nKG分数分布:")
            kg_scores_list = [float(s) for s in self.interest_scores.values() if s not in [-10, -5]]
            if kg_scores_list:
                print(f"  最小值: {min(kg_scores_list):.8f}")
                print(f"  最大值: {max(kg_scores_list):.8f}")
                print(f"  平均值: {np.mean(kg_scores_list):.8f}")
                print(f"  中位数: {np.median(kg_scores_list):.8f}")

            # 打印特殊分数统计
            neg10_count = sum(1 for s in self.interest_scores.values() if s == -10)
            neg5_count = sum(1 for s in self.interest_scores.values() if s == -5)
            zero_count = sum(1 for s in self.interest_scores.values() if s == 0)
            positive_count = sum(1 for s in self.interest_scores.values() if s > 0)
            total_scores = len(self.interest_scores)

            print(f"\n分数标记分布:")
            print(f"  占位符(-10): {neg10_count} ({neg10_count / total_scores * 100:.1f}%)")
            print(f"  LLM增强(-5): {neg5_count} ({neg5_count / total_scores * 100:.1f}%)")
            print(f"  零分(0): {zero_count} ({zero_count / total_scores * 100:.1f}%)")
            print(f"  正分(>0): {positive_count} ({positive_count / total_scores * 100:.1f}%)")

        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
            import traceback
            traceback.print_exc()

    # ==================== 流水线方法 ====================

    def run_full_pipeline(self):
        """运行完整的兴趣分数计算流水线"""
        print("开始计算候选池商品兴趣分数...")
        print("=" * 60)

        try:
            self.load_all_data()

            if not self.user_candidates:
                print("❌ 没有候选池数据，无法继续")
                return

            self.calculate_interest_scores()
            self.save_results()

            print("\n" + "=" * 60)
            print("✅ 兴趣分数计算完成！")
            print("=" * 60)

        except Exception as e:
            print(f"❌ 流程执行失败: {e}")
            import traceback
            traceback.print_exc()


# ==================== 使用示例 ====================
if __name__ == "__main__":
    scorer = KGInterestScorer(
        preprocessed_dir='./kg_preprocessed',
        transfer_matrix_dir='./kg_transfer_matrix',
        personalized_vectors_dir='./kg_personalized_vectors',
        candidate_file='val_predictions_with_metadata.csv',
        output_dir='./kg_interest_scores'
    )

    scorer.run_full_pipeline()