# kg_personalized_restart.py
import pickle
import numpy as np
import pandas as pd
import os
import warnings
from collections import defaultdict, Counter
from tqdm import tqdm
import scipy.sparse as sp
import json

warnings.filterwarnings('ignore')


class PersonalizedRestartBuilder:
    """
    第三阶段：个性化重启向量构建
    为每个用户构建个性化的随机游走重启向量
    """

    def __init__(self,
                 preprocessed_dir='./kg_preprocessed',
                 transfer_matrix_dir='./kg_transfer_matrix',
                 output_dir='./kg_personalized_vectors'):
        """
        初始化个性化重启向量构建器

        Args:
            preprocessed_dir: 第一阶段预处理结果目录
            transfer_matrix_dir: 第二阶段转移矩阵目录
            output_dir: 输出目录
        """
        self.preprocessed_dir = preprocessed_dir
        self.transfer_matrix_dir = transfer_matrix_dir
        self.output_dir = output_dir

        # 数据存储
        self.graph = None  # 图结构
        self.transfer_matrix = None  # 转移矩阵
        self.node_index = None  # 节点ID到矩阵索引映射
        self.index_node = None  # 矩阵索引到节点ID映射
        self.entity_id_to_info = None  # 实体信息
        self.relation_id_to_name = None  # 关系映射

        # 用户数据
        self.user_history = defaultdict(list)  # 用户历史交互
        self.user_interaction_strength = defaultdict(lambda: defaultdict(int))  # 用户对商品的交互强度

        # 重启向量
        self.user_restart_vectors = {}  # 用户重启向量（稀疏格式）
        self.global_item_popularity = None  # 全局商品流行度

        # 统计信息
        self.stats = {
            'num_users': 0,
            'users_with_history': 0,
            'total_interactions': 0,
            'avg_interactions_per_user': 0,
            'user_interaction_distribution': {},
            'vector_sparsity': {}
        }

    def load_necessary_data(self):
        """
        加载必要的数据
        """
        print("=" * 50)
        print("加载必要数据")
        print("=" * 50)

        try:
            # 1. 加载图结构
            print("1. 加载图结构...")
            graph_path = os.path.join(self.preprocessed_dir, 'graph_adjacency.pkl')
            with open(graph_path, 'rb') as f:
                self.graph = pickle.load(f)
            print(f"   ✅ 图结构加载完成，节点数: {len(self.graph)}")

            # 2. 加载实体映射
            print("2. 加载实体映射...")
            mappings_path = os.path.join(self.preprocessed_dir, 'entity_mappings.pkl')
            with open(mappings_path, 'rb') as f:
                mappings = pickle.load(f)
                self.entity_id_to_info = mappings['entity_id_to_info']
                self.relation_id_to_name = mappings['relation_id_to_name']
            print(f"   ✅ 实体映射加载完成，实体数: {len(self.entity_id_to_info)}")

            # 3. 加载转移矩阵和节点索引
            print("3. 加载转移矩阵...")
            matrix_path = os.path.join(self.transfer_matrix_dir, 'transfer_matrix.npz')
            if not os.path.exists(matrix_path):
                # 尝试其他可能的路径
                matrix_path = os.path.join(self.transfer_matrix_dir, 'transfer_matrix.npz')

            self.transfer_matrix = sp.load_npz(matrix_path)
            print(f"   ✅ 转移矩阵加载完成，形状: {self.transfer_matrix.shape}")

            # 4. 加载节点索引
            print("4. 加载节点索引...")
            indices_path = os.path.join(self.transfer_matrix_dir, 'node_indices.pkl')
            with open(indices_path, 'rb') as f:
                index_data = pickle.load(f)
                self.node_index = index_data['node_index']
                self.index_node = index_data['index_node']
            print(f"   ✅ 节点索引加载完成，节点数: {len(self.node_index)}")

            # 5. 加载训练三元组（用户-商品交互）
            print("5. 加载用户历史交互...")
            self._load_user_interactions()

            print("\n✅ 所有必要数据加载完成！")

        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _load_user_interactions(self):
        """
        步骤8：用户历史交互分析
        从训练三元组中提取用户的历史交互记录
        """
        print("分析用户历史交互...")

        # 尝试从多个位置加载训练三元组
        possible_paths = [
            os.path.join(self.preprocessed_dir, 'train_triples.csv'),
            './kg_train_triples.txt',
            '../kg_train_triples.txt',
            'D:/graduation_recsys/data/KG-related_Files/kg_train_triples.txt'
        ]

        train_triples = None
        for train_path in possible_paths:
            if os.path.exists(train_path):
                print(f"  从 {train_path} 加载...")
                try:
                    if train_path.endswith('.csv'):
                        train_triples = pd.read_csv(train_path)
                    else:
                        train_triples = pd.read_csv(train_path, sep='\t',
                                                    header=None, names=['head', 'tail', 'relation'])
                    print(f"  加载成功，三元组数: {len(train_triples)}")
                    break
                except Exception as e:
                    print(f"  加载失败: {e}")
                    continue

        if train_triples is None:
            raise FileNotFoundError("无法找到训练三元组文件")

        # 提取用户历史交互
        user_interaction_counts = Counter()

        print("提取用户历史交互记录...")
        for idx, row in tqdm(train_triples.iterrows(), total=len(train_triples)):
            user_id = str(row['head'])
            item_id = str(row['tail'])

            # 添加到用户历史
            self.user_history[user_id].append(item_id)

            # 记录交互强度（这里简单的使用交互次数）
            self.user_interaction_strength[user_id][item_id] += 1

            user_interaction_counts[user_id] += 1

        # 统计信息
        self.stats['num_users'] = len(self.user_history)
        self.stats['total_interactions'] = len(train_triples)
        self.stats['users_with_history'] = len([u for u in self.user_history if len(self.user_history[u]) > 0])

        if self.user_history:
            self.stats['avg_interactions_per_user'] = self.stats['total_interactions'] / len(self.user_history)

        # 交互次数分布
        interaction_counts = list(user_interaction_counts.values())
        if interaction_counts:
            self.stats['user_interaction_distribution'] = {
                'min': min(interaction_counts),
                'max': max(interaction_counts),
                'mean': np.mean(interaction_counts),
                'median': np.median(interaction_counts),
                'std': np.std(interaction_counts)
            }

        print(f"  总用户数: {self.stats['num_users']}")
        print(f"  有历史交互的用户数: {self.stats['users_with_history']}")
        print(f"  总交互次数: {self.stats['total_interactions']}")
        print(f"  平均每个用户交互数: {self.stats['avg_interactions_per_user']:.2f}")

        # 计算全局商品流行度
        self._compute_global_item_popularity(train_triples)

    def _compute_global_item_popularity(self, train_triples):
        """
        计算全局商品流行度
        """
        print("计算全局商品流行度...")
        item_popularity = Counter()

        for _, row in train_triples.iterrows():
            item_id = str(row['tail'])
            item_popularity[item_id] += 1

        self.global_item_popularity = dict(item_popularity)

        # 统计流行度分布
        if item_popularity:
            pop_values = list(item_popularity.values())
            self.stats['item_popularity_distribution'] = {
                'total_items': len(item_popularity),
                'min_popularity': min(pop_values),
                'max_popularity': max(pop_values),
                'mean_popularity': np.mean(pop_values),
                'median_popularity': np.median(pop_values)
            }

            print(f"  有交互的商品数: {len(item_popularity)}")
            print(f"  最高流行度: {max(pop_values)} 次")
            print(f"  平均流行度: {np.mean(pop_values):.2f} 次")

    def build_restart_vectors(self):
        """
        步骤9：构建重启向量
        为每个用户构建个性化的重启概率向量
        """
        print("\n" + "=" * 50)
        print("步骤9：构建个性化重启向量")
        print("=" * 50)

        # 策略选择：可以根据用户交互丰富程度选择不同策略
        print("构建重启向量策略:")
        print("  1. 交互丰富的用户（>20次交互）：使用加权重启向量")
        print("  2. 交互适中的用户（5-20次）：使用简单重启向量")
        print("  3. 交互较少的用户（<5次）：使用全局流行度辅助")

        total_users = len(self.user_history)
        rich_users = 0
        medium_users = 0
        sparse_users = 0

        # 遍历所有用户
        print(f"\n为 {total_users} 个用户构建重启向量...")
        for user_idx, (user_id, item_list) in enumerate(tqdm(self.user_history.items(),
                                                             total=total_users,
                                                             desc="处理用户")):

            interaction_count = len(item_list)

            if interaction_count >= 20:
                # 丰富交互用户：使用加权重启向量
                restart_vector = self._build_weighted_restart_vector(user_id, item_list)
                rich_users += 1
                strategy = "rich"

            elif interaction_count >= 5:
                # 中等交互用户：使用简单重启向量
                restart_vector = self._build_simple_restart_vector(user_id, item_list)
                medium_users += 1
                strategy = "medium"

            else:
                # 稀疏交互用户：使用全局流行度辅助
                restart_vector = self._build_sparse_user_restart_vector(user_id, item_list)
                sparse_users += 1
                strategy = "sparse"

            # 保存重启向量（稀疏格式）
            self.user_restart_vectors[user_id] = {
                'vector': restart_vector,
                'strategy': strategy,
                'interaction_count': interaction_count
            }

        # 统计信息
        print(f"\n重启向量构建统计:")
        print(f"  丰富交互用户（≥20次）: {rich_users} ({rich_users / total_users * 100:.1f}%)")
        print(f"  中等交互用户（5-20次）: {medium_users} ({medium_users / total_users * 100:.1f}%)")
        print(f"  稀疏交互用户（<5次）: {sparse_users} ({sparse_users / total_users * 100:.1f}%)")

        # 分析向量稀疏性
        self._analyze_vector_sparsity()

        print("\n✅ 个性化重启向量构建完成！")

    def _build_weighted_restart_vector(self, user_id, item_list):
        """
        为交互丰富的用户构建加权重启向量
        考虑交互频次、时间衰减（如果有时间信息）、商品流行度等因素
        """
        # 获取用户的交互强度
        interaction_strengths = self.user_interaction_strength[user_id]

        # 计算每个交互商品的权重
        weights = {}
        total_weight = 0

        for item_id in item_list:
            # 基础权重 = 交互频次
            base_weight = interaction_strengths.get(item_id, 1)

            # 反流行度调整：降低过于热门商品的影响，鼓励个性化
            popularity = self.global_item_popularity.get(item_id, 1)
            popularity_factor = 1.0 / (1 + np.log1p(popularity))  # 对数衰减

            # 最终权重
            weight = base_weight * popularity_factor
            weights[item_id] = weight
            total_weight += weight

        # 归一化并构建稀疏向量
        return self._create_sparse_vector(weights, total_weight)

    def _build_simple_restart_vector(self, user_id, item_list):
        """
        为中等交互用户构建简单重启向量
        平均分配权重，适当考虑流行度
        """
        weights = {}
        total_weight = 0

        for item_id in item_list:
            # 基础权重为1
            base_weight = 1.0

            # 轻微的反流行度调整
            popularity = self.global_item_popularity.get(item_id, 1)
            popularity_factor = 1.0 / (1 + 0.5 * np.log1p(popularity))

            weight = base_weight * popularity_factor
            weights[item_id] = weight
            total_weight += weight

        return self._create_sparse_vector(weights, total_weight)

    def _build_sparse_user_restart_vector(self, user_id, item_list):
        """
        为稀疏交互用户构建重启向量
        结合个人历史和全局流行度
        """
        weights = {}
        total_weight = 0

        if len(item_list) > 0:
            # 如果有历史交互，使用个人历史
            for item_id in item_list:
                weights[item_id] = 1.0
                total_weight += 1.0
        else:
            # 如果没有历史交互，使用全局热门商品（备用策略）
            # 这里可以根据业务需求调整
            print(f"  ⚠️ 用户 {user_id} 无历史交互，使用备用策略")

            # 可以选择不创建重启向量，或者在推荐时使用其他策略
            # 这里返回空向量
            return self._create_sparse_vector({}, 0)

        return self._create_sparse_vector(weights, total_weight)

    def _create_sparse_vector(self, weights, total_weight):
        """
        创建稀疏格式的重启向量
        """
        if total_weight == 0:
            # 返回全零向量
            return sp.csr_matrix((1, len(self.node_index)))

        # 构建稀疏向量的数据
        indices = []
        data = []

        for item_id, weight in weights.items():
            if item_id in self.node_index:
                idx = self.node_index[item_id]
                indices.append(idx)
                # 归一化概率
                probability = weight / total_weight
                data.append(probability)

        # 创建稀疏向量
        if indices:
            vector = sp.csr_matrix((data, ([0] * len(indices), indices)),
                                   shape=(1, len(self.node_index)))
        else:
            vector = sp.csr_matrix((1, len(self.node_index)))

        return vector

    def _analyze_vector_sparsity(self):
        """
        分析重启向量的稀疏性
        """
        print("\n分析重启向量稀疏性...")

        sparsity_stats = {
            'zero_vectors': 0,
            'non_zero_counts': [],
            'avg_non_zero': 0,
            'max_non_zero': 0,
            'min_non_zero': float('inf')
        }

        for user_id, vector_data in self.user_restart_vectors.items():
            vector = vector_data['vector']
            nnz = vector.nnz  # 非零元素数量

            if nnz == 0:
                sparsity_stats['zero_vectors'] += 1
            else:
                sparsity_stats['non_zero_counts'].append(nnz)
                sparsity_stats['max_non_zero'] = max(sparsity_stats['max_non_zero'], nnz)
                sparsity_stats['min_non_zero'] = min(sparsity_stats['min_non_zero'], nnz)

        # 计算平均值
        if sparsity_stats['non_zero_counts']:
            sparsity_stats['avg_non_zero'] = np.mean(sparsity_stats['non_zero_counts'])

        self.stats['vector_sparsity'] = sparsity_stats

        print(f"  零向量用户数: {sparsity_stats['zero_vectors']}")
        if sparsity_stats['non_zero_counts']:
            print(f"  平均非零元素数: {sparsity_stats['avg_non_zero']:.2f}")
            print(f"  最少非零元素: {sparsity_stats['min_non_zero']}")
            print(f"  最多非零元素: {sparsity_stats['max_non_zero']}")

    def optimize_for_rich_users(self, min_interactions=10):
        """
        针对丰富交互用户的优化策略
        步骤9补充：为交互丰富的用户构建更精细的重启向量
        """
        print("\n" + "=" * 50)
        print("为丰富交互用户进行优化")
        print("=" * 50)

        rich_users = [uid for uid in self.user_history
                      if len(self.user_history[uid]) >= min_interactions]

        print(f"识别到 {len(rich_users)} 个丰富交互用户（≥{min_interactions}次交互）")

        if not rich_users:
            print("没有满足条件的丰富交互用户")
            return

        print("为丰富交互用户构建优化重启向量...")
        optimized_count = 0

        for user_id in tqdm(rich_users, desc="优化丰富用户"):
            item_list = self.user_history[user_id]

            # 使用更复杂的优化策略
            optimized_vector = self._optimize_rich_user_vector(user_id, item_list)

            # 更新重启向量
            self.user_restart_vectors[user_id]['vector'] = optimized_vector
            self.user_restart_vectors[user_id]['strategy'] = 'optimized_rich'

            optimized_count += 1

        print(f"✅ 为 {optimized_count} 个丰富交互用户进行了优化")

    def _optimize_rich_user_vector(self, user_id, item_list):
        """
        为丰富交互用户优化重启向量
        考虑更多因素：品类分布、品牌偏好、交互时间（如果有）等
        """
        # 这里可以添加更复杂的逻辑
        # 例如：分析用户的品类偏好、品牌偏好等

        # 当前先使用加权版本
        return self._build_weighted_restart_vector(user_id, item_list)

    def validate_restart_vectors(self):
        """
        验证重启向量的有效性
        """
        print("\n" + "=" * 50)
        print("验证重启向量")
        print("=" * 50)

        validation_results = {
            'valid_vectors': 0,
            'invalid_vectors': 0,
            'sum_not_one': [],
            'negative_values': 0
        }

        for user_id, vector_data in self.user_restart_vectors.items():
            vector = vector_data['vector']

            # 检查向量是否有效
            if vector.shape[1] != len(self.node_index):
                print(f"  ❌ 用户 {user_id}: 向量维度不匹配")
                validation_results['invalid_vectors'] += 1
                continue

            # 检查概率和是否为1（允许微小误差）
            vector_sum = vector.sum()
            if vector_sum > 0:  # 非零向量
                if abs(vector_sum - 1.0) > 0.01:
                    validation_results['sum_not_one'].append((user_id, vector_sum))

            # 检查是否有负值
            if (vector.data < 0).any():
                validation_results['negative_values'] += 1

            validation_results['valid_vectors'] += 1

        # 打印验证结果
        print(f"有效向量数: {validation_results['valid_vectors']}")
        print(f"无效向量数: {validation_results['invalid_vectors']}")

        if validation_results['sum_not_one']:
            print(f"概率和不为1的用户数: {len(validation_results['sum_not_one'])}")
            # 打印前几个
            for user_id, vec_sum in validation_results['sum_not_one'][:3]:
                print(f"  用户 {user_id}: 概率和 = {vec_sum:.4f}")

        if validation_results['negative_values'] > 0:
            print(f"包含负值的向量数: {validation_results['negative_values']}")

        # 建议
        if validation_results['invalid_vectors'] == 0:
            print("\n✅ 重启向量验证通过！")
        else:
            print("\n⚠️  重启向量验证发现问题，建议检查")

    def save_personalized_vectors(self):
        """
        保存个性化重启向量和相关数据
        """
        print("\n" + "=" * 50)
        print("保存个性化重启向量")
        print("=" * 50)

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        try:
            # 1. 保存重启向量（分批保存，避免内存问题）
            print("1. 保存重启向量...")
            self._save_vectors_in_batches()

            # 2. 保存用户历史
            print("2. 保存用户历史...")
            user_history_path = os.path.join(self.output_dir, 'user_history.pkl')
            with open(user_history_path, 'wb') as f:
                pickle.dump(dict(self.user_history), f, protocol=pickle.HIGHEST_PROTOCOL)

            # 3. 保存交互强度
            print("3. 保存交互强度...")
            interaction_strength_path = os.path.join(self.output_dir, 'interaction_strength.pkl')
            with open(interaction_strength_path, 'wb') as f:
                # 转换为普通字典
                strength_dict = {uid: dict(strength) for uid, strength in self.user_interaction_strength.items()}
                pickle.dump(strength_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 4. 保存统计信息
            print("4. 保存统计信息...")
            stats_path = os.path.join(self.output_dir, 'personalization_stats.pkl')
            with open(stats_path, 'wb') as f:
                pickle.dump(self.stats, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 5. 保存全局商品流行度
            print("5. 保存全局商品流行度...")
            popularity_path = os.path.join(self.output_dir, 'global_popularity.pkl')
            with open(popularity_path, 'wb') as f:
                pickle.dump(self.global_item_popularity, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 6. 生成报告
            print("6. 生成报告...")
            self._generate_personalization_report()

            print(f"\n✅ 所有数据已保存到: {self.output_dir}")

        except Exception as e:
            print(f"❌ 保存失败: {e}")
            import traceback
            traceback.print_exc()

    def _save_vectors_in_batches(self, batch_size=1000):
        """
        分批保存重启向量，避免内存问题
        """
        user_ids = list(self.user_restart_vectors.keys())
        num_batches = (len(user_ids) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(user_ids))
            batch_users = user_ids[start_idx:end_idx]

            # 提取批处理数据
            batch_data = {}
            for user_id in batch_users:
                vector_data = self.user_restart_vectors[user_id]
                # 转换为可序列化的格式
                batch_data[user_id] = {
                    'strategy': vector_data['strategy'],
                    'interaction_count': vector_data['interaction_count'],
                    'vector_data': vector_data['vector'].data,
                    'vector_indices': vector_data['vector'].indices,
                    'vector_shape': vector_data['vector'].shape
                }

            # 保存批处理文件
            batch_path = os.path.join(self.output_dir, f'restart_vectors_batch_{batch_idx}.pkl')
            with open(batch_path, 'wb') as f:
                pickle.dump(batch_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"  批次 {batch_idx + 1}/{num_batches}: {len(batch_users)} 个用户")

        # 保存批次索引
        batches_info = {
            'num_batches': num_batches,
            'batch_size': batch_size,
            'total_users': len(user_ids)
        }
        batches_path = os.path.join(self.output_dir, 'batches_info.pkl')
        with open(batches_path, 'wb') as f:
            pickle.dump(batches_info, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _generate_personalization_report(self):
        """
        生成个性化重启向量报告
        """
        report_lines = [
            "=" * 60,
            "个性化重启向量构建报告",
            "=" * 60,
            f"生成时间: {pd.Timestamp.now()}",
            "",
            "1. 用户统计:",
            f"   总用户数: {self.stats['num_users']}",
            f"   有历史交互用户数: {self.stats['users_with_history']}",
            f"   总交互次数: {self.stats['total_interactions']}",
            f"   平均交互次数/用户: {self.stats['avg_interactions_per_user']:.2f}",
            "",
            "2. 用户交互分布:",
        ]

        # 添加交互分布
        if 'user_interaction_distribution' in self.stats:
            dist = self.stats['user_interaction_distribution']
            report_lines.extend([
                f"   最少交互: {dist.get('min', 0)}",
                f"   最多交互: {dist.get('max', 0)}",
                f"   平均交互: {dist.get('mean', 0):.2f}",
                f"   中位数交互: {dist.get('median', 0):.2f}",
            ])

        report_lines.extend([
            "",
            "3. 重启向量策略分布:",
        ])

        # 统计策略分布
        strategy_counts = Counter()
        for user_data in self.user_restart_vectors.values():
            strategy_counts[user_data['strategy']] += 1

        for strategy, count in strategy_counts.most_common():
            percentage = count / len(self.user_restart_vectors) * 100
            report_lines.append(f"   {strategy}: {count} 用户 ({percentage:.1f}%)")

        report_lines.extend([
            "",
            "4. 向量稀疏性:",
        ])

        # 添加稀疏性统计
        if 'vector_sparsity' in self.stats:
            sparsity = self.stats['vector_sparsity']
            report_lines.extend([
                f"   零向量用户数: {sparsity.get('zero_vectors', 0)}",
                f"   平均非零元素: {sparsity.get('avg_non_zero', 0):.2f}",
                f"   最少非零元素: {sparsity.get('min_non_zero', 0)}",
                f"   最多非零元素: {sparsity.get('max_non_zero', 0)}",
            ])

        report_lines.extend([
            "",
            "5. 商品流行度:",
        ])

        # 添加商品流行度
        if 'item_popularity_distribution' in self.stats:
            pop_dist = self.stats['item_popularity_distribution']
            report_lines.extend([
                f"   有交互商品数: {pop_dist.get('total_items', 0)}",
                f"   最低流行度: {pop_dist.get('min_popularity', 0)}",
                f"   最高流行度: {pop_dist.get('max_popularity', 0)}",
                f"   平均流行度: {pop_dist.get('mean_popularity', 0):.2f}",
            ])

        report_lines.extend([
            "",
            "=" * 60,
        ])

        # 写入文件
        report_path = os.path.join(self.output_dir, 'personalization_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"  报告已生成: {report_path}")

    def run_full_pipeline(self):
        """
        运行完整的个性化重启向量构建流水线
        """
        print("开始构建个性化重启向量...")
        print("=" * 60)

        # 1. 加载必要数据
        self.load_necessary_data()

        # 2. 构建重启向量
        self.build_restart_vectors()

        # 3. 为丰富交互用户优化
        self.optimize_for_rich_users(min_interactions=10)

        # 4. 验证重启向量
        self.validate_restart_vectors()

        # 5. 保存结果
        self.save_personalized_vectors()

        print("\n" + "=" * 60)
        print("✅ 个性化重启向量构建完成！")
        print("=" * 60)


# 辅助工具：重启向量加载器
class RestartVectorLoader:
    """
    重启向量加载器，用于后续阶段加载使用
    """

    @staticmethod
    def load_vectors(vector_dir='./kg_personalized_vectors'):
        """
        加载重启向量
        """
        vectors = {}

        # 加载批次信息
        batches_info_path = os.path.join(vector_dir, 'batches_info.pkl')
        with open(batches_info_path, 'rb') as f:
            batches_info = pickle.load(f)

        # 加载所有批次
        for batch_idx in range(batches_info['num_batches']):
            batch_path = os.path.join(vector_dir, f'restart_vectors_batch_{batch_idx}.pkl')
            with open(batch_path, 'rb') as f:
                batch_data = pickle.load(f)

            for user_id, user_data in batch_data.items():
                # 重建稀疏向量
                vector = sp.csr_matrix((user_data['vector_data'],
                                        ([0] * len(user_data['vector_indices']), user_data['vector_indices'])),
                                       shape=user_data['vector_shape'])
                vectors[user_id] = {
                    'vector': vector,
                    'strategy': user_data['strategy'],
                    'interaction_count': user_data['interaction_count']
                }

        return vectors


# 使用示例
if __name__ == "__main__":
    # 初始化构建器
    restart_builder = PersonalizedRestartBuilder(
        preprocessed_dir='./kg_preprocessed',
        transfer_matrix_dir='./kg_transfer_matrix',
        output_dir='./kg_personalized_vectors'
    )

    # 运行完整流水线
    restart_builder.run_full_pipeline()

    # 或者分步运行
    # restart_builder.load_necessary_data()
    # restart_builder.build_restart_vectors()
    # restart_builder.optimize_for_rich_users()
    # restart_builder.validate_restart_vectors()
    # restart_builder.save_personalized_vectors()