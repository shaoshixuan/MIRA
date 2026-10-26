import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import scipy.sparse as sp
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')
class KGTransferMatrixBuilder:
    """
    第二阶段：转移概率矩阵预计算
    """

    def __init__(self, preprocessed_dir='./kg_preprocessed'):
        """
        初始化转移矩阵构建器

        Args:
            preprocessed_dir: 第一阶段预处理结果目录
        """
        self.preprocessed_dir = preprocessed_dir

        # 加载第一阶段的结果
        self.graph = None  # 正向邻接表
        self.entity_id_to_info = None  # 实体ID到信息映射
        self.relation_id_to_name = None  # 关系ID到名称映射
        self.entity_name_to_id = None  # 实体名称到ID映射
        self.stats = None  # 统计信息
        self.layer_indices = None  # 分层索引

        # 第二阶段的数据结构
        self.relation_categories = {}  # 关系分类：{relation_id: category}
        self.relation_weights = {}  # 关系权重：{relation_id: weight}

        # 转移矩阵
        self.transfer_matrix = None  # 稀疏转移矩阵
        self.node_index = {}  # 节点ID到矩阵索引的映射
        self.index_node = {}  # 矩阵索引到节点ID的映射

        # 高频路径模式
        self.frequent_path_patterns = []

        # 转移概率矩阵的统计信息
        self.matrix_stats = {
            'num_nodes': 0,
            'matrix_density': 0.0,
            'category_distribution': {},
            'weight_distribution': {}
        }

    def load_preprocessed_data(self):
        """
        加载第一阶段预处理的数据
        """
        print("=" * 50)
        print("加载第一阶段预处理数据")
        print("=" * 50)

        try:
            # 加载图结构
            print("加载图结构...")
            print(os.path.join(self.preprocessed_dir, 'graph_adjacency.pkl'))
            with open(os.path.join(self.preprocessed_dir, 'graph_adjacency.pkl'), 'rb') as f:
                self.graph = pickle.load(f)

            # 加载实体映射
            print("加载实体映射...")
            with open(os.path.join(self.preprocessed_dir, 'entity_mappings.pkl'), 'rb') as f:
                mappings = pickle.load(f)
                self.entity_id_to_info = mappings['entity_id_to_info']
                self.entity_name_to_id = mappings['entity_name_to_id']
                self.relation_id_to_name = mappings['relation_id_to_name']
                self.relation_name_to_id = mappings['relation_name_to_id']

            # 加载统计信息
            print("加载统计信息...")
            with open(os.path.join(self.preprocessed_dir, 'statistics.pkl'), 'rb') as f:
                self.stats = pickle.load(f)

            # 加载分层索引
            print("加载分层索引...")
            with open(os.path.join(self.preprocessed_dir, 'layer_indices.pkl'), 'rb') as f:
                layer_data = pickle.load(f)
                self.layer_indices = {
                    'core': set(layer_data['core']),
                    'extended': set(layer_data['extended']),
                    'exploration': set(layer_data['exploration'])
                }

            print("数据加载完成！")
            print(f"  节点总数: {len(self.graph)}")
            print(f"  关系类型数: {len(self.relation_id_to_name)}")

        except FileNotFoundError as e:
            print(f"文件未找到: {e}")
            raise
        except Exception as e:
            print(f"数据加载失败: {e}")
            raise

    def categorize_relations(self):
        """
        步骤5：定义关系类型分类
        将知识图谱中的关系分为四类
        """
        print("\n" + "=" * 50)
        print("步骤5：定义关系类型分类")
        print("=" * 50)

        # 定义关系分类的关键词模式
        relation_patterns = {
            '深化关系': ['belongs_to', 'has_feature', 'produced_by', 'made_of',
                         'contains', 'part_of', 'located_in', 'created_by'],
            '泛化关系': ['subclass_of', 'broader_than', 'parent_of', 'generalization_of',
                         'type_of', 'category_of', 'abstract_of'],
            '关联关系': ['similar_to', 'also_bought', 'also_viewed', 'related_to',
                         'compatible_with', 'alternative_to', 'replacement_for'],
            '创新关系': ['cross_sell', 'complementary', 'bundle_with', 'upgrade_to',
                         'inspires', 'leads_to', 'enables']
        }

        # 统计关系分类结果
        category_counts = Counter()

        print("关系分类结果:")
        for rel_id, rel_name in self.relation_id_to_name.items():
            category = '未知关系'  # 默认分类

            # 根据关系名称匹配分类
            rel_name_lower = rel_name.lower()

            # 检查是否匹配深化关系模式
            for pattern in relation_patterns['深化关系']:
                if pattern in rel_name_lower:
                    category = '深化关系'
                    break

            # 检查泛化关系
            if category == '未知关系':
                for pattern in relation_patterns['泛化关系']:
                    if pattern in rel_name_lower:
                        category = '泛化关系'
                        break

            # 检查关联关系
            if category == '未知关系':
                for pattern in relation_patterns['关联关系']:
                    if pattern in rel_name_lower:
                        category = '关联关系'
                        break

            # 检查创新关系
            if category == '未知关系':
                for pattern in relation_patterns['创新关系']:
                    if pattern in rel_name_lower:
                        category = '创新关系'
                        break

            self.relation_categories[rel_id] = category
            category_counts[category] += 1

            print(f"  {rel_name} ({rel_id}) -> {category}")

        print("\n关系分类统计:")
        for category, count in category_counts.most_common():
            print(f"  {category}: {count}")

        # 保存分类结果
        self.matrix_stats['category_distribution'] = dict(category_counts)

        print("\n 关系分类完成！")

    def assign_relation_weights(self):
        """
        步骤6：设置关系权重
        为每类关系设置转移概率权重
        """
        print("\n" + "=" * 50)
        print("步骤6：设置关系权重")
        print("=" * 50)

        # 定义各类关系的权重（用于探索性随机游走）
        category_weights = {
            '深化关系': 0.3,  # 指向具体属性
            '泛化关系': 0.4,  # 关键探索路径（较高）
            '关联关系': 0.2,  # 同级关联
            '创新关系': 0.1,  # 跨域连接
            '未知关系': 0.15  # 默认权重
        }

        # 应用权重到每个关系
        weight_counts = Counter()

        print("关系权重分配:")
        for rel_id, category in self.relation_categories.items():
            weight = category_weights.get(category, 0.15)
            self.relation_weights[rel_id] = weight
            weight_counts[weight] += 1

            rel_name = self.relation_id_to_name.get(rel_id, rel_id)
            print(f"  {rel_name}: {category} -> 权重 {weight}")

        print("\n权重分布统计:")
        for weight, count in weight_counts.most_common():
            print(f"  权重 {weight}: {count} 个关系")

        # 验证权重设置
        print("\n权重设置验证:")
        print(f"  深化关系权重: {category_weights['深化关系']}")
        print(f"  泛化关系权重: {category_weights['泛化关系']} (探索性最高)")
        print(f"  关联关系权重: {category_weights['关联关系']}")
        print(f"  创新关系权重: {category_weights['创新关系']}")

        # 保存权重分布
        self.matrix_stats['weight_distribution'] = dict(weight_counts)

        print("\n关系权重分配完成！")

    def build_transfer_matrix(self):
        """
        步骤7：构建转移概率矩阵
        创建节点数×节点数的稀疏转移矩阵
        """
        print("\n" + "=" * 50)
        print("步骤7：构建转移概率矩阵")
        print("=" * 50)

        # 7.1 建立节点索引映射
        print("建立节点索引映射...")
        all_nodes = list(self.graph.keys())
        self.num_nodes = len(all_nodes)

        # 创建双向映射
        for i, node in enumerate(all_nodes):
            self.node_index[node] = i
            self.index_node[i] = node

        # 7.2 准备构建稀疏矩阵的数据
        print("准备稀疏矩阵数据...")
        rows = []
        cols = []
        data = []

        # 统计信息
        total_edges = 0
        nodes_without_edges = 0

        # 遍历所有节点构建转移概率
        for node_idx, node_id in tqdm(enumerate(all_nodes),
                                      total=self.num_nodes,
                                      desc="构建转移矩阵"):
            if node_id not in self.graph or not self.graph[node_id]:
                # 没有出边的节点，跳过或自循环
                nodes_without_edges += 1
                continue

            # 收集该节点的所有出边
            out_edges = []
            for relation, neighbors in self.graph[node_id].items():
                relation_weight = self.relation_weights.get(relation, 0.15)
                for neighbor in neighbors:
                    out_edges.append((neighbor, relation_weight))

            if not out_edges:
                nodes_without_edges += 1
                continue

            # 计算总权重用于归一化
            total_weight = sum(weight for _, weight in out_edges)

            # 添加转移概率到稀疏矩阵
            for neighbor, weight in out_edges:
                if neighbor in self.node_index:
                    neighbor_idx = self.node_index[neighbor]
                    prob = weight / total_weight

                    rows.append(node_idx)
                    cols.append(neighbor_idx)
                    data.append(prob)
                    total_edges += 1

        # 7.3 构建稀疏转移矩阵
        print("构建稀疏转移矩阵...")
        self.transfer_matrix = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(self.num_nodes, self.num_nodes)
        )

        # 7.4 矩阵统计
        print("\n转移矩阵统计:")
        print(f"  矩阵形状: {self.transfer_matrix.shape}")
        print(f"  非零元素数量: {self.transfer_matrix.nnz}")
        print(f"  矩阵密度: {self.transfer_matrix.nnz / (self.num_nodes * self.num_nodes):.6f}")
        print(f"  总边数: {total_edges}")
        print(f"  无出边节点数: {nodes_without_edges}")

        # 7.5 验证转移矩阵性质
        print("\n转移矩阵验证:")

        # 检查行和是否为1（概率矩阵性质）
        row_sums = self.transfer_matrix.sum(axis=1)
        valid_rows = np.where(row_sums > 0)[0]

        if len(valid_rows) > 0:
            avg_row_sum = row_sums[valid_rows].mean()
            min_row_sum = row_sums[valid_rows].min()
            max_row_sum = row_sums[valid_rows].max()

            print(f"  有效行数: {len(valid_rows)}")
            print(f"  平均行和: {avg_row_sum:.4f}")
            print(f"  最小行和: {min_row_sum:.4f}")
            print(f"  最大行和: {max_row_sum:.4f}")

            # 检查是否接近1（允许微小误差）
            if abs(avg_row_sum - 1.0) < 0.01:
                print("   转移矩阵行和验证通过")
            else:
                print(f"    行和平均值偏离1.0: {avg_row_sum:.4f}")
        else:
            print("  没有有效行")

        # 更新矩阵统计
        self.matrix_stats.update({
            'num_nodes': self.num_nodes,
            'matrix_density': self.transfer_matrix.nnz / (self.num_nodes * self.num_nodes),
            'total_edges': total_edges,
            'nodes_without_edges': nodes_without_edges
        })

        print("\n 转移概率矩阵构建完成！")

    def precompute_frequent_path_patterns(self):
        """
        预计算高频路径模式
        这些模式可以加速后续的路径查询
        """
        print("\n" + "=" * 50)
        print("预计算高频路径模式")
        print("=" * 50)

        # 8.1 提取高频的三元组模式（2跳路径）
        print("提取高频三元组模式...")
        triple_patterns = self.stats.get('triple_patterns', {})

        if triple_patterns:
            print("前20个高频三元组模式:")
            for pattern, count in Counter(triple_patterns).most_common(20):
                print(f"  {pattern}: {count}次")
                self.frequent_path_patterns.append(pattern)

        # 8.2 提取高频的2跳路径模式
        print("\n提取高频2跳路径模式...")
        two_hop_patterns = Counter()

        # 采样部分节点进行分析
        sample_nodes = list(self.graph.keys())[:1000]  # 采样1000个节点

        for node in tqdm(sample_nodes, desc="分析2跳路径"):
            if node in self.graph:
                # 第一跳
                for rel1, neighbors1 in self.graph[node].items():
                    rel1_name = self.relation_id_to_name.get(rel1, rel1)
                    rel1_category = self.relation_categories.get(rel1, '未知关系')

                    for neighbor in neighbors1[:5]:  # 每个关系采样5个邻居
                        if neighbor in self.graph:
                            # 第二跳
                            for rel2, neighbors2 in self.graph[neighbor].items():
                                rel2_name = self.relation_id_to_name.get(rel2, rel2)
                                rel2_category = self.relation_categories.get(rel2, '未知关系')

                                # 构建路径模式
                                pattern = f"{rel1_category}--{rel2_category}"
                                two_hop_patterns[pattern] += 1

        print("\n高频2跳路径模式:")
        for pattern, count in two_hop_patterns.most_common(15):
            print(f"  {pattern}: {count}次")
            self.frequent_path_patterns.append(f"2hop:{pattern}")

        # 8.3 按关系类别统计转移模式
        print("\n按关系类别的转移模式统计:")
        category_transitions = Counter()

        for node in sample_nodes:
            if node in self.graph:
                for rel, neighbors in self.graph[node].items():
                    rel_category = self.relation_categories.get(rel, '未知关系')

                    for neighbor in neighbors[:3]:
                        if neighbor in self.graph:
                            for rel2, _ in self.graph[neighbor].items():
                                rel2_category = self.relation_categories.get(rel2, '未知关系')
                                transition = f"{rel_category}->{rel2_category}"
                                category_transitions[transition] += 1

        print("常见关系类别转移:")
        for transition, count in category_transitions.most_common(10):
            print(f"  {transition}: {count}次")
            self.frequent_path_patterns.append(f"trans:{transition}")

        print(f"\n 预计算了 {len(self.frequent_path_patterns)} 个高频路径模式")

    def analyze_transfer_matrix_properties(self):
        """
        分析转移矩阵的数学性质
        """
        print("\n" + "=" * 50)
        print("分析转移矩阵性质")
        print("=" * 50)

        if self.transfer_matrix is None:
            print("转移矩阵未构建")
            return

        # 9.1 计算矩阵的稀疏性
        density = self.transfer_matrix.nnz / (self.num_nodes * self.num_nodes)
        print(f"矩阵密度: {density:.6f}")
        print(f"稀疏度: {1 - density:.6f}")

        # 9.2 计算节点的出度和入度分布
        print("\n节点度分布:")
        out_degrees = self.transfer_matrix.getnnz(axis=1)  # 每行的非零元素数
        in_degrees = self.transfer_matrix.getnnz(axis=0)  # 每列的非零元素数

        print(f"平均出度: {out_degrees.mean():.2f}")
        print(f"最大出度: {out_degrees.max()}")
        print(f"最小出度: {out_degrees.min()}")

        print(f"平均入度: {in_degrees.mean():.2f}")
        print(f"最大入度: {in_degrees.max()}")
        print(f"最小入度: {in_degrees.min()}")

        # 9.3 按节点类型分析度分布
        print("\n按实体类型的度分布:")
        type_out_degrees = defaultdict(list)
        type_in_degrees = defaultdict(list)

        for idx in range(self.num_nodes):
            node_id = self.index_node[idx]
            entity_info = self.entity_id_to_info.get(node_id, {})
            entity_type = entity_info.get('type', 'unknown')

            type_out_degrees[entity_type].append(out_degrees[idx])
            type_in_degrees[entity_type].append(in_degrees[idx])

        for entity_type in sorted(type_out_degrees.keys()):
            if type_out_degrees[entity_type]:
                avg_out = np.mean(type_out_degrees[entity_type])
                avg_in = np.mean(type_in_degrees[entity_type])
                print(f"  {entity_type}: 平均出度={avg_out:.2f}, 平均入度={avg_in:.2f}")

        # 9.4 检查矩阵的连通性
        print("\n矩阵连通性分析:")

        # 计算强连通分量数量（近似）
        # 注意：对于大型稀疏矩阵，精确计算可能很耗时
        try:
            # 使用较小的子图进行分析
            if self.num_nodes <= 10000:
                from scipy.sparse.csgraph import connected_components

                # 转换为无向图（对称化）
                symmetric_matrix = self.transfer_matrix + self.transfer_matrix.T
                symmetric_matrix.data = np.ones_like(symmetric_matrix.data)

                n_components, labels = connected_components(
                    symmetric_matrix, directed=False, return_labels=True
                )

                print(f"  连通分量数: {n_components}")

                if n_components > 1:
                    component_sizes = Counter(labels)
                    print(f"  最大连通分量大小: {max(component_sizes.values())}")
                    print(f"  最小连通分量大小: {min(component_sizes.values())}")
            else:
                print("  节点数过多，跳过连通性分析")

        except Exception as e:
            print(f"  连通性分析出错: {e}")

        print("\n转移矩阵性质分析完成！")

    def save_transfer_matrix(self, output_dir='./kg_transfer_matrix'):
        """
        保存转移矩阵和相关数据
        """
        print("\n" + "=" * 50)
        print("保存转移矩阵和相关数据")
        print("=" * 50)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        try:
            # 保存转移矩阵
            print("保存转移矩阵...")
            matrix_path = os.path.join(output_dir, 'transfer_matrix.npz')
            sp.save_npz(matrix_path, self.transfer_matrix)
            print(f"  矩阵已保存: {matrix_path}")

            # 保存节点索引映射
            print("保存节点索引映射...")
            index_data = {
                'node_index': self.node_index,
                'index_node': self.index_node,
                'num_nodes': self.num_nodes
            }
            with open(os.path.join(output_dir, 'node_indices.pkl'), 'wb') as f:
                pickle.dump(index_data, f)

            # 保存关系分类和权重
            print("保存关系分类和权重...")
            relation_data = {
                'relation_categories': self.relation_categories,
                'relation_weights': self.relation_weights,
                'relation_id_to_name': self.relation_id_to_name
            }
            with open(os.path.join(output_dir, 'relation_data.pkl'), 'wb') as f:
                pickle.dump(relation_data, f)

            # 保存高频路径模式
            print("保存高频路径模式...")
            with open(os.path.join(output_dir, 'frequent_paths.pkl'), 'wb') as f:
                pickle.dump(self.frequent_path_patterns, f)

            # 保存矩阵统计信息
            print("保存矩阵统计信息...")
            with open(os.path.join(output_dir, 'matrix_stats.pkl'), 'wb') as f:
                pickle.dump(self.matrix_stats, f)

            # 生成转移矩阵报告
            self.generate_transfer_matrix_report(output_dir)

            print(f"\n所有数据已保存到: {output_dir}")

        except Exception as e:
            print(f"保存失败: {e}")
            raise

    def generate_transfer_matrix_report(self, output_dir):
        """
        生成转移矩阵分析报告
        """
        report_lines = [
            "=" * 60,
            "转移概率矩阵分析报告",
            "=" * 60,
            f"生成时间: {pd.Timestamp.now()}",
            "",
            "1. 矩阵基本信息:",
            f"   节点总数: {self.matrix_stats['num_nodes']}",
            f"   矩阵形状: {self.transfer_matrix.shape}",
            f"   非零元素数: {self.transfer_matrix.nnz}",
            f"   矩阵密度: {self.matrix_stats['matrix_density']:.6f}",
            f"   总边数: {self.matrix_stats.get('total_edges', 0)}",
            "",
            "2. 关系分类分布:",
        ]

        # 添加关系分类
        for category, count in self.matrix_stats['category_distribution'].items():
            report_lines.append(f"   {category}: {count}")

        report_lines.extend([
            "",
            "3. 关系权重设置:",
        ])

        # 添加权重分布
        weight_dist = self.matrix_stats.get('weight_distribution', {})
        for weight, count in sorted(weight_dist.items(), key=lambda x: x[0], reverse=True):
            report_lines.append(f"   权重 {weight}: {count} 个关系")

        report_lines.extend([
            "",
            "4. 节点度数统计:",
            f"   平均出度: {self.transfer_matrix.getnnz(axis=1).mean():.2f}",
            f"   平均入度: {self.transfer_matrix.getnnz(axis=0).mean():.2f}",
            "",
            "5. 高频路径模式 (前10个):",
        ])

        # 添加高频路径
        for i, pattern in enumerate(self.frequent_path_patterns[:10]):
            report_lines.append(f"   {i + 1}. {pattern}")

        report_lines.extend([
            "",
            "=" * 60,
        ])

        # 写入文件
        report_path = os.path.join(output_dir, 'transfer_matrix_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"  报告已生成: {report_path}")

    def run_full_pipeline(self):
        """
        运行完整的转移矩阵构建流水线
        """
        print("开始构建转移概率矩阵...")
        print("=" * 60)

        # 加载预处理数据
        self.load_preprocessed_data()

        # 定义关系分类
        self.categorize_relations()

        # 分配关系权重
        self.assign_relation_weights()

        # 构建转移矩阵
        self.build_transfer_matrix()

        # 预计算高频路径模式
        self.precompute_frequent_path_patterns()

        # 分析矩阵性质
        self.analyze_transfer_matrix_properties()

        # 保存结果
        self.save_transfer_matrix()

        print("\n" + "=" * 60)
        print("转移概率矩阵构建完成！")
        print("=" * 60)


# 使用示例
if __name__ == "__main__":
    # 初始化构建器
    matrix_builder = KGTransferMatrixBuilder(preprocessed_dir='./kg_preprocessed')

    # 运行完整流水线
    matrix_builder.run_full_pipeline()

    # 或者分步运行
    # matrix_builder.load_preprocessed_data()
    # matrix_builder.categorize_relations()
    # matrix_builder.assign_relation_weights()
    # matrix_builder.build_transfer_matrix()
    # matrix_builder.precompute_frequent_path_patterns()
    # matrix_builder.analyze_transfer_matrix_properties()
    # matrix_builder.save_transfer_matrix()