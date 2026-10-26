# import pandas as pd
# import numpy as np
# from collections import defaultdict, Counter
# import pickle
# import os
# from tqdm import tqdm
# import warnings
# from pathlib import Path
#
# warnings.filterwarnings('ignore')
#
#
# class KGGraphBuilder:
#     """
#     知识图谱构建器 - 第一阶段：数据预处理与索引构建
#     """
#
#     def __init__(self,data_dir=None):
#         """
#         初始化知识图谱构建器
#
#         Args:
#             data_dir: 数据文件目录路径
#         """
#         import os
#         from pathlib import Path
#
#         if data_dir is None:
#             # 使用脚本所在目录
#             self.data_dir = Path(os.path.dirname(os.path.abspath(__file__)))
#         else:
#             # 原有的处理逻辑
#             self.data_dir = Path(data_dir)
#
#         print(f"数据目录: {self.data_dir}")
#
#         # 输出目录
#         self.output_dir = self.data_dir / 'kg_preprocessed'
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#
#         # 数据存储结构
#         self.user_entities = None  # 用户实体
#         self.item_entities = None  # 商品实体
#         self.other_entities = None  # 其他实体
#         self.relations = None  # 关系映射
#         self.train_triples = None  # 训练三元组（用户-商品交互）
#         self.other_triples = None  # 其他三元组
#
#         # 图结构
#         self.graph = defaultdict(lambda: defaultdict(list))  # 邻接表: {head: {relation: [tail1, tail2, ...]}}
#         self.reverse_graph = defaultdict(lambda: defaultdict(list))  # 反向邻接表: {tail: {relation: [head1, head2, ...]}}
#
#         # 索引映射
#         self.entity_id_to_info = {}  # 实体ID -> {name, type}
#         self.entity_name_to_id = {}  # 实体名称 -> 实体ID
#         self.relation_id_to_name = {}  # 关系ID -> 关系名称
#         self.relation_name_to_id = {}  # 关系名称 -> 关系ID
#
#         # 统计信息
#         self.stats = {
#             'num_users': 0,
#             'num_items': 0,
#             'num_other_entities': 0,
#             'num_relations': 0,
#             'num_train_triples': 0,
#             'num_other_triples': 0,
#             'item_popularity': Counter(),  # 商品流行度统计
#             'node_degrees': defaultdict(lambda: {'in_degree': 0, 'out_degree': 0}),  # 节点度数
#         }
#
#         # 分层索引
#         self.layer_indices = {
#             'core': set(),  # 核心层：用户历史交互商品及其1跳邻居
#             'extended': set(),  # 扩展层：2-3跳邻居
#             'exploration': set(),  # 探索层：4-5跳邻居
#         }
#
#     def load_data(self):
#         """
#         步骤1：加载并解析知识图谱数据
#         """
#         print("=" * 50)
#         print("步骤1：加载知识图谱数据")
#         print("=" * 50)
#
#         try:
#             # 1.1 加载用户实体
#             print("加载用户实体...")
#             self.user_entities = self._load_entities_file('kg_users.txt', 'user')
#             self.stats['num_users'] = len(self.user_entities)
#             print(f"  加载完成: {self.stats['num_users']} 个用户")
#
#             # 1.2 加载商品实体
#             print("加载商品实体...")
#             self.item_entities = self._load_entities_file('kg_items.txt', 'item')
#             self.stats['num_items'] = len(self.item_entities)
#             print(f"  加载完成: {self.stats['num_items']} 个商品")
#
#             # 1.3 加载其他实体
#             print("加载其他实体...")
#             self.other_entities = self._load_entities_file('kg_entities.txt', 'other')
#             self.stats['num_other_entities'] = len(self.other_entities)
#             print(f"  加载完成: {self.stats['num_other_entities']} 个其他实体")
#
#             # 1.4 加载关系映射
#             print("加载关系映射...")
#             self.relations = pd.read_csv(
#                 os.path.join(self.data_dir, 'kg_relations.txt'),
#                 sep='\t',
#                 header=None,
#                 names=['relation_name', 'relation_id']
#             )
#
#             # 建立关系ID和名称的双向映射
#             for _, row in self.relations.iterrows():
#                 rel_name, rel_id = row['relation_name'], row['relation_id']
#                 self.relation_id_to_name[rel_id] = rel_name
#                 self.relation_name_to_id[rel_name] = rel_id
#
#             self.stats['num_relations'] = len(self.relations)
#             print(f"  加载完成: {self.stats['num_relations']} 种关系")
#
#             # 1.5 加载训练三元组（用户-商品交互）
#             print("加载训练三元组（用户-商品交互）...")
#             self.train_triples = pd.read_csv(
#                 os.path.join(self.data_dir, 'kg_train_triples.txt'),
#                 sep='\t',
#                 header=None,
#                 names=['head', 'tail', 'relation']
#             )
#             self.stats['num_train_triples'] = len(self.train_triples)
#
#             # 统计商品流行度（基于用户交互）
#             for _, row in self.train_triples.iterrows():
#                 item_id = row['tail']  # 假设tail是商品ID
#                 self.stats['item_popularity'][item_id] += 1
#
#             print(f"  加载完成: {self.stats['num_train_triples']} 个交互三元组")
#             print(f"  统计了 {len(self.stats['item_popularity'])} 个商品的流行度")
#
#             # 1.6 加载其他三元组
#             print("加载其他三元组...")
#             self.other_triples = pd.read_csv(
#                 os.path.join(self.data_dir, 'kg_other_triples.txt'),
#                 sep='\t',
#                 header=None,
#                 names=['head', 'tail', 'relation']
#             )
#             self.stats['num_other_triples'] = len(self.other_triples)
#             print(f"  加载完成: {self.stats['num_other_triples']} 个其他三元组")
#
#             print("\n所有数据加载完成！")
#
#         except FileNotFoundError as e:
#             print(f"文件未找到: {e}")
#             raise
#         except Exception as e:
#             print(f"数据加载失败: {e}")
#             raise
#
#     def _load_entities_file(self, filename, entity_type):
#         """
#         加载实体文件
#
#         Args:
#             filename: 文件名
#             entity_type: 实体类型（user/item/other）
#
#         Returns:
#             DataFrame: 实体数据
#         """
#
#         filepath = os.path.join(self.data_dir, filename)
#         df = pd.read_csv(filepath, sep='\t', header=None)
#
#         if entity_type == 'other':
#             # 其他实体的格式: [entity_type]::[entity_name]\t[entity_id]
#             df.columns = ['raw', 'entity_id']
#             # 解析实体类型和名称
#             df['entity_type'] = df['raw'].apply(lambda x: x.split('::')[0])
#             df['entity_name'] = df['raw'].apply(lambda x: x.split('::')[1])
#             df = df[['entity_id', 'entity_type', 'entity_name']]
#         else:
#             # 用户和商品实体的格式: [entity_type]::[entity_name]\t[entity_id]
#             df.columns = ['raw', 'entity_id']
#             # 解析实体类型和名称
#             parts = df['raw'].iloc[0].split('::')
#             entity_prefix = parts[0]
#             df['entity_name'] = df['raw'].apply(lambda x: x.split('::')[1])
#             df['entity_type'] = entity_prefix
#             df = df[['entity_id', 'entity_type', 'entity_name']]
#
#         # 建立实体ID到信息的映射
#         for _, row in df.iterrows():
#             entity_id = row['entity_id']
#             entity_name = row['entity_name']
#             entity_type = row['entity_type']
#
#             self.entity_id_to_info[entity_id] = {
#                 'name': entity_name,
#                 'type': entity_type
#             }
#
#             # 建立名称到ID的映射（注意：名称可能重复，所以使用列表存储）
#             if entity_name not in self.entity_name_to_id:
#                 self.entity_name_to_id[entity_name] = []
#             self.entity_name_to_id[entity_name].append(entity_id)
#
#         return df
#
#     def build_graph_structure(self):
#         """
#         步骤2：构建图存储结构
#         """
#         print("\n" + "=" * 50)
#         print("步骤2：构建图存储结构")
#         print("=" * 50)
#
#         # 2.1 合并所有三元组
#         print("合并所有三元组...")
#         all_triples = pd.concat([self.train_triples, self.other_triples], ignore_index=True)
#         print(f"  总三元组数量: {len(all_triples)}")
#
#         # 2.2 构建邻接表和反向邻接表
#         print("构建邻接表和反向邻接表...")
#         for idx, row in tqdm(all_triples.iterrows(), total=len(all_triples)):
#             head = str(row['head'])
#             tail = str(row['tail'])
#             relation = str(row['relation'])
#
#             # 添加到正向邻接表
#             if relation not in self.graph[head]:
#                 self.graph[head][relation] = []
#             self.graph[head][relation].append(tail)
#
#             # 添加到反向邻接表
#             if relation not in self.reverse_graph[tail]:
#                 self.reverse_graph[tail][relation] = []
#             self.reverse_graph[tail][relation].append(head)
#
#             # 更新节点度数统计
#             self.stats['node_degrees'][head]['out_degree'] += 1
#             self.stats['node_degrees'][tail]['in_degree'] += 1
#
#         # 2.3 验证图构建结果
#         print("\n图结构统计:")
#         print(f"  总节点数: {len(self.graph)}")
#         print(f"  平均出度: {sum(d['out_degree'] for d in self.stats['node_degrees'].values()) / len(self.graph):.2f}")
#         print(f"  平均入度: {sum(d['in_degree'] for d in self.stats['node_degrees'].values()) / len(self.graph):.2f}")
#
#         # 统计不同类型实体的连接情况
#         user_connections = 0
#         item_connections = 0
#         for node_id in self.graph.keys():
#             if node_id in self.entity_id_to_info:
#                 entity_type = self.entity_id_to_info[node_id]['type']
#                 if entity_type == 'user':
#                     user_connections += 1
#                 elif entity_type == 'product':
#                     item_connections += 1
#
#         print(f"  有连接的用户节点: {user_connections}/{self.stats['num_users']}")
#         print(f"  有连接的商品节点: {item_connections}/{self.stats['num_items']}")
#
#         print("\n 图结构构建完成！")
#
#     def analyze_statistics(self):
#         """
#         步骤3：统计分析与元数据提取
#         """
#         print("\n" + "=" * 50)
#         print("步骤3：统计分析与元数据提取")
#         print("=" * 50)
#
#         # 3.1 商品流行度分析
#         print("商品流行度分析:")
#         popularity_counts = list(self.stats['item_popularity'].values())
#         if popularity_counts:
#             print(f"  最多交互的商品: {max(popularity_counts)} 次")
#             print(f"  最少交互的商品: {min(popularity_counts)} 次")
#             print(f"  平均交互次数: {np.mean(popularity_counts):.2f}")
#             print(f"  中位数交互次数: {np.median(popularity_counts):.2f}")
#
#             # 分析流行度分布
#             pop_quartiles = np.percentile(popularity_counts, [25, 50, 75, 95])
#             print(f"  25%分位数: {pop_quartiles[0]:.2f}")
#             print(f"  75%分位数: {pop_quartiles[2]:.2f}")
#             print(f"  95%分位数: {pop_quartiles[3]:.2f}")
#
#         # 3.2 节点度数分析
#         print("\n节点度数分析:")
#         out_degrees = [d['out_degree'] for d in self.stats['node_degrees'].values()]
#         in_degrees = [d['in_degree'] for d in self.stats['node_degrees'].values()]
#
#         print(f"  最大出度: {max(out_degrees) if out_degrees else 0}")
#         print(f"  最大入度: {max(in_degrees) if in_degrees else 0}")
#         print(f"  平均出度: {np.mean(out_degrees):.2f}")
#         print(f"  平均入度: {np.mean(in_degrees):.2f}")
#
#         # 3.3 高频关系模式提取
#         print("\n高频关系模式提取:")
#         relation_counts = Counter()
#         for head in self.graph:
#             for relation in self.graph[head]:
#                 relation_counts[relation] += len(self.graph[head][relation])
#
#         print("  前10个最常见的关系:")
#         for relation, count in relation_counts.most_common(10):
#             rel_name = self.relation_id_to_name.get(relation, relation)
#             print(f"    {rel_name}: {count} 次")
#
#         # 3.4 保存高频三元组模式
#         triple_patterns = Counter()
#         for head in self.graph:
#             for relation in self.graph[head]:
#                 for tail in self.graph[head][relation]:
#                     # 获取实体类型
#                     head_type = self.entity_id_to_info.get(head, {}).get('type', 'unknown')
#                     tail_type = self.entity_id_to_info.get(tail, {}).get('type', 'unknown')
#                     rel_name = self.relation_id_to_name.get(relation, relation)
#
#                     pattern = f"{head_type}--{rel_name}--{tail_type}"
#                     triple_patterns[pattern] += 1
#
#         print("\n  前10个最常见的关系模式:")
#         for pattern, count in triple_patterns.most_common(10):
#             print(f"    {pattern}: {count} 次")
#
#         # 保存统计结果
#         self.stats['relation_counts'] = dict(relation_counts)
#         self.stats['triple_patterns'] = dict(triple_patterns)
#
#         print("\n 统计分析完成！")
#
#     def build_layer_indices(self):
#         """
#         步骤4：构建分层图索引
#         """
#         print("\n" + "=" * 50)
#         print("步骤4：构建分层图索引")
#         print("=" * 50)
#
#         # 4.1 提取用户历史交互商品
#         print("提取用户历史交互商品...")
#         user_history_items = defaultdict(set)
#         for _, row in self.train_triples.iterrows():
#             user_id = str(row['head'])
#             item_id = str(row['tail'])
#             user_history_items[user_id].add(item_id)
#
#         print(f"  涉及用户数: {len(user_history_items)}")
#
#         # 4.2 为每个用户构建分层索引
#         print("\n构建分层索引（按用户）...")
#         total_core = 0
#         total_extended = 0
#         total_exploration = 0
#
#         for user_id, items in tqdm(user_history_items.items(), desc="处理用户"):
#             # 初始化当前用户的分层集合
#             user_core = set(items)  # 核心层：历史交互商品
#
#             # 第一跳邻居（从历史商品出发）
#             first_hop = set()
#             for item in items:
#                 if item in self.graph:
#                     # 获取所有邻居
#                     for relation in self.graph[item]:
#                         first_hop.update(self.graph[item][relation])
#
#             user_core.update(first_hop)
#
#             # 第二跳和第三跳（扩展层）
#             current_layer = first_hop
#             for hop in range(2, 4):  # 第2、3跳
#                 next_layer = set()
#                 for node in current_layer:
#                     if node in self.graph:
#                         for relation in self.graph[node]:
#                             next_layer.update(self.graph[node][relation])
#
#                 # 移除已经在核心层的节点
#                 next_layer = next_layer - user_core
#
#                 # 更新当前层和扩展层
#                 current_layer = next_layer
#                 if hop <= 3:  # 2-3跳属于扩展层
#                     self.layer_indices['extended'].update(next_layer)
#                 else:  # 4+跳属于探索层
#                     self.layer_indices['exploration'].update(next_layer)
#
#             # 第四跳和第五跳（探索层）
#             for hop in range(4, 6):  # 第4、5跳
#                 next_layer = set()
#                 for node in current_layer:
#                     if node in self.graph:
#                         for relation in self.graph[node]:
#                             next_layer.update(self.graph[node][relation])
#
#                 # 移除已经在其他层的节点
#                 next_layer = next_layer - user_core - self.layer_indices['extended']
#
#                 # 更新当前层和探索层
#                 current_layer = next_layer
#                 self.layer_indices['exploration'].update(next_layer)
#
#             # 将核心层添加到全局索引
#             self.layer_indices['core'].update(user_core)
#
#         # 4.3 去重并统计
#         print("\n分层索引统计:")
#         print(f"  核心层节点数: {len(self.layer_indices['core'])}")
#         print(f"  扩展层节点数: {len(self.layer_indices['extended'])}")
#         print(f"  探索层节点数: {len(self.layer_indices['exploration'])}")
#
#         # 计算重叠情况
#         core_extended_overlap = len(self.layer_indices['core'] & self.layer_indices['extended'])
#         core_exploration_overlap = len(self.layer_indices['core'] & self.layer_indices['exploration'])
#         extended_exploration_overlap = len(self.layer_indices['extended'] & self.layer_indices['exploration'])
#
#         print(f"  核心层与扩展层重叠: {core_extended_overlap} 个节点")
#         print(f"  核心层与探索层重叠: {core_exploration_overlap} 个节点")
#         print(f"  扩展层与探索层重叠: {extended_exploration_overlap} 个节点")
#
#         # 4.4 分层节点类型分析
#         print("\n分层节点类型分布:")
#         for layer_name, layer_nodes in self.layer_indices.items():
#             type_counter = Counter()
#             for node in layer_nodes:
#                 if node in self.entity_id_to_info:
#                     entity_type = self.entity_id_to_info[node]['type']
#                     type_counter[entity_type] += 1
#
#             print(f"  {layer_name}层:")
#             for etype, count in type_counter.most_common():
#                 print(f"    {etype}: {count}")
#
#         print("\n分层索引构建完成！")
#
#     # def save_results(self, output_dir='./kg_preprocessed'):
#     #     """
#     #     保存预处理结果
#     #
#     #     Args:
#     #         output_dir: 输出目录
#     #     """
#     #     print("\n" + "=" * 50)
#     #     print("保存预处理结果")
#     #     print("=" * 50)
#     #
#     #     # 创建输出目录
#     #     os.makedirs(output_dir, exist_ok=True)
#     #
#     #     # 保存图结构
#     #     print("保存图结构...")
#     #     with open(os.path.join(output_dir, 'graph_adjacency.pkl'), 'wb') as f:
#     #         pickle.dump(dict(self.graph), f)
#     #
#     #     with open(os.path.join(output_dir, 'graph_reverse.pkl'), 'wb') as f:
#     #         pickle.dump(dict(self.reverse_graph), f)
#     #
#     #     # 保存索引映射
#     #     print("保存索引映射...")
#     #     with open(os.path.join(output_dir, 'entity_mappings.pkl'), 'wb') as f:
#     #         pickle.dump({
#     #             'entity_id_to_info': self.entity_id_to_info,
#     #             'entity_name_to_id': self.entity_name_to_id,
#     #             'relation_id_to_name': self.relation_id_to_name,
#     #             'relation_name_to_id': self.relation_name_to_id,
#     #         }, f)
#     #
#     #     # 保存统计信息
#     #     print("保存统计信息...")
#     #     with open(os.path.join(output_dir, 'statistics.pkl'), 'wb') as f:
#     #         pickle.dump(self.stats, f)
#     #
#     #     # 保存分层索引
#     #     print("保存分层索引...")
#     #     with open(os.path.join(output_dir, 'layer_indices.pkl'), 'wb') as f:
#     #         pickle.dump({
#     #             'core': list(self.layer_indices['core']),
#     #             'extended': list(self.layer_indices['extended']),
#     #             'exploration': list(self.layer_indices['exploration']),
#     #         }, f)
#     #
#     #     # 保存原始数据（可选，用于调试）
#     #     print("保存原始数据...")
#     #     self.train_triples.to_csv(os.path.join(output_dir, 'train_triples.csv'), index=False)
#     #     self.other_triples.to_csv(os.path.join(output_dir, 'other_triples.csv'), index=False)
#     #
#     #     # 生成汇总报告
#     #     self.generate_summary_report(output_dir)
#     #
#     #     print(f"\n✅ 所有结果已保存到: {output_dir}")
#     def save_results(self, filepath=None):
#         """保存处理结果"""
#         if filepath is None:
#             filepath = self.output_dir / "kg_builder_stats.pkl"
#
#         # 创建可序列化的统计信息副本
#         serializable_stats = {}
#
#         for key, value in self.stats.items():
#             try:
#                 # 尝试序列化测试
#                 pickle.dumps(value)
#                 serializable_stats[key] = value
#             except (pickle.PickleError, TypeError, AttributeError) as e:
#                 # 如果是函数，转换为字符串表示
#                 if callable(value):
#                     serializable_stats[key] = f"<function: {key}>"
#                 # 如果是特殊对象，转换为基本类型
#                 elif hasattr(value, '__dict__'):
#                     # 尝试提取可序列化的属性
#                     try:
#                         serializable_stats[key] = {
#                             k: v for k, v in value.__dict__.items()
#                             if not callable(v) and not k.startswith('_')
#                         }
#                     except:
#                         serializable_stats[key] = f"<object: {type(value).__name__}>"
#                 else:
#                     serializable_stats[key] = f"<non-serializable: {type(value).__name__}>"
#                 print(f"警告: 无法序列化统计项 '{key}'，已转换为字符串表示")
#
#         # 保存可序列化的统计信息
#         with open(filepath, 'wb') as f:
#             pickle.dump(serializable_stats, f)
#
#         print(f"统计信息已保存到: {filepath}")
#
#         # 同时保存为JSON（可读性更好）
#         json_path = filepath.with_suffix('.json')
#         try:
#             import json
#             # 转换为JSON可序列化的格式
#             json_stats = {}
#             for key, value in serializable_stats.items():
#                 if isinstance(value, (dict, list, tuple, str, int, float, bool, type(None))):
#                     json_stats[key] = value
#                 else:
#                     json_stats[key] = str(value)
#
#             with open(json_path, 'w', encoding='utf-8') as f:
#                 json.dump(json_stats, f, indent=2, ensure_ascii=False)
#
#             print(f"JSON统计信息已保存到: {json_path}")
#         except Exception as e:
#             print(f"保存JSON失败: {e}")
#
#     def generate_summary_report(self, output_dir):
#         """
#         生成汇总报告
#
#         Args:
#             output_dir: 输出目录
#         """
#         report_lines = [
#             "=" * 60,
#             "知识图谱预处理汇总报告",
#             "=" * 60,
#             f"生成时间: {pd.Timestamp.now()}",
#             "",
#             "1. 数据统计:",
#             f"   用户数量: {self.stats['num_users']}",
#             f"   商品数量: {self.stats['num_items']}",
#             f"   其他实体数量: {self.stats['num_other_entities']}",
#             f"   关系类型数量: {self.stats['num_relations']}",
#             f"   用户-商品交互数量: {self.stats['num_train_triples']}",
#             f"   其他三元组数量: {self.stats['num_other_triples']}",
#             "",
#             "2. 图结构统计:",
#             f"   总节点数: {len(self.graph)}",
#             f"   平均出度: {sum(d['out_degree'] for d in self.stats['node_degrees'].values()) / len(self.graph):.2f}",
#             f"   平均入度: {sum(d['in_degree'] for d in self.stats['node_degrees'].values()) / len(self.graph):.2f}",
#             "",
#             "3. 分层索引统计:",
#             f"   核心层节点数: {len(self.layer_indices['core'])}",
#             f"   扩展层节点数: {len(self.layer_indices['extended'])}",
#             f"   探索层节点数: {len(self.layer_indices['exploration'])}",
#             "",
#             "4. 商品流行度分布:",
#         ]
#
#         # 添加流行度分布
#         pop_values = list(self.stats['item_popularity'].values())
#         if pop_values:
#             report_lines.extend([
#                 f"   最多交互: {max(pop_values)} 次",
#                 f"   最少交互: {min(pop_values)} 次",
#                 f"   平均交互: {np.mean(pop_values):.2f} 次",
#                 f"   中位数交互: {np.median(pop_values):.2f} 次",
#             ])
#
#         report_lines.extend([
#             "",
#             "5. 前5个最常见的关系:",
#         ])
#
#         # 添加最常见的关系
#         relation_counts = self.stats.get('relation_counts', {})
#         if relation_counts:
#             for rel_id, count in Counter(relation_counts).most_common(5):
#                 rel_name = self.relation_id_to_name.get(rel_id, rel_id)
#                 report_lines.append(f"   {rel_name}: {count} 次")
#
#         report_lines.extend([
#             "",
#             "=" * 60,
#         ])
#
#         # 写入文件
#         report_path = os.path.join(output_dir, 'preprocessing_report.txt')
#         with open(report_path, 'w', encoding='utf-8') as f:
#             f.write('\n'.join(report_lines))
#
#         print(f"  报告已生成: {report_path}")
#
#     def run_full_pipeline(self):
#         """
#         运行完整的预处理流水线
#         """
#         print("开始知识图谱数据预处理...")
#         print("=" * 60)
#
#         # 步骤1: 加载数据
#         self.load_data()
#
#         # 步骤2: 构建图结构
#         self.build_graph_structure()
#
#         # 步骤3: 统计分析
#         self.analyze_statistics()
#
#         # 步骤4: 构建分层索引
#         self.build_layer_indices()
#
#         # 保存结果
#         self.save_results()
#
#         print("\n" + "=" * 60)
#         print("知识图谱预处理完成！")
#         print("=" * 60)
#
#
# # 使用示例
# if __name__ == "__main__":
#     # 初始化构建器
#     kg_builder = KGGraphBuilder(None)  # 修改为您的数据目录
#
#     # 运行完整流水线
#     # kg_builder.run_full_pipeline()
#
#     # 或者分步运行
#     kg_builder.load_data()
#     kg_builder.build_graph_structure()
#     kg_builder.analyze_statistics()
#     kg_builder.build_layer_indices()
#     kg_builder.save_results()
#     kg_builder.generate_summary_report(output_dir='./kg_preprocessed')
#

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import pickle
import os
from tqdm import tqdm
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


class KGGraphBuilder:
    """
    知识图谱构建器 - 第一阶段：数据预处理与索引构建
    """

    def __init__(self, data_dir=None):
        """
        初始化知识图谱构建器

        Args:
            data_dir: 数据文件目录路径
        """
        if data_dir is None:
            # 使用脚本所在目录
            self.data_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        else:
            # 原有的处理逻辑
            self.data_dir = Path(data_dir)

        print(f"数据目录: {self.data_dir}")

        # 输出目录
        self.output_dir = self.data_dir / 'kg_preprocessed'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 数据存储结构
        self.user_entities = None  # 用户实体
        self.item_entities = None  # 商品实体
        self.other_entities = None  # 其他实体
        self.relations = None  # 关系映射
        self.train_triples = None  # 训练三元组（用户-商品交互）
        self.other_triples = None  # 其他三元组

        # 图结构
        self.graph = defaultdict(lambda: defaultdict(list))  # 邻接表: {head: {relation: [tail1, tail2, ...]}}
        self.reverse_graph = defaultdict(lambda: defaultdict(list))  # 反向邻接表: {tail: {relation: [head1, head2, ...]}}

        # 索引映射
        self.entity_id_to_info = {}  # 实体ID -> {name, type}
        self.entity_name_to_id = {}  # 实体名称 -> 实体ID
        self.relation_id_to_name = {}  # 关系ID -> 关系名称
        self.relation_name_to_id = {}  # 关系名称 -> 关系ID

        # 统计信息 - 使用可序列化的结构
        self.stats = {
            'num_users': 0,
            'num_items': 0,
            'num_other_entities': 0,
            'num_relations': 0,
            'num_train_triples': 0,
            'num_other_triples': 0,
            'item_popularity': {},  # 使用普通字典而不是Counter
            'node_degrees': {},  # 使用普通字典而不是defaultdict
        }

        # 分层索引
        self.layer_indices = {
            'core': set(),  # 核心层：用户历史交互商品及其1跳邻居
            'extended': set(),  # 扩展层：2-3跳邻居
            'exploration': set(),  # 探索层：4-5跳邻居
        }

    def load_data(self):
        """
        步骤1：加载并解析知识图谱数据
        """
        print("=" * 50)
        print("步骤1：加载知识图谱数据")
        print("=" * 50)

        try:
            # 1.1 加载用户实体
            print("加载用户实体...")
            self.user_entities = self._load_entities_file('kg_users.txt', 'user')
            self.stats['num_users'] = len(self.user_entities)
            print(f"  加载完成: {self.stats['num_users']} 个用户")

            # 1.2 加载商品实体
            print("加载商品实体...")
            self.item_entities = self._load_entities_file('kg_items.txt', 'item')
            self.stats['num_items'] = len(self.item_entities)
            print(f"  加载完成: {self.stats['num_items']} 个商品")

            # 1.3 加载其他实体
            print("加载其他实体...")
            self.other_entities = self._load_entities_file('kg_entities.txt', 'other')
            self.stats['num_other_entities'] = len(self.other_entities)
            print(f"  加载完成: {self.stats['num_other_entities']} 个其他实体")

            # 1.4 加载关系映射
            print("加载关系映射...")
            self.relations = pd.read_csv(
                self.data_dir / 'kg_relations.txt',
                sep='\t',
                header=None,
                names=['relation_name', 'relation_id'],
                dtype=str
            )

            # 建立关系ID和名称的双向映射
            for _, row in self.relations.iterrows():
                rel_name, rel_id = str(row['relation_name']), str(row['relation_id'])
                self.relation_id_to_name[rel_id] = rel_name
                self.relation_name_to_id[rel_name] = rel_id

            self.stats['num_relations'] = len(self.relations)
            print(f"  加载完成: {self.stats['num_relations']} 种关系")

            # 1.5 加载训练三元组（用户-商品交互）
            print("加载训练三元组（用户-商品交互）...")
            self.train_triples = pd.read_csv(
                self.data_dir / 'kg_train_triples.txt',
                sep='\t',
                header=None,
                names=['head', 'tail', 'relation'],
                dtype=str
            )
            self.stats['num_train_triples'] = len(self.train_triples)

            # 统计商品流行度（基于用户交互） - 使用普通字典
            item_popularity_counter = Counter()
            for _, row in self.train_triples.iterrows():
                item_id = str(row['tail'])  # 假设tail是商品ID
                item_popularity_counter[item_id] += 1

            # 将Counter转换为普通字典
            self.stats['item_popularity'] = dict(item_popularity_counter)

            print(f"  加载完成: {self.stats['num_train_triples']} 个交互三元组")
            print(f"  统计了 {len(self.stats['item_popularity'])} 个商品的流行度")

            # 1.6 加载其他三元组
            print("加载其他三元组...")
            self.other_triples = pd.read_csv(
                self.data_dir / 'kg_other_triples.txt',
                sep='\t',
                header=None,
                names=['head', 'tail', 'relation'],
                dtype=str
            )
            self.stats['num_other_triples'] = len(self.other_triples)
            print(f"  加载完成: {self.stats['num_other_triples']} 个其他三元组")

            print("\n所有数据加载完成！")

        except FileNotFoundError as e:
            print(f"文件未找到: {e}")
            raise
        except Exception as e:
            print(f"数据加载失败: {e}")
            raise

    def _load_entities_file(self, filename, entity_type):
        """
        加载实体文件

        Args:
            filename: 文件名
            entity_type: 实体类型（user/item/other）

        Returns:
            DataFrame: 实体数据
        """
        filepath = self.data_dir / filename
        df = pd.read_csv(filepath, sep='\t', header=None, dtype=str)

        if entity_type == 'other':
            # 其他实体的格式: [entity_type]::[entity_name]\t[entity_id]
            df.columns = ['raw', 'entity_id']
            # 解析实体类型和名称
            df['entity_type'] = df['raw'].apply(lambda x: str(x).split('::')[0] if isinstance(x, str) else 'unknown')
            df['entity_name'] = df['raw'].apply(
                lambda x: str(x).split('::')[1] if isinstance(x, str) and '::' in x else str(x))
            df = df[['entity_id', 'entity_type', 'entity_name']]
        else:
            # 用户和商品实体的格式: [entity_type]::[entity_name]\t[entity_id]
            df.columns = ['raw', 'entity_id']
            # 解析实体类型和名称
            try:
                parts = str(df['raw'].iloc[0]).split('::')
                entity_prefix = parts[0]
                df['entity_name'] = df['raw'].apply(
                    lambda x: str(x).split('::')[1] if isinstance(x, str) and '::' in x else str(x))
                df['entity_type'] = entity_prefix
            except:
                df['entity_name'] = df['raw']
                df['entity_type'] = entity_type
            df = df[['entity_id', 'entity_type', 'entity_name']]

        # 建立实体ID到信息的映射
        for _, row in df.iterrows():
            entity_id = str(row['entity_id'])
            entity_name = str(row['entity_name'])
            entity_type = str(row['entity_type'])

            self.entity_id_to_info[entity_id] = {
                'name': entity_name,
                'type': entity_type
            }

            # 建立名称到ID的映射（注意：名称可能重复，所以使用列表存储）
            if entity_name not in self.entity_name_to_id:
                self.entity_name_to_id[entity_name] = []
            self.entity_name_to_id[entity_name].append(entity_id)

        return df

    def build_graph_structure(self):
        """
        步骤2：构建图存储结构
        """
        print("\n" + "=" * 50)
        print("步骤2：构建图存储结构")
        print("=" * 50)

        # 2.1 合并所有三元组
        print("合并所有三元组...")
        all_triples = pd.concat([self.train_triples, self.other_triples], ignore_index=True)
        print(f"  总三元组数量: {len(all_triples)}")

        # 2.2 构建邻接表和反向邻接表
        print("构建邻接表和反向邻接表...")

        # 初始化一个普通字典来存储节点度数
        node_degrees = {}

        for idx, row in tqdm(all_triples.iterrows(), total=len(all_triples)):
            head = str(row['head'])
            tail = str(row['tail'])
            relation = str(row['relation'])

            # 初始化节点度数
            if head not in node_degrees:
                node_degrees[head] = {'in_degree': 0, 'out_degree': 0}
            if tail not in node_degrees:
                node_degrees[tail] = {'in_degree': 0, 'out_degree': 0}

            # 添加到正向邻接表
            if relation not in self.graph[head]:
                self.graph[head][relation] = []
            self.graph[head][relation].append(tail)

            # 添加到反向邻接表
            if relation not in self.reverse_graph[tail]:
                self.reverse_graph[tail][relation] = []
            self.reverse_graph[tail][relation].append(head)

            # 更新节点度数统计
            node_degrees[head]['out_degree'] += 1
            node_degrees[tail]['in_degree'] += 1

        # 将节点度数存入stats
        self.stats['node_degrees'] = node_degrees

        # 2.3 验证图构建结果
        print("\n图结构统计:")
        print(f"  总节点数: {len(self.graph)}")
        if node_degrees:
            avg_out_degree = sum(d['out_degree'] for d in node_degrees.values()) / len(node_degrees)
            avg_in_degree = sum(d['in_degree'] for d in node_degrees.values()) / len(node_degrees)
            print(f"  平均出度: {avg_out_degree:.2f}")
            print(f"  平均入度: {avg_in_degree:.2f}")
        else:
            print(f"  平均出度: 0.00")
            print(f"  平均入度: 0.00")

        # 统计不同类型实体的连接情况
        user_connections = 0
        item_connections = 0
        for node_id in self.graph.keys():
            if node_id in self.entity_id_to_info:
                entity_type = self.entity_id_to_info[node_id]['type']
                if entity_type == 'user':
                    user_connections += 1
                elif entity_type == 'product':
                    item_connections += 1

        print(f"  有连接的用户节点: {user_connections}/{self.stats['num_users']}")
        print(f"  有连接的商品节点: {item_connections}/{self.stats['num_items']}")

        print("\n图结构构建完成！")

    def analyze_statistics(self):
        """
        步骤3：统计分析与元数据提取
        """
        print("\n" + "=" * 50)
        print("步骤3：统计分析与元数据提取")
        print("=" * 50)

        # 3.1 商品流行度分析
        print("商品流行度分析:")
        popularity_counts = list(self.stats['item_popularity'].values())
        if popularity_counts:
            print(f"  最多交互的商品: {max(popularity_counts)} 次")
            print(f"  最少交互的商品: {min(popularity_counts)} 次")
            print(f"  平均交互次数: {np.mean(popularity_counts):.2f}")
            print(f"  中位数交互次数: {np.median(popularity_counts):.2f}")

            # 分析流行度分布
            pop_quartiles = np.percentile(popularity_counts, [25, 50, 75, 95])
            print(f"  25%分位数: {pop_quartiles[0]:.2f}")
            print(f"  75%分位数: {pop_quartiles[2]:.2f}")
            print(f"  95%分位数: {pop_quartiles[3]:.2f}")

        # 3.2 节点度数分析
        print("\n节点度数分析:")
        node_degrees = self.stats.get('node_degrees', {})
        if node_degrees:
            out_degrees = [d['out_degree'] for d in node_degrees.values()]
            in_degrees = [d['in_degree'] for d in node_degrees.values()]

            print(f"  最大出度: {max(out_degrees) if out_degrees else 0}")
            print(f"  最大入度: {max(in_degrees) if in_degrees else 0}")
            print(f"  平均出度: {np.mean(out_degrees):.2f}" if out_degrees else "0.00")
            print(f"  平均入度: {np.mean(in_degrees):.2f}" if in_degrees else "0.00")
        else:
            print("  没有节点度数数据")

        # 3.3 高频关系模式提取
        print("\n高频关系模式提取:")
        relation_counts = Counter()
        for head in self.graph:
            for relation in self.graph[head]:
                relation_counts[relation] += len(self.graph[head][relation])

        print("  前10个最常见的关系:")
        for relation, count in relation_counts.most_common(10):
            rel_name = self.relation_id_to_name.get(relation, relation)
            print(f"    {rel_name}: {count} 次")

        # 3.4 保存高频三元组模式
        triple_patterns = Counter()
        for head in self.graph:
            for relation in self.graph[head]:
                for tail in self.graph[head][relation]:
                    # 获取实体类型
                    head_type = self.entity_id_to_info.get(head, {}).get('type', 'unknown')
                    tail_type = self.entity_id_to_info.get(tail, {}).get('type', 'unknown')
                    rel_name = self.relation_id_to_name.get(relation, relation)

                    pattern = f"{head_type}--{rel_name}--{tail_type}"
                    triple_patterns[pattern] += 1

        print("\n  前10个最常见的关系模式:")
        for pattern, count in triple_patterns.most_common(10):
            print(f"    {pattern}: {count} 次")

        # 保存统计结果
        self.stats['relation_counts'] = dict(relation_counts)
        self.stats['triple_patterns'] = dict(triple_patterns)

        print("\n统计分析完成！")

    def build_layer_indices(self):
        """
        步骤4：构建分层图索引
        """
        print("\n" + "=" * 50)
        print("步骤4：构建分层图索引")
        print("=" * 50)

        # 4.1 提取用户历史交互商品
        print("提取用户历史交互商品...")
        user_history_items = defaultdict(set)
        for _, row in self.train_triples.iterrows():
            user_id = str(row['head'])
            item_id = str(row['tail'])
            user_history_items[user_id].add(item_id)

        print(f"  涉及用户数: {len(user_history_items)}")

        # 4.2 为每个用户构建分层索引
        print("\n构建分层索引（按用户）...")
        total_core = 0
        total_extended = 0
        total_exploration = 0

        for user_id, items in tqdm(user_history_items.items(), desc="处理用户"):
            # 初始化当前用户的分层集合
            user_core = set(items)  # 核心层：历史交互商品

            # 第一跳邻居（从历史商品出发）
            first_hop = set()
            for item in items:
                if item in self.graph:
                    # 获取所有邻居
                    for relation in self.graph[item]:
                        first_hop.update(self.graph[item][relation])

            user_core.update(first_hop)

            # 第二跳和第三跳（扩展层）
            current_layer = first_hop
            for hop in range(2, 4):  # 第2、3跳
                next_layer = set()
                for node in current_layer:
                    if node in self.graph:
                        for relation in self.graph[node]:
                            next_layer.update(self.graph[node][relation])

                # 移除已经在核心层的节点
                next_layer = next_layer - user_core

                # 更新当前层和扩展层
                current_layer = next_layer
                if hop <= 3:  # 2-3跳属于扩展层
                    self.layer_indices['extended'].update(next_layer)
                else:  # 4+跳属于探索层
                    self.layer_indices['exploration'].update(next_layer)

            # 第四跳和第五跳（探索层）
            for hop in range(4, 6):  # 第4、5跳
                next_layer = set()
                for node in current_layer:
                    if node in self.graph:
                        for relation in self.graph[node]:
                            next_layer.update(self.graph[node][relation])

                # 移除已经在其他层的节点
                next_layer = next_layer - user_core - self.layer_indices['extended']

                # 更新当前层和探索层
                current_layer = next_layer
                self.layer_indices['exploration'].update(next_layer)

            # 将核心层添加到全局索引
            self.layer_indices['core'].update(user_core)

        # 4.3 去重并统计
        print("\n分层索引统计:")
        print(f"  核心层节点数: {len(self.layer_indices['core'])}")
        print(f"  扩展层节点数: {len(self.layer_indices['extended'])}")
        print(f"  探索层节点数: {len(self.layer_indices['exploration'])}")

        # 计算重叠情况
        core_extended_overlap = len(self.layer_indices['core'] & self.layer_indices['extended'])
        core_exploration_overlap = len(self.layer_indices['core'] & self.layer_indices['exploration'])
        extended_exploration_overlap = len(self.layer_indices['extended'] & self.layer_indices['exploration'])

        print(f"  核心层与扩展层重叠: {core_extended_overlap} 个节点")
        print(f"  核心层与探索层重叠: {core_exploration_overlap} 个节点")
        print(f"  扩展层与探索层重叠: {extended_exploration_overlap} 个节点")

        # 4.4 分层节点类型分析
        print("\n分层节点类型分布:")
        for layer_name, layer_nodes in self.layer_indices.items():
            type_counter = Counter()
            for node in layer_nodes:
                if node in self.entity_id_to_info:
                    entity_type = self.entity_id_to_info[node]['type']
                    type_counter[entity_type] += 1

            print(f"  {layer_name}层:")
            for etype, count in type_counter.most_common():
                print(f"    {etype}: {count}")

        print("\n分层索引构建完成！")

    def save_results(self, filepath=None):
        """保存处理结果"""
        if filepath is None:
            filepath = self.output_dir / "kg_builder_stats.pkl"

        # 创建可序列化的统计信息副本
        serializable_stats = {}

        for key, value in self.stats.items():
            try:
                # 对于字典类型的值，确保所有键都是字符串
                if isinstance(value, dict):
                    # 递归处理嵌套字典
                    def convert_dict_keys(d):
                        if isinstance(d, dict):
                            return {str(k): convert_dict_keys(v) for k, v in d.items()}
                        elif isinstance(d, (list, tuple)):
                            return [convert_dict_keys(item) for item in d]
                        else:
                            return d

                    serializable_stats[key] = convert_dict_keys(value)
                else:
                    # 非字典值直接存储
                    serializable_stats[key] = value

                # 测试序列化
                pickle.dumps(serializable_stats[key])

            except (pickle.PickleError, TypeError, AttributeError) as e:
                # 如果是函数，转换为字符串表示
                if callable(value):
                    serializable_stats[key] = f"<function: {key}>"
                # 如果是特殊对象，转换为基本类型
                elif hasattr(value, '__dict__'):
                    # 尝试提取可序列化的属性
                    try:
                        serializable_stats[key] = {
                            k: v for k, v in value.__dict__.items()
                            if not callable(v) and not k.startswith('_')
                        }
                    except:
                        serializable_stats[key] = f"<object: {type(value).__name__}>"
                else:
                    serializable_stats[key] = f"<non-serializable: {type(value).__name__}>"
                print(f"警告: 无法序列化统计项 '{key}'，已转换为字符串表示")

        # 保存可序列化的统计信息
        with open(filepath, 'wb') as f:
            pickle.dump(serializable_stats, f)

        print(f"统计信息已保存到: {filepath}")

        # 同时保存其他结构
        self._save_additional_data()

    def _save_additional_data(self):
        """保存其他重要数据"""
        try:
            # 保存图结构
            graph_dict = {}
            for head in self.graph:
                graph_dict[str(head)] = {}
                for relation in self.graph[head]:
                    graph_dict[str(head)][str(relation)] = [str(tail) for tail in self.graph[head][relation]]

            reverse_graph_dict = {}
            for tail in self.reverse_graph:
                reverse_graph_dict[str(tail)] = {}
                for relation in self.reverse_graph[tail]:
                    reverse_graph_dict[str(tail)][str(relation)] = [str(head) for head in
                                                                    self.reverse_graph[tail][relation]]

            with open(self.output_dir / 'graph_adjacency.pkl', 'wb') as f:
                pickle.dump(graph_dict, f)

            with open(self.output_dir / 'graph_reverse.pkl', 'wb') as f:
                pickle.dump(reverse_graph_dict, f)
            print("图结构保存成功")
        except Exception as e:
            print(f"图结构保存失败: {e}")

        try:
            # 保存索引映射
            with open(self.output_dir / 'entity_mappings.pkl', 'wb') as f:
                pickle.dump({
                    'entity_id_to_info': {str(k): v for k, v in self.entity_id_to_info.items()},
                    'entity_name_to_id': {str(k): v for k, v in self.entity_name_to_id.items()},
                    'relation_id_to_name': {str(k): str(v) for k, v in self.relation_id_to_name.items()},
                    'relation_name_to_id': {str(k): str(v) for k, v in self.relation_name_to_id.items()},
                }, f)
            print("索引映射保存成功")
        except Exception as e:
            print(f"索引映射保存失败: {e}")

        try:
            # 保存分层索引
            with open(self.output_dir / 'layer_indices.pkl', 'wb') as f:
                pickle.dump({
                    'core': [str(node) for node in self.layer_indices['core']],
                    'extended': [str(node) for node in self.layer_indices['extended']],
                    'exploration': [str(node) for node in self.layer_indices['exploration']],
                }, f)
            print("分层索引保存成功")
        except Exception as e:
            print(f"分层索引保存失败: {e}")

        try:
            # 保存原始数据
            self.train_triples.to_csv(self.output_dir / 'train_triples.csv', index=False)
            self.other_triples.to_csv(self.output_dir / 'other_triples.csv', index=False)
            print("原始数据保存成功")
        except Exception as e:
            print(f"原始数据保存失败: {e}")

    def generate_summary_report(self):
        """
        生成汇总报告
        """
        print("\n" + "=" * 50)
        print("生成汇总报告")
        print("=" * 50)

        report_lines = [
            "=" * 60,
            "知识图谱预处理汇总报告",
            "=" * 60,
            f"生成时间: {pd.Timestamp.now()}",
            f"数据目录: {self.data_dir}",
            f"输出目录: {self.output_dir}",
            "",
            "1. 数据统计:",
            f"   用户数量: {self.stats['num_users']}",
            f"   商品数量: {self.stats['num_items']}",
            f"   其他实体数量: {self.stats['num_other_entities']}",
            f"   关系类型数量: {self.stats['num_relations']}",
            f"   用户-商品交互数量: {self.stats['num_train_triples']}",
            f"   其他三元组数量: {self.stats['num_other_triples']}",
            "",
            "2. 图结构统计:",
            f"   总节点数: {len(self.graph)}",
        ]

        # 添加节点度数统计
        node_degrees = self.stats.get('node_degrees', {})
        if node_degrees:
            avg_out_degree = sum(d.get('out_degree', 0) for d in node_degrees.values()) / len(node_degrees)
            avg_in_degree = sum(d.get('in_degree', 0) for d in node_degrees.values()) / len(node_degrees)
            report_lines.append(f"   平均出度: {avg_out_degree:.2f}")
            report_lines.append(f"   平均入度: {avg_in_degree:.2f}")
        else:
            report_lines.append(f"   平均出度: 0.00")
            report_lines.append(f"   平均入度: 0.00")

        report_lines.extend([
            "",
            "3. 分层索引统计:",
            f"   核心层节点数: {len(self.layer_indices['core'])}",
            f"   扩展层节点数: {len(self.layer_indices['extended'])}",
            f"   探索层节点数: {len(self.layer_indices['exploration'])}",
            "",
            "4. 商品流行度分布:",
        ])

        # 添加流行度分布
        pop_values = list(self.stats['item_popularity'].values())
        if pop_values:
            report_lines.extend([
                f"   最多交互: {max(pop_values)} 次",
                f"   最少交互: {min(pop_values)} 次",
                f"   平均交互: {np.mean(pop_values):.2f} 次",
                f"   中位数交互: {np.median(pop_values):.2f} 次",
            ])
        else:
            report_lines.append("   没有商品流行度数据")

        report_lines.extend([
            "",
            "5. 前5个最常见的关系:",
        ])

        # 添加最常见的关系
        relation_counts = self.stats.get('relation_counts', {})
        if relation_counts:
            # 转换为普通字典并排序
            sorted_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for rel_id, count in sorted_relations:
                rel_name = self.relation_id_to_name.get(rel_id, rel_id)
                report_lines.append(f"   {rel_name}: {count} 次")
        else:
            report_lines.append("   没有关系统计数据")

        report_lines.extend([
            "",
            "=" * 60,
        ])

        # 写入文件
        report_path = self.output_dir / 'preprocessing_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"报告已生成: {report_path}")

    def run_full_pipeline(self):
        """
        运行完整的预处理流水线
        """
        print("开始知识图谱数据预处理...")
        print("=" * 60)

        # 步骤1: 加载数据
        self.load_data()

        # 步骤2: 构建图结构
        self.build_graph_structure()

        # 步骤3: 统计分析
        self.analyze_statistics()

        # 步骤4: 构建分层索引
        self.build_layer_indices()

        # 保存结果
        self.save_results()

        # 生成报告
        self.generate_summary_report()

        print("\n" + "=" * 60)
        print("知识图谱预处理完成！")
        print("=" * 60)


# 使用示例
if __name__ == "__main__":
    # 初始化构建器
    kg_builder = KGGraphBuilder(None)

    # 运行完整流水线
    kg_builder.run_full_pipeline()