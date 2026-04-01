"""
拓扑算法模块
负责计算节点关联度，构建拓扑结构
"""

import time
import math
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
import logging
from collections import defaultdict, deque
import heapq

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import networkx as nx

from ..api.schemas import MemoryNode, MemoryEdge


logger = logging.getLogger(__name__)


@dataclass
class TopologyConfig:
    """拓扑算法配置"""
    min_similarity: float = 0.3
    max_edges_per_node: int = 10
    community_detection: bool = True
    pca_reduction: bool = False
    pca_components: int = 50
    clustering_enabled: bool = True
    clustering_threshold: float = 0.5


class TopologyAlgorithms:
    """拓扑算法核心类"""
    
    def __init__(self, config: Optional[TopologyConfig] = None):
        """
        初始化拓扑算法
        
        Args:
            config: 算法配置
        """
        self.config = config or TopologyConfig()
        self._pca_model = None
        
        logger.info(f"TopologyAlgorithms initialized with config: {self.config}")
    
    def calculate_node_relationships(self, 
                                   nodes: List[MemoryNode],
                                   edges: List[MemoryEdge]) -> List[Tuple[str, str, float]]:
        """
        计算节点关联度
        
        Args:
            nodes: 记忆节点列表
            edges: 现有边列表
            
        Returns:
            List[Tuple[str, str, float]]: 节点关系列表(源ID, 目标ID, 关联度)
        """
        start_time = time.time()
        
        # 构建节点索引
        node_index = {node.node_id: node for node in nodes}
        vector_nodes = [node for node in nodes if node.vector is not None]
        
        if not vector_nodes:
            logger.warning("No vectors available for relationship calculation")
            return []
        
        # 提取向量
        vectors = []
        vector_node_ids = []
        
        for node in vector_nodes:
            if node.vector:
                vectors.append(node.vector)
                vector_node_ids.append(node.node_id)
        
        if not vectors:
            return []
        
        vectors_array = np.array(vectors)
        
        # 可选：PCA降维
        if self.config.pca_reduction and len(vectors) > self.config.pca_components:
            vectors_array = self._apply_pca(vectors_array)
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(vectors_array)
        
        # 构建关系列表
        relationships = []
        existing_edges = {(edge.source_id, edge.target_id) for edge in edges}
        
        n = len(vector_node_ids)
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                
                if similarity >= self.config.min_similarity:
                    source_id = vector_node_ids[i]
                    target_id = vector_node_ids[j]
                    
                    # 检查是否已存在边
                    if (source_id, target_id) not in existing_edges:
                        relationships.append((source_id, target_id, similarity))
        
        # 限制每个节点的边数
        relationships = self._limit_edges_per_node(relationships)
        
        calc_time = time.time() - start_time
        logger.info(f"Calculated {len(relationships)} relationships in {calc_time:.3f}s")
        
        return relationships
    
    def build_topology_graph(self,
                           nodes: List[MemoryNode],
                           edges: List[MemoryEdge]) -> nx.Graph:
        """
        构建拓扑图
        
        Args:
            nodes: 记忆节点列表
            edges: 记忆边列表
            
        Returns:
            nx.Graph: 网络图对象
        """
        # 创建图
        G = nx.Graph()
        
        # 添加节点
        for node in nodes:
            G.add_node(
                node.node_id,
                content=node.content,
                vector=node.vector,
                metadata=node.metadata
            )
        
        # 添加边
        for edge in edges:
            G.add_edge(
                edge.source_id,
                edge.target_id,
                relationship=edge.relationship,
                weight=edge.weight
            )
        
        return G
    
    def analyze_topology(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        分析拓扑结构
        
        Args:
            graph: 拓扑图
            
        Returns:
            Dict[str, Any]: 拓扑分析结果
        """
        if not graph.nodes():
            return {"error": "Empty graph"}
        
        start_time = time.time()
        
        analysis = {
            "basic_stats": self._calculate_basic_stats(graph),
            "centrality_measures": self._calculate_centrality(graph),
            "community_structure": {},
            "clustering_coefficient": nx.average_clustering(graph) if graph.nodes() else 0.0,
            "diameter": nx.diameter(graph) if nx.is_connected(graph) else None,
            "density": nx.density(graph)
        }
        
        # 社区检测
        if self.config.community_detection and len(graph.nodes()) > 2:
            analysis["community_structure"] = self._detect_communities(graph)
        
        # 聚类分析
        if self.config.clustering_enabled:
            analysis["clusters"] = self._cluster_nodes(graph)
        
        analysis_time = time.time() - start_time
        analysis["analysis_time"] = analysis_time
        
        logger.info(f"Topology analysis completed in {analysis_time:.3f}s")
        
        return analysis
    
    def find_central_nodes(self, graph: nx.Graph, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        查找中心节点
        
        Args:
            graph: 拓扑图
            top_k: 返回数量
            
        Returns:
            List[Tuple[str, float]]: 中心节点ID和中心性分数
        """
        if not graph.nodes():
            return []
        
        # 计算度中心性
        degree_centrality = nx.degree_centrality(graph)
        
        # 计算接近中心性
        try:
            closeness_centrality = nx.closeness_centrality(graph)
        except:
            closeness_centrality = {node: 0.0 for node in graph.nodes()}
        
        # 计算特征向量中心性
        try:
            eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000)
        except:
            eigenvector_centrality = {node: 0.0 for node in graph.nodes()}
        
        # 综合中心性分数
        combined_scores = {}
        for node in graph.nodes():
            combined_score = (
                degree_centrality.get(node, 0) * 0.4 +
                closeness_centrality.get(node, 0) * 0.3 +
                eigenvector_centrality.get(node, 0) * 0.3
            )
            combined_scores[node] = combined_score
        
        # 排序并返回top_k
        sorted_nodes = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_nodes[:top_k]
    
    def find_shortest_path(self, graph: nx.Graph, 
                          source_id: str, 
                          target_id: str) -> Optional[List[str]]:
        """
        查找最短路径
        
        Args:
            graph: 拓扑图
            source_id: 源节点ID
            target_id: 目标节点ID
            
        Returns:
            Optional[List[str]]: 最短路径节点ID列表
        """
        if source_id not in graph or target_id not in graph:
            return None
        
        try:
            path = nx.shortest_path(graph, source=source_id, target=target_id)
            return path
        except nx.NetworkXNoPath:
            return None
    
    def find_bridges(self, graph: nx.Graph) -> List[Tuple[str, str]]:
        """
        查找桥接边
        
        Args:
            graph: 拓扑图
            
        Returns:
            List[Tuple[str, str]]: 桥接边列表
        """
        if not nx.is_connected(graph):
            return []
        
        try:
            bridges = list(nx.bridges(graph))
            return bridges
        except:
            return []
    
    def find_communities(self, graph: nx.Graph) -> List[List[str]]:
        """
        查找社区
        
        Args:
            graph: 拓扑图
            
        Returns:
            List[List[str]]: 社区列表
        """
        if len(graph.nodes()) < 3:
            return [list(graph.nodes())]
        
        try:
            # 使用Louvain算法
            import community as community_louvain
            
            partition = community_louvain.best_partition(graph)
            
            # 按社区分组
            communities_dict = defaultdict(list)
            for node, comm_id in partition.items():
                communities_dict[comm_id].append(node)
            
            communities = list(communities_dict.values())
            return communities
            
        except ImportError:
            # 回退到连通组件
            return [list(comp) for comp in nx.connected_components(graph)]
    
    def calculate_semantic_distance(self, 
                                  node1: MemoryNode, 
                                  node2: MemoryNode) -> float:
        """
        计算语义距离
        
        Args:
            node1: 第一个节点
            node2: 第二个节点
            
        Returns:
            float: 语义距离(0-1，越小越相似)
        """
        # 向量距离
        vector_distance = 1.0
        
        if node1.vector and node2.vector:
            vec1 = np.array(node1.vector).reshape(1, -1)
            vec2 = np.array(node2.vector).reshape(1, -1)
            similarity = cosine_similarity(vec1, vec2)[0][0]
            vector_distance = 1.0 - similarity
        
        # 文本距离
        text_distance = self._calculate_text_distance(node1.content, node2.content)
        
        # 综合距离
        combined_distance = vector_distance * 0.7 + text_distance * 0.3
        
        return combined_distance
    
    def optimize_topology(self, 
                         nodes: List[MemoryNode],
                         edges: List[MemoryEdge],
                         target_density: float = 0.1) -> List[MemoryEdge]:
        """
        优化拓扑结构
        
        Args:
            nodes: 记忆节点列表
            edges: 现有边列表
            target_density: 目标密度
            
        Returns:
            List[MemoryEdge]: 优化后的边列表
        """
        if not nodes:
            return []
        
        # 构建图
        graph = self.build_topology_graph(nodes, edges)
        
        # 计算当前密度
        current_density = nx.density(graph)
        
        if current_density <= target_density:
            logger.info(f"Current density {current_density:.3f} <= target {target_density}, no optimization needed")
            return edges
        
        # 需要减少边数
        edges_to_remove = self._select_edges_to_remove(graph, current_density, target_density)
        
        # 创建新的边列表
        optimized_edges = []
        edge_set = {(edge.source_id, edge.target_id) for edge in edges}
        
        for edge in edges:
            if (edge.source_id, edge.target_id) not in edges_to_remove:
                optimized_edges.append(edge)
        
        logger.info(f"Optimized topology: removed {len(edges_to_remove)} edges, "
                   f"density from {current_density:.3f} to {nx.density(graph):.3f}")
        
        return optimized_edges
    
    # 私有辅助方法
    
    def _apply_pca(self, vectors: np.ndarray) -> np.ndarray:
        """应用PCA降维"""
        try:
            if self._pca_model is None:
                self._pca_model = PCA(n_components=self.config.pca_components)
                reduced_vectors = self._pca_model.fit_transform(vectors)
            else:
                reduced_vectors = self._pca_model.transform(vectors)
            
            logger.info(f"Applied PCA: {vectors.shape[1]} -> {reduced_vectors.shape[1]} dimensions")
            return reduced_vectors
            
        except Exception as e:
            logger.error(f"PCA failed: {e}")
            return vectors
    
    def _limit_edges_per_node(self, 
                             relationships: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
        """限制每个节点的边数"""
        edge_counts = defaultdict(int)
        filtered_relationships = []
        
        # 按相似度排序
        relationships.sort(key=lambda x: x[2], reverse=True)
        
        for source_id, target_id, similarity in relationships:
            if (edge_counts[source_id] < self.config.max_edges_per_node and 
                edge_counts[target_id] < self.config.max_edges_per_node):
                filtered_relationships.append((source_id, target_id, similarity))
                edge_counts[source_id] += 1
                edge_counts[target_id] += 1
        
        return filtered_relationships
    
    def _calculate_basic_stats(self, graph: nx.Graph) -> Dict[str, Any]:
        """计算基本统计信息"""
        stats = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "avg_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
            "max_degree": max(dict(graph.degree()).values()) if graph.number_of_nodes() > 0 else 0,
            "min_degree": min(dict(graph.degree()).values()) if graph.number_of_nodes() > 0 else 0,
            "is_connected": nx.is_connected(graph),
            "num_components": nx.number_connected_components(graph)
        }
        
        return stats
    
    def _calculate_centrality(self, graph: nx.Graph) -> Dict[str, Dict[str, float]]:
        """计算中心性指标"""
        centrality = {}
        
        try:
            # 度中心性
            centrality["degree"] = nx.degree_centrality(graph)
        except:
            centrality["degree"] = {}
        
        try:
            # 接近中心性
            centrality["closeness"] = nx.closeness_centrality(graph)
        except:
            centrality["closeness"] = {}
        
        try:
            # 介数中心性
            centrality["betweenness"] = nx.betweenness_centrality(graph)
        except:
            centrality["betweenness"] = {}
        
        try:
            # 特征向量中心性
            centrality["eigenvector"] = nx.eigenvector_centrality(graph, max_iter=1000)
        except:
            centrality["eigenvector"] = {}
        
        return centrality
    
    def _detect_communities(self, graph: nx.Graph) -> Dict[str, Any]:
        """检测社区结构"""
        try:
            import community as community_louvain
            
            partition = community_louvain.best_partition(graph)
            
            # 计算模块度
            modularity = community_louvain.modularity(partition, graph)
            
            # 统计社区信息
            communities_dict = defaultdict(list)
            for node, comm_id in partition.items():
                communities_dict[comm_id].append(node)
            
            communities_info = []
            for comm_id, nodes in communities_dict.items():
                subgraph = graph.subgraph(nodes)
                communities_info.append({
                    "id": comm_id,
                    "size": len(nodes),
                    "density": nx.density(subgraph),
                    "avg_degree": sum(dict(subgraph.degree()).values()) / len(nodes) if nodes else 0
                })
            
            return {
                "modularity": modularity,
                "num_communities": len(communities_dict),
                "communities": communities_info
            }
            
        except ImportError:
            return {
                "modularity": 0.0,
                "num_communities": 1,
                "communities": [{"id": 0, "size": len(graph.nodes()), "density": nx.density(graph)}]
            }
    
    def _cluster_nodes(self, graph: nx.Graph) -> Dict[str, Any]:
        """聚类节点"""
        if not graph.nodes():
            return {"clusters": [], "silhouette_score": 0.0}
        
        try:
            # 使用谱聚类
            from sklearn.cluster import SpectralClustering
            
            # 构建邻接矩阵
            nodes = list(graph.nodes())
            n = len(nodes)
            
            if n < 2:
                return {"clusters": [nodes], "silhouette_score": 0.0}
            
            # 创建邻接矩阵
            adj_matrix = nx.to_numpy_array(graph, nodelist=nodes)
            
            # 确定聚类数量
            n_clusters = min(10, max(2, n // 5))
            
            # 谱聚类
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42
            )
            
            labels = clustering.fit_predict(adj_matrix)
            
            # 组织聚类结果
            clusters_dict = defaultdict(list)
            for node, label in zip(nodes, labels):
                clusters_dict[label].append(node)
            
            clusters = list(clusters_dict.values())
            
            return {
                "clusters": clusters,
                "num_clusters": len(clusters),
                "labels": labels.tolist()
            }
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return {"clusters": [list(graph.nodes())], "silhouette_score": 0.0}
    
    def _calculate_text_distance(self, text1: str, text2: str) -> float:
        """计算文本距离"""
        if not text1 or not text2:
            return 1.0
        
        # 简单实现：基于词袋的Jaccard距离
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 1.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        return 1.0 - jaccard_similarity
    
    def _select_edges_to_remove(self, 
                               graph: nx.Graph, 
                               current_density: float,
                               target_density: float) -> Set[Tuple[str, str]]:
        """选择要删除的边"""
        num_edges_to_remove = int(
            (current_density - target_density) * 
            graph.number_of_nodes() * (graph.number_of_nodes() - 1) / 2
        )
        
        if num_edges_to_remove <= 0:
            return set()
        
        # 按权重排序边（删除权重最低的边）
        edges_with_weights = []
        for u, v, data in graph.edges(data=True):
            weight = data.get('weight', 1.0)
            edges_with_weights.append((weight, u, v))
        
        # 按权重升序排序
        edges_with_weights.sort(key=lambda x: x[0])
        
        # 选择要删除的边
        edges_to_remove = set()
        for i in range(min(num_edges_to_remove, len(edges_with_weights))):
            _, u, v = edges_with_weights[i]
            edges_to_remove.add((u, v))
        
        return edges_to_remove