"""
拓扑记忆管理器 - 核心引擎
负责记忆节点的创建、链接和搜索
"""

import uuid
import time
import json
import math
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
import logging
from collections import defaultdict, deque
import heapq

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..api.schemas import MemoryNode, MemoryEdge


logger = logging.getLogger(__name__)


@dataclass
class MemoryNodeData:
    """记忆节点数据类"""
    node_id: str
    content: str
    vector: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def update(self, content: Optional[str] = None, 
               vector: Optional[List[float]] = None,
               metadata: Optional[Dict[str, Any]] = None) -> None:
        """更新节点数据"""
        if content is not None:
            self.content = content
        if vector is not None:
            self.vector = vector
        if metadata is not None:
            self.metadata.update(metadata)
        self.updated_at = datetime.now()
    
    def mark_accessed(self) -> None:
        """标记为已访问"""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def to_schema(self) -> MemoryNode:
        """转换为API模型"""
        return MemoryNode(
            node_id=self.node_id,
            content=self.content,
            vector=self.vector,
            metadata=self.metadata
        )


@dataclass
class MemoryEdgeData:
    """记忆边数据类"""
    source_id: str
    target_id: str
    relationship: str
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, weight: Optional[float] = None,
               relationship: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None) -> None:
        """更新边数据"""
        if weight is not None:
            self.weight = weight
        if relationship is not None:
            self.relationship = relationship
        if metadata is not None:
            self.metadata.update(metadata)
    
    def to_schema(self) -> MemoryEdge:
        """转换为API模型"""
        return MemoryEdge(
            source_id=self.source_id,
            target_id=self.target_id,
            relationship=self.relationship,
            weight=self.weight
        )


class MemoryManager:
    """记忆管理器核心类"""
    
    def __init__(self, 
                 max_nodes: int = 10000,
                 max_edges_per_node: int = 50,
                 vector_dimension: int = 384,
                 similarity_threshold: float = 0.5):
        """
        初始化记忆管理器
        
        Args:
            max_nodes: 最大节点数量
            max_edges_per_node: 每个节点最大边数
            vector_dimension: 向量维度
            similarity_threshold: 相似度阈值
        """
        self.max_nodes = max_nodes
        self.max_edges_per_node = max_edges_per_node
        self.vector_dimension = vector_dimension
        self.similarity_threshold = similarity_threshold
        
        # 存储结构
        self.nodes: Dict[str, MemoryNodeData] = {}
        self.edges: Dict[str, List[MemoryEdgeData]] = defaultdict(list)
        
        # 反向索引
        self.reverse_edges: Dict[str, List[MemoryEdgeData]] = defaultdict(list)
        
        # 向量索引（简单实现）
        self.vector_index: Dict[str, np.ndarray] = {}
        
        # 内容索引（用于文本搜索）
        self.content_index: Dict[str, Set[str]] = defaultdict(set)
        
        # 访问频率统计
        self.access_stats: Dict[str, int] = defaultdict(int)
        
        # 缓存
        self._similarity_cache: Dict[Tuple[str, str], float] = {}
        
        logger.info(f"MemoryManager initialized with max_nodes={max_nodes}")
    
    def create_memory_node(self, content: str, 
                          vector: Optional[List[float]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        创建记忆节点
        
        Args:
            content: 节点内容
            vector: 向量表示
            metadata: 元数据
            
        Returns:
            str: 创建的节点ID
        """
        # 检查节点数量限制
        if len(self.nodes) >= self.max_nodes:
            self._evict_least_used_node()
        
        # 生成唯一ID
        node_id = str(uuid.uuid4())
        
        # 创建节点数据
        node_data = MemoryNodeData(
            node_id=node_id,
            content=content,
            vector=vector,
            metadata=metadata or {}
        )
        
        # 存储节点
        self.nodes[node_id] = node_data
        
        # 更新索引
        if vector is not None:
            self.vector_index[node_id] = np.array(vector)
        
        # 更新内容索引
        self._update_content_index(node_id, content)
        
        logger.info(f"Created memory node {node_id}")
        
        return node_id
    
    def get_memory_node(self, node_id: str) -> Optional[MemoryNode]:
        """
        获取记忆节点
        
        Args:
            node_id: 节点ID
            
        Returns:
            Optional[MemoryNode]: 记忆节点或None
        """
        node_data = self.nodes.get(node_id)
        if not node_data:
            return None
        
        # 标记为已访问
        node_data.mark_accessed()
        self.access_stats[node_id] += 1
        
        return node_data.to_schema()
    
    def update_memory_node(self, node_id: str,
                          content: Optional[str] = None,
                          vector: Optional[List[float]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        更新记忆节点
        
        Args:
            node_id: 节点ID
            content: 新内容
            vector: 新向量
            metadata: 新元数据
            
        Returns:
            bool: 是否成功更新
        """
        node_data = self.nodes.get(node_id)
        if not node_data:
            return False
        
        # 更新内容索引
        if content is not None and content != node_data.content:
            self._remove_from_content_index(node_id, node_data.content)
            self._update_content_index(node_id, content)
        
        # 更新向量索引
        if vector is not None:
            self.vector_index[node_id] = np.array(vector)
        
        # 更新节点数据
        node_data.update(content, vector, metadata)
        
        logger.info(f"Updated memory node {node_id}")
        
        return True
    
    def delete_memory_node(self, node_id: str) -> bool:
        """
        删除记忆节点
        
        Args:
            node_id: 节点ID
            
        Returns:
            bool: 是否成功删除
        """
        if node_id not in self.nodes:
            return False
        
        # 删除节点
        del self.nodes[node_id]
        
        # 删除相关边
        if node_id in self.edges:
            for edge in self.edges[node_id]:
                self._remove_edge_from_reverse(edge)
            del self.edges[node_id]
        
        if node_id in self.reverse_edges:
            for edge in self.reverse_edges[node_id]:
                self._remove_edge_from_forward(edge)
            del self.reverse_edges[node_id]
        
        # 删除索引
        if node_id in self.vector_index:
            del self.vector_index[node_id]
        
        # 删除内容索引
        self._remove_from_content_index(node_id, self.nodes.get(node_id, MemoryNodeData("", "")).content)
        
        # 删除统计
        if node_id in self.access_stats:
            del self.access_stats[node_id]
        
        # 清理缓存
        self._clean_cache_for_node(node_id)
        
        logger.info(f"Deleted memory node {node_id}")
        
        return True
    
    def link_nodes(self, source_id: str, target_id: str,
                  relationship: str = "related_to",
                  weight: float = 1.0,
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        链接两个记忆节点
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            relationship: 关系类型
            weight: 权重
            metadata: 元数据
            
        Returns:
            bool: 是否成功链接
        """
        # 检查节点是否存在
        if source_id not in self.nodes or target_id not in self.nodes:
            return False
        
        # 检查边数量限制
        if len(self.edges.get(source_id, [])) >= self.max_edges_per_node:
            self._evict_weakest_edge(source_id)
        
        # 检查是否已存在边
        existing_edge = self._find_edge(source_id, target_id)
        if existing_edge:
            # 更新现有边
            existing_edge.update(weight=weight, relationship=relationship, metadata=metadata)
            logger.info(f"Updated edge between {source_id} and {target_id}")
            return True
        
        # 创建新边
        edge_data = MemoryEdgeData(
            source_id=source_id,
            target_id=target_id,
            relationship=relationship,
            weight=weight,
            metadata=metadata or {}
        )
        
        # 存储边
        self.edges[source_id].append(edge_data)
        self.reverse_edges[target_id].append(edge_data)
        
        logger.info(f"Linked nodes {source_id} -> {target_id} with relationship '{relationship}'")
        
        return True
    
    def unlink_nodes(self, source_id: str, target_id: str) -> bool:
        """
        取消两个记忆节点的链接
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            
        Returns:
            bool: 是否成功取消链接
        """
        # 查找并删除边
        edge_to_remove = None
        
        for edge in self.edges.get(source_id, []):
            if edge.target_id == target_id:
                edge_to_remove = edge
                break
        
        if not edge_to_remove:
            return False
        
        # 删除边
        self.edges[source_id].remove(edge_to_remove)
        
        # 从反向索引中删除
        if target_id in self.reverse_edges:
            for edge in self.reverse_edges[target_id]:
                if edge.source_id == source_id:
                    self.reverse_edges[target_id].remove(edge)
                    break
        
        # 清理缓存
        cache_key = (source_id, target_id)
        if cache_key in self._similarity_cache:
            del self._similarity_cache[cache_key]
        
        logger.info(f"Unlinked nodes {source_id} and {target_id}")
        
        return True
    
    def search_memory(self, query: str, 
                     limit: int = 10,
                     threshold: float = 0.5,
                     include_vectors: bool = False) -> List[MemoryNode]:
        """
        搜索记忆节点
        
        Args:
            query: 查询文本
            limit: 返回数量限制
            threshold: 相似度阈值
            include_vectors: 是否包含向量
            
        Returns:
            List[MemoryNode]: 匹配的记忆节点列表
        """
        start_time = time.time()
        
        # 文本搜索
        text_results = self._search_by_text(query, limit)
        
        # 向量搜索（如果有向量）
        vector_results = []
        if self.vector_index:
            vector_results = self._search_by_semantic(query, limit, threshold)
        
        # 合并结果
        all_results = self._merge_search_results(text_results, vector_results, limit)
        
        # 转换为API模型
        results = []
        for node_id, score in all_results:
            node_data = self.nodes.get(node_id)
            if node_data:
                memory_node = node_data.to_schema()
                if not include_vectors:
                    memory_node.vector = None
                results.append(memory_node)
        
        query_time = time.time() - start_time
        logger.info(f"Memory search found {len(results)} nodes in {query_time:.3f}s")
        
        return results
    
    def get_related_nodes(self, node_id: str, 
                         depth: int = 1,
                         limit: int = 20) -> Tuple[List[MemoryNode], List[MemoryEdge]]:
        """
        获取相关节点
        
        Args:
            node_id: 节点ID
            depth: 搜索深度
            limit: 返回数量限制
            
        Returns:
            Tuple[List[MemoryNode], List[MemoryEdge]]: 相关节点和边列表
        """
        if node_id not in self.nodes:
            return [], []
        
        # 广度优先搜索
        visited = set([node_id])
        nodes_to_process = deque([(node_id, 0)])
        related_nodes = []
        related_edges = []
        
        while nodes_to_process:
            current_node, current_depth = nodes_to_process.popleft()
            
            if current_depth >= depth:
                continue
            
            # 获取出边
            for edge in self.edges.get(current_node, []):
                if edge.target_id not in visited:
                    visited.add(edge.target_id)
                    nodes_to_process.append((edge.target_id, current_depth + 1))
                    
                    # 添加节点和边
                    target_node = self.nodes.get(edge.target_id)
                    if target_node:
                        related_nodes.append(target_node.to_schema())
                        related_edges.append(edge.to_schema())
            
            # 获取入边
            for edge in self.reverse_edges.get(current_node, []):
                if edge.source_id not in visited:
                    visited.add(edge.source_id)
                    nodes_to_process.append((edge.source_id, current_depth + 1))
                    
                    # 添加节点和边
                    source_node = self.nodes.get(edge.source_id)
                    if source_node:
                        related_nodes.append(source_node.to_schema())
                        related_edges.append(edge.to_schema())
            
            # 限制结果数量
            if len(related_nodes) >= limit:
                break
        
        # 移除起始节点
        related_nodes = [node for node in related_nodes if node.node_id != node_id]
        
        return related_nodes[:limit], related_edges[:limit]
    
    def calculate_node_similarity(self, node_id1: str, node_id2: str) -> float:
        """
        计算两个节点的相似度
        
        Args:
            node_id1: 第一个节点ID
            node_id2: 第二个节点ID
            
        Returns:
            float: 相似度分数(0-1)
        """
        # 检查缓存
        cache_key = (node_id1, node_id2)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        node1 = self.nodes.get(node_id1)
        node2 = self.nodes.get(node_id2)
        
        if not node1 or not node2:
            return 0.0
        
        similarity = 0.0
        
        # 向量相似度
        if node1.vector and node2.vector:
            vec1 = np.array(node1.vector).reshape(1, -1)
            vec2 = np.array(node2.vector).reshape(1, -1)
            vector_sim = cosine_similarity(vec1, vec2)[0][0]
            similarity = max(similarity, vector_sim)
        
        # 文本相似度（简单实现）
        text_sim = self._calculate_text_similarity(node1.content, node2.content)
        similarity = max(similarity, text_sim)
        
        # 边相似度
        edge_sim = self._calculate_edge_similarity(node_id1, node_id2)
        similarity = max(similarity, edge_sim)
        
        # 确保相似度在0-1范围内（处理浮点数精度问题）
        similarity = max(0.0, min(1.0, similarity))
        
        # 缓存结果
        self._similarity_cache[cache_key] = similarity
        
        return similarity
    
    def build_topology(self, center_node_id: str,
                      max_nodes: int = 50,
                      min_similarity: float = 0.3) -> Tuple[List[MemoryNode], List[MemoryEdge]]:
        """
        构建拓扑结构
        
        Args:
            center_node_id: 中心节点ID
            max_nodes: 最大节点数
            min_similarity: 最小相似度
            
        Returns:
            Tuple[List[MemoryNode], List[MemoryEdge]]: 拓扑节点和边列表
        """
        if center_node_id not in self.nodes:
            return [], []
        
        # 获取中心节点
        center_node = self.nodes[center_node_id]
        
        # 收集相关节点
        candidate_nodes = []
        
        # 直接连接的节点
        for edge in self.edges.get(center_node_id, []):
            target_node = self.nodes.get(edge.target_id)
            if target_node:
                candidate_nodes.append((target_node, edge.weight))
        
        # 相似节点
        for node_id, node_data in self.nodes.items():
            if node_id == center_node_id or node_id in [n.node_id for n, _ in candidate_nodes]:
                continue
            
            similarity = self.calculate_node_similarity(center_node_id, node_id)
            if similarity >= min_similarity:
                candidate_nodes.append((node_data, similarity))
        
        # 排序并限制数量
        candidate_nodes.sort(key=lambda x: x[1], reverse=True)
        selected_nodes = candidate_nodes[:max_nodes]
        
        # 构建节点列表
        topology_nodes = [center_node.to_schema()]
        for node_data, score in selected_nodes:
            topology_nodes.append(node_data.to_schema())
        
        # 构建边列表
        topology_edges = []
        
        # 添加中心节点到其他节点的边
        for node_data, score in selected_nodes:
            # 检查是否已有边
            existing_edge = self._find_edge(center_node_id, node_data.node_id)
            if existing_edge:
                topology_edges.append(existing_edge.to_schema())
            else:
                # 创建虚拟边
                topology_edges.append(
                    MemoryEdge(
                        source_id=center_node_id,
                        target_id=node_data.node_id,
                        relationship="similar_to",
                        weight=score
                    )
                )
        
        # 添加节点之间的边
        for i in range(len(selected_nodes)):
            for j in range(i + 1, len(selected_nodes)):
                node1_id = selected_nodes[i][0].node_id
                node2_id = selected_nodes[j][0].node_id
                
                existing_edge = self._find_edge(node1_id, node2_id)
                if existing_edge:
                    topology_edges.append(existing_edge.to_schema())
                else:
                    # 计算相似度
                    similarity = self.calculate_node_similarity(node1_id, node2_id)
                    if similarity >= min_similarity:
                        topology_edges.append(
                            MemoryEdge(
                                source_id=node1_id,
                                target_id=node2_id,
                                relationship="related_to",
                                weight=similarity
                            )
                        )
        
        logger.info(f"Built topology with {len(topology_nodes)} nodes and {len(topology_edges)} edges")
        
        return topology_nodes, topology_edges
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取管理器统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        total_nodes = len(self.nodes)
        total_edges = sum(len(edges) for edges in self.edges.values())
        
        # 计算平均边数
        avg_edges_per_node = total_edges / total_nodes if total_nodes > 0 else 0
        
        # 计算向量覆盖率
        vector_coverage = len(self.vector_index) / total_nodes if total_nodes > 0 else 0
        
        # 获取最活跃的节点
        most_active_nodes = sorted(
            self.access_stats.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "avg_edges_per_node": avg_edges_per_node,
            "vector_coverage": vector_coverage,
            "max_nodes": self.max_nodes,
            "max_edges_per_node": self.max_edges_per_node,
            "most_active_nodes": most_active_nodes,
            "cache_size": len(self._similarity_cache)
        }
    
    def clear_all(self) -> None:
        """清除所有记忆数据"""
        self.nodes.clear()
        self.edges.clear()
        self.reverse_edges.clear()
        self.vector_index.clear()
        self.content_index.clear()
        self.access_stats.clear()
        self._similarity_cache.clear()
        
        logger.info("Cleared all memory data")
    
    # 私有辅助方法
    
    def _evict_least_used_node(self) -> None:
        """逐出最少使用的节点"""
        if not self.nodes:
            return
        
        # 找到访问次数最少的节点
        least_used_node = min(
            self.access_stats.items(),
            key=lambda x: x[1],
            default=(None, 0)
        )[0]
        
        if least_used_node:
            self.delete_memory_node(least_used_node)
            logger.debug(f"Evicted least used node: {least_used_node}")
    
    def _evict_weakest_edge(self, node_id: str) -> None:
        """逐出最弱的边"""
        edges = self.edges.get(node_id, [])
        if not edges:
            return
        
        # 找到权重最低的边
        weakest_edge = min(edges, key=lambda e: e.weight)
        self.unlink_nodes(weakest_edge.source_id, weakest_edge.target_id)
        
        logger.debug(f"Evicted weakest edge from node {node_id}")
    
    def _find_edge(self, source_id: str, target_id: str) -> Optional[MemoryEdgeData]:
        """查找边"""
        for edge in self.edges.get(source_id, []):
            if edge.target_id == target_id:
                return edge
        return None
    
    def _remove_edge_from_reverse(self, edge: MemoryEdgeData) -> None:
        """从反向索引中删除边"""
        if edge.target_id in self.reverse_edges:
            for rev_edge in self.reverse_edges[edge.target_id]:
                if rev_edge.source_id == edge.source_id:
                    self.reverse_edges[edge.target_id].remove(rev_edge)
                    break
    
    def _remove_edge_from_forward(self, edge: MemoryEdgeData) -> None:
        """从正向索引中删除边"""
        if edge.source_id in self.edges:
            for fwd_edge in self.edges[edge.source_id]:
                if fwd_edge.target_id == edge.target_id:
                    self.edges[edge.source_id].remove(fwd_edge)
                    break
    
    def _update_content_index(self, node_id: str, content: str) -> None:
        """更新内容索引"""
        # 简单分词（实际应该使用更好的分词器）
        words = content.lower().split()
        for word in words:
            if len(word) > 2:  # 忽略短词
                self.content_index[word].add(node_id)
    
    def _remove_from_content_index(self, node_id: str, content: str) -> None:
        """从内容索引中删除"""
        words = content.lower().split()
        for word in words:
            if len(word) > 2 and word in self.content_index:
                self.content_index[word].discard(node_id)
                if not self.content_index[word]:
                    del self.content_index[word]
    
    def _search_by_text(self, query: str, limit: int) -> List[Tuple[str, float]]:
        """文本搜索"""
        query_words = query.lower().split()
        if not query_words:
            return []
        
        # 计算每个节点的匹配分数
        node_scores = defaultdict(float)
        
        for word in query_words:
            if len(word) <= 2:
                continue
            
            matching_nodes = self.content_index.get(word, set())
            for node_id in matching_nodes:
                node_scores[node_id] += 1.0
        
        # 归一化分数
        max_score = max(node_scores.values()) if node_scores else 1.0
        normalized_scores = [
            (node_id, score / max_score)
            for node_id, score in node_scores.items()
        ]
        
        # 排序并限制数量
        normalized_scores.sort(key=lambda x: x[1], reverse=True)
        return normalized_scores[:limit]
    
    def _search_by_semantic(self, query: str, limit: int, threshold: float) -> List[Tuple[str, float]]:
        """语义搜索（需要向量）"""
        if not self.vector_index:
            return []
        
        # 这里应该使用文本嵌入模型生成查询向量
        # 为简单起见，我们使用现有节点的向量进行搜索
        results = []
        
        for node_id, vector in self.vector_index.items():
            # 计算与所有其他节点的平均相似度作为查询相似度
            similarities = []
            for other_id, other_vector in self.vector_index.items():
                if node_id == other_id:
                    continue
                
                sim = cosine_similarity(
                    vector.reshape(1, -1),
                    other_vector.reshape(1, -1)
                )[0][0]
                similarities.append(sim)
            
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                if avg_similarity >= threshold:
                    results.append((node_id, avg_similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def _merge_search_results(self, text_results: List[Tuple[str, float]],
                             vector_results: List[Tuple[str, float]],
                             limit: int) -> List[Tuple[str, float]]:
        """合并搜索结果"""
        # 合并分数
        merged_scores = defaultdict(float)
        
        for node_id, score in text_results:
            merged_scores[node_id] += score * 0.7  # 文本搜索权重
        
        for node_id, score in vector_results:
            merged_scores[node_id] += score * 0.3  # 向量搜索权重
        
        # 转换为列表并排序
        merged_list = list(merged_scores.items())
        merged_list.sort(key=lambda x: x[1], reverse=True)
        
        return merged_list[:limit]
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简单实现）"""
        if not text1 or not text2:
            return 0.0
        
        # 转换为小写并分词
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # 计算Jaccard相似度
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_edge_similarity(self, node_id1: str, node_id2: str) -> float:
        """计算边相似度"""
        # 获取共同邻居
        neighbors1 = set(e.target_id for e in self.edges.get(node_id1, []))
        neighbors2 = set(e.target_id for e in self.edges.get(node_id2, []))
        
        if not neighbors1 or not neighbors2:
            return 0.0
        
        # 计算Jaccard相似度
        intersection = len(neighbors1.intersection(neighbors2))
        union = len(neighbors1.union(neighbors2))
        
        return intersection / union if union > 0 else 0.0
    
    def _clean_cache_for_node(self, node_id: str) -> None:
        """清理节点的缓存"""
        keys_to_remove = []
        for key in self._similarity_cache:
            if node_id in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._similarity_cache[key]