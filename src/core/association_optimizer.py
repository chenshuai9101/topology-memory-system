"""
关联优化器 - 提升记忆关联的准确性和清晰度
负责建立、优化和评估记忆之间的关联
"""

import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
import logging
from collections import defaultdict, deque
import heapq

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


@dataclass
class AssociationMetrics:
    """关联质量指标"""
    relevance_score: float  # 相关性 (0-1)
    confidence_score: float  # 置信度 (0-1)
    strength_score: float  # 强度 (0-1)
    clarity_score: float  # 清晰度 (0-1)
    novelty_score: float  # 新颖性 (0-1)
    
    @property
    def overall_score(self) -> float:
        """综合质量分数"""
        weights = {
            'relevance': 0.35,
            'confidence': 0.25,
            'strength': 0.15,
            'clarity': 0.15,
            'novelty': 0.1
        }
        return (
            self.relevance_score * weights['relevance'] +
            self.confidence_score * weights['confidence'] +
            self.strength_score * weights['strength'] +
            self.clarity_score * weights['clarity'] +
            self.novelty_score * weights['novelty']
        )


@dataclass
class AssociationConfig:
    """关联配置"""
    min_similarity_threshold: float = 0.3  # 最小相似度阈值
    max_associations_per_node: int = 15  # 每个节点最大关联数
    temporal_weight: float = 0.3  # 时间权重
    semantic_weight: float = 0.5  # 语义权重
    structural_weight: float = 0.2  # 结构权重
    decay_factor: float = 0.95  # 衰减因子
    clustering_enabled: bool = True  # 是否启用聚类
    community_detection: bool = True  # 是否启用社区发现


class AssociationOptimizer:
    """关联优化器核心类"""
    
    def __init__(self, config: Optional[AssociationConfig] = None):
        """
        初始化关联优化器
        
        Args:
            config: 关联配置
        """
        self.config = config or AssociationConfig()
        
        # 关联缓存
        self._association_cache: Dict[Tuple[str, str], float] = {}
        self._similarity_cache: Dict[Tuple[str, str], float] = {}
        
        # 时间衰减缓存
        self._temporal_decay_cache: Dict[str, float] = {}
        
        logger.info("AssociationOptimizer initialized")
    
    def optimize_associations(self, 
                             nodes: List[Dict[str, Any]],
                             existing_edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        优化记忆关联
        
        Args:
            nodes: 记忆节点列表
            existing_edges: 现有边列表
            
        Returns:
            List[Dict[str, Any]]: 优化后的关联列表
        """
        start_time = time.time()
        
        # 1. 分析现有关联
        existing_analysis = self._analyze_existing_associations(existing_edges)
        
        # 2. 计算节点相似度矩阵
        similarity_matrix = self._calculate_similarity_matrix(nodes)
        
        # 3. 发现新关联
        new_associations = self._discover_new_associations(
            nodes=nodes,
            similarity_matrix=similarity_matrix,
            existing_edges=existing_edges
        )
        
        # 4. 优化现有关联
        optimized_edges = self._optimize_existing_associations(
            existing_edges=existing_edges,
            similarity_matrix=similarity_matrix,
            existing_analysis=existing_analysis
        )
        
        # 5. 合并新旧关联
        all_associations = self._merge_associations(optimized_edges, new_associations)
        
        # 6. 应用关联限制
        final_associations = self._apply_association_limits(all_associations, nodes)
        
        # 7. 计算关联质量
        quality_report = self._calculate_association_quality(final_associations, nodes)
        
        processing_time = time.time() - start_time
        logger.info(f"Association optimization completed: {processing_time:.3f}s, "
                   f"{len(final_associations)} associations, "
                   f"quality: {quality_report['overall_quality']:.3f}")
        
        return final_associations
    
    def _analyze_existing_associations(self, edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析现有关联"""
        analysis = {
            'total_edges': len(edges),
            'edge_weights': [],
            'node_degrees': defaultdict(int),
            'relationship_types': defaultdict(int),
            'temporal_patterns': [],
            'strength_distribution': []
        }
        
        for edge in edges:
            # 节点度
            analysis['node_degrees'][edge['source_id']] += 1
            analysis['node_degrees'][edge['target_id']] += 1
            
            # 边权重
            weight = edge.get('weight', 1.0)
            analysis['edge_weights'].append(weight)
            
            # 关系类型
            rel_type = edge.get('relationship', 'related')
            analysis['relationship_types'][rel_type] += 1
            
            # 时间模式
            if 'created_at' in edge:
                analysis['temporal_patterns'].append(edge['created_at'])
            
            # 强度分布
            analysis['strength_distribution'].append(weight)
        
        # 计算统计信息
        if analysis['edge_weights']:
            analysis['avg_weight'] = np.mean(analysis['edge_weights'])
            analysis['std_weight'] = np.std(analysis['edge_weights'])
            analysis['max_weight'] = np.max(analysis['edge_weights'])
            analysis['min_weight'] = np.min(analysis['edge_weights'])
        
        # 节点度分布
        if analysis['node_degrees']:
            degrees = list(analysis['node_degrees'].values())
            analysis['avg_degree'] = np.mean(degrees)
            analysis['max_degree'] = np.max(degrees)
            analysis['min_degree'] = np.min(degrees)
        
        return analysis
    
    def _calculate_similarity_matrix(self, nodes: List[Dict[str, Any]]) -> np.ndarray:
        """计算相似度矩阵"""
        n = len(nodes)
        similarity_matrix = np.zeros((n, n))
        
        # 提取向量
        vectors = []
        valid_indices = []
        
        for i, node in enumerate(nodes):
            vector = node.get('vector')
            if vector is not None and len(vector) > 0:
                vectors.append(vector)
                valid_indices.append(i)
        
        if not vectors:
            return similarity_matrix
        
        vectors_array = np.array(vectors)
        
        # 计算余弦相似度
        if len(vectors) > 1:
            similarities = cosine_similarity(vectors_array)
            
            # 填充到完整矩阵
            for idx_i, i in enumerate(valid_indices):
                for idx_j, j in enumerate(valid_indices):
                    if i != j:
                        similarity_matrix[i][j] = similarities[idx_i][idx_j]
        
        return similarity_matrix
    
    def _discover_new_associations(self, 
                                  nodes: List[Dict[str, Any]],
                                  similarity_matrix: np.ndarray,
                                  existing_edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """发现新关联"""
        new_associations = []
        
        # 构建现有关联集合
        existing_set = set()
        for edge in existing_edges:
            key = (edge['source_id'], edge['target_id'])
            existing_set.add(key)
            existing_set.add((edge['target_id'], edge['source_id']))  # 无向图
        
        n = len(nodes)
        
        # 寻找高相似度但未关联的节点对
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                
                if similarity < self.config.min_similarity_threshold:
                    continue
                
                node_i = nodes[i]
                node_j = nodes[j]
                
                # 检查是否已存在关联
                if (node_i['node_id'], node_j['node_id']) in existing_set:
                    continue
                
                # 计算综合关联分数
                association_score = self._calculate_association_score(
                    node_i=node_i,
                    node_j=node_j,
                    similarity=similarity,
                    existing_edges=existing_edges
                )
                
                if association_score > 0.5:  # 阈值
                    new_association = {
                        'source_id': node_i['node_id'],
                        'target_id': node_j['node_id'],
                        'relationship': self._determine_relationship_type(node_i, node_j),
                        'weight': association_score,
                        'confidence': similarity,
                        'discovery_method': 'similarity',
                        'created_at': datetime.now().isoformat(),
                        'metadata': {
                            'similarity': float(similarity),
                            'temporal_relevance': self._calculate_temporal_relevance(node_i, node_j),
                            'semantic_relevance': similarity
                        }
                    }
                    
                    new_associations.append(new_association)
        
        # 基于聚类发现关联
        if self.config.clustering_enabled and len(nodes) > 10:
            cluster_associations = self._discover_cluster_associations(nodes, similarity_matrix)
            new_associations.extend(cluster_associations)
        
        # 基于社区发现关联
        if self.config.community_detection and len(existing_edges) > 20:
            community_associations = self._discover_community_associations(nodes, existing_edges)
            new_associations.extend(community_associations)
        
        return new_associations
    
    def _optimize_existing_associations(self,
                                       existing_edges: List[Dict[str, Any]],
                                       similarity_matrix: np.ndarray,
                                       existing_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """优化现有关联"""
        optimized_edges = []
        
        # 构建节点ID到索引的映射
        node_id_to_index = {}
        # 这里需要节点列表，但函数参数中没有，需要调整
        
        for edge in existing_edges:
            optimized_edge = edge.copy()
            
            # 1. 更新权重（基于相似度和时间衰减）
            current_weight = edge.get('weight', 1.0)
            
            # 计算时间衰减
            if 'created_at' in edge:
                age = self._calculate_edge_age(edge['created_at'])
                temporal_decay = self.config.decay_factor ** age
            else:
                temporal_decay = 1.0
            
            # 获取相似度（如果有）
            similarity = self._get_edge_similarity(edge, similarity_matrix, node_id_to_index)
            
            # 计算新权重
            new_weight = self._calculate_optimized_weight(
                current_weight=current_weight,
                similarity=similarity,
                temporal_decay=temporal_decay,
                existing_analysis=existing_analysis
            )
            
            optimized_edge['weight'] = new_weight
            
            # 2. 更新置信度
            confidence = edge.get('confidence', 0.5)
            new_confidence = self._update_confidence(
                current_confidence=confidence,
                similarity=similarity,
                temporal_decay=temporal_decay
            )
            optimized_edge['confidence'] = new_confidence
            
            # 3. 更新关系类型（如果需要）
            if 'relationship' not in edge or edge['relationship'] == 'related':
                # 尝试确定更具体的关系类型
                pass
            
            # 4. 添加优化元数据
            if 'metadata' not in optimized_edge:
                optimized_edge['metadata'] = {}
            
            optimized_edge['metadata']['optimized_at'] = datetime.now().isoformat()
            optimized_edge['metadata']['optimization_score'] = self._calculate_optimization_score(
                old_weight=current_weight,
                new_weight=new_weight,
                similarity=similarity
            )
            
            optimized_edges.append(optimized_edge)
        
        return optimized_edges
    
    def _merge_associations(self, 
                           optimized_edges: List[Dict[str, Any]],
                           new_associations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并新旧关联"""
        all_associations = []
        
        # 使用集合跟踪已处理的关联
        processed_keys = set()
        
        # 首先添加优化后的现有关联
        for edge in optimized_edges:
            key = (edge['source_id'], edge['target_id'])
            if key not in processed_keys:
                all_associations.append(edge)
                processed_keys.add(key)
        
        # 然后添加新关联（避免重复）
        for association in new_associations:
            key = (association['source_id'], association['target_id'])
            reverse_key = (association['target_id'], association['source_id'])
            
            if key not in processed_keys and reverse_key not in processed_keys:
                all_associations.append(association)
                processed_keys.add(key)
        
        return all_associations
    
    def _apply_association_limits(self, 
                                 associations: List[Dict[str, Any]],
                                 nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """应用关联限制"""
        # 统计每个节点的关联数
        node_degree = defaultdict(int)
        for edge in associations:
            node_degree[edge['source_id']] += 1
            node_degree[edge['target_id']] += 1
        
        # 构建节点ID到节点的映射
        node_map = {node['node_id']: node for node in nodes}
        
        # 对每个节点的关联进行排序和限制
        node_edges = defaultdict(list)
        for edge in associations:
            node_edges[edge['source_id']].append(edge)
            node_edges[edge['target_id']].append(edge)
        
        final_associations = []
        processed_edges = set()
        
        for node_id, edges in node_edges.items():
            if len(edges) <= self.config.max_associations_per_node:
                # 未超过限制，全部保留
                for edge in edges:
                    edge_key = (edge['source_id'], edge['target_id'])
                    if edge_key not in processed_edges:
                        final_associations.append(edge)
                        processed_edges.add(edge_key)
            else:
                # 超过限制，选择最重要的关联
                # 按权重排序
                sorted_edges = sorted(edges, key=lambda e: e.get('weight', 0), reverse=True)
                
                # 保留前N个
                for edge in sorted_edges[:self.config.max_associations_per_node]:
                    edge_key = (edge['source_id'], edge['target_id'])
                    if edge_key not in processed_edges:
                        final_associations.append(edge)
                        processed_edges.add(edge_key)
        
        return final_associations
    
    def _calculate_association_quality(self, 
                                      associations: List[Dict[str, Any]],
                                      nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算关联质量"""
        if not associations:
            return {'overall_quality': 0.0}
        
        quality_scores = []
        
        for edge in associations:
            metrics = self._calculate_edge_metrics(edge, nodes)
            quality_scores.append(metrics.overall_score)
        
        overall_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        return {
            'overall_quality': float(overall_quality),
            'avg_quality': float(np.mean(quality_scores)) if quality_scores else 0.0,
            'std_quality': float(np.std(quality_scores)) if len(quality_scores) > 1 else 0.0,
            'min_quality': float(np.min(quality_scores)) if quality_scores else 0.0,
            'max_quality': float(np.max(quality_scores)) if quality_scores else 0.0,
            'total_associations': len(associations),
            'quality_distribution': self._calculate_quality_distribution(quality_scores)
        }
    
    def _calculate_association_score(self, 
                                    node_i: Dict[str, Any],
                                    node_j: Dict[str, Any],
                                    similarity: float,
                                    existing_edges: List[Dict[str, Any]]) -> float:
        """计算关联分数"""
        score = 0.0
        
        # 1. 语义相似度
        semantic_score = similarity * self.config.semantic_weight
        
        # 2. 时间相关性
        temporal_score = self._calculate_temporal_relevance(node_i, node_j) * self.config.temporal_weight
        
        # 3. 结构相关性
        structural_score = self._calculate_structural_relevance(
            node_i, node_j, existing_edges
        ) * self.config.structural_weight
        
        score = semantic_score + temporal_score + structural_score
        
        # 4. 应用衰减（如果节点较旧）
        age_i = self._calculate_node_age(node_i)
        age_j = self._calculate_node_age(node_j)
        avg_age = (age_i + age_j) / 2
        age_decay = self.config.decay_factor ** avg_age
        
        return score * age_decay
    
    def _calculate_temporal_relevance(self, node_i: Dict[str, Any], node_j: Dict[str, Any]) -> float:
        """计算时间相关性"""
        time_i = node_i.get('created_at')
        time_j = node_j.get('created_at')
        
        if not time_i or not time_j:
            return 0.5  # 默认值
        
        try:
            # 解析时间
            if isinstance(time_i, str):
                dt_i =