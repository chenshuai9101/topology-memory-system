"""
检索优化器 - 提升记忆检索的准确性和效率
负责优化搜索算法、排序策略和结果质量
"""

import time
import math
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
import logging
from collections import defaultdict, deque
import heapq

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """检索质量指标"""
    relevance_score: float  # 相关性 (0-1)
    precision_score: float  # 精确度 (0-1)
    recall_score: float  # 召回率 (0-1)
    diversity_score: float  # 多样性 (0-1)
    novelty_score: float  # 新颖性 (0-1)
    
    @property
    def overall_score(self) -> float:
        """综合质量分数"""
        weights = {
            'relevance': 0.4,
            'precision': 0.25,
            'recall': 0.15,
            'diversity': 0.1,
            'novelty': 0.1
        }
        return (
            self.relevance_score * weights['relevance'] +
            self.precision_score * weights['precision'] +
            self.recall_score * weights['recall'] +
            self.diversity_score * weights['diversity'] +
            self.novelty_score * weights['novelty']
        )


@dataclass
class RetrievalConfig:
    """检索配置"""
    hybrid_search_enabled: bool = True  # 是否启用混合搜索
    semantic_weight: float = 0.6  # 语义搜索权重
    keyword_weight: float = 0.3  # 关键词搜索权重
    temporal_weight: float = 0.1  # 时间权重
    max_results: int = 20  # 最大返回结果数
    diversity_threshold: float = 0.3  # 多样性阈值
    novelty_boost: float = 0.2  # 新颖性提升
    personalization_enabled: bool = True  # 是否启用个性化


class RetrievalOptimizer:
    """检索优化器核心类"""
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        """
        初始化检索优化器
        
        Args:
            config: 检索配置
        """
        self.config = config or RetrievalConfig()
        
        # 搜索历史记录
        self.search_history: deque = deque(maxlen=1000)
        
        # 个性化模型
        self.user_preferences: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # 缓存
        self._query_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._similarity_cache: Dict[Tuple[str, str], float] = {}
        
        # TF-IDF向量化器
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        logger.info("RetrievalOptimizer initialized")
    
    def optimize_retrieval(self, 
                          query: str,
                          nodes: List[Dict[str, Any]],
                          context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        优化记忆检索
        
        Args:
            query: 查询字符串
            nodes: 记忆节点列表
            context: 上下文信息
            
        Returns:
            List[Dict[str, Any]]: 优化后的检索结果
        """
        start_time = time.time()
        
        # 1. 分析查询
        query_analysis = self._analyze_query(query, context)
        
        # 2. 执行混合搜索
        search_results = self._hybrid_search(query, nodes, query_analysis)
        
        # 3. 优化排序
        optimized_results = self._optimize_ranking(
            results=search_results,
            query=query,
            query_analysis=query_analysis,
            context=context
        )
        
        # 4. 应用结果多样性
        diversified_results = self._apply_diversity(
            results=optimized_results,
            query_analysis=query_analysis
        )
        
        # 5. 个性化调整
        if self.config.personalization_enabled:
            user_id = context.get('user_id')
            if user_id:
                diversified_results = self._apply_personalization(
                    results=diversified_results,
                    user_id=user_id,
                    query=query
                )
        
        # 6. 限制结果数量
        final_results = diversified_results[:self.config.max_results]
        
        # 7. 计算检索质量
        quality_metrics = self._calculate_retrieval_quality(
            query=query,
            results=final_results,
            all_nodes=nodes,
            query_analysis=query_analysis
        )
        
        # 8. 更新搜索历史
        self._update_search_history(query, final_results, context)
        
        # 9. 更新个性化模型
        if self.config.personalization_enabled and context.get('user_id'):
            self._update_user_preferences(
                user_id=context['user_id'],
                query=query,
                results=final_results
            )
        
        processing_time = time.time() - start_time
        logger.info(f"Retrieval optimized: '{query}' -> {len(final_results)} results, "
                   f"time: {processing_time:.3f}s, "
                   f"quality: {quality_metrics.overall_score:.3f}")
        
        return final_results
    
    def _analyze_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析查询"""
        analysis = {
            'original_query': query,
            'length': len(query),
            'word_count': len(query.split()),
            'query_type': self._determine_query_type(query),
            'keywords': self._extract_keywords(query),
            'entities': self._extract_entities(query),
            'intent': self._infer_query_intent(query, context),
            'complexity': self._calculate_query_complexity(query),
            'context_relevance': self._calculate_context_relevance(query, context)
        }
        
        # 查询扩展
        analysis['expanded_queries'] = self._expand_query(query, analysis)
        
        return analysis
    
    def _hybrid_search(self, 
                      query: str,
                      nodes: List[Dict[str, Any]],
                      query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """执行混合搜索"""
        all_results = []
        
        # 1. 语义搜索（基于向量）
        semantic_results = self._semantic_search(query, nodes, query_analysis)
        for result in semantic_results:
            result['search_method'] = 'semantic'
            result['search_score'] *= self.config.semantic_weight
        
        # 2. 关键词搜索
        keyword_results = self._keyword_search(query, nodes, query_analysis)
        for result in keyword_results:
            result['search_method'] = 'keyword'
            result['search_score'] *= self.config.keyword_weight
        
        # 3. 时间相关搜索
        temporal_results = self._temporal_search(query, nodes, query_analysis)
        for result in temporal_results:
            result['search_method'] = 'temporal'
            result['search_score'] *= self.config.temporal_weight
        
        # 合并结果
        all_results.extend(semantic_results)
        all_results.extend(keyword_results)
        all_results.extend(temporal_results)
        
        # 去重和合并分数
        merged_results = self._merge_search_results(all_results)
        
        return merged_results
    
    def _optimize_ranking(self,
                         results: List[Dict[str, Any]],
                         query: str,
                         query_analysis: Dict[str, Any],
                         context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """优化结果排序"""
        if not results:
            return []
        
        scored_results = []
        
        for result in results:
            # 基础分数
            base_score = result.get('search_score', 0.0)
            
            # 计算优化分数
            optimized_score = self._calculate_optimized_score(
                result=result,
                query=query,
                query_analysis=query_analysis,
                context=context,
                base_score=base_score
            )
            
            # 创建优化后的结果
            optimized_result = result.copy()
            optimized_result['optimized_score'] = optimized_score
            optimized_result['ranking_factors'] = self._extract_ranking_factors(
                result=result,
                optimized_score=optimized_score,
                query_analysis=query_analysis
            )
            
            scored_results.append(optimized_result)
        
        # 按优化分数排序
        scored_results.sort(key=lambda x: x['optimized_score'], reverse=True)
        
        return scored_results
    
    def _apply_diversity(self,
                        results: List[Dict[str, Any]],
                        query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """应用结果多样性"""
        if len(results) <= 1:
            return results
        
        diversified_results = []
        selected_indices = set()
        
        # 第一轮：选择最相关的结果
        if results:
            diversified_results.append(results[0])
            selected_indices.add(0)
        
        # 后续轮次：基于多样性选择
        for i in range(1, min(len(results), self.config.max_results * 2)):
            best_index = -1
            best_diversity_score = -1
            
            for j in range(len(results)):
                if j in selected_indices:
                    continue
                
                # 计算多样性分数
                diversity_score = self._calculate_diversity_score(
                    candidate=results[j],
                    selected_results=[results[idx] for idx in selected_indices],
                    query_analysis=query_analysis
                )
                
                # 结合相关性和多样性
                combined_score = (
                    results[j]['optimized_score'] * (1 - self.config.diversity_threshold) +
                    diversity_score * self.config.diversity_threshold
                )
                
                if combined_score > best_diversity_score:
                    best_diversity_score = combined_score
                    best_index = j
            
            if best_index != -1:
                diversified_results.append(results[best_index])
                selected_indices.add(best_index)
        
        return diversified_results
    
    def _apply_personalization(self,
                              results: List[Dict[str, Any]],
                              user_id: str,
                              query: str) -> List[Dict[str, Any]]:
        """应用个性化调整"""
        if not results:
            return results
        
        # 获取用户偏好
        user_prefs = self.user_preferences.get(user_id, {})
        
        personalized_results = []
        
        for result in results:
            personalized_result = result.copy()
            
            # 计算个性化分数
            personalization_score = self._calculate_personalization_score(
                result=result,
                user_prefs=user_prefs,
                query=query
            )
            
            # 调整分数
            current_score = result.get('optimized_score', 0.0)
            personalized_score = current_score * (1 + personalization_score)
            
            personalized_result['personalized_score'] = personalized_score
            personalized_result['personalization_factor'] = personalization_score
            
            personalized_results.append(personalized_result)
        
        # 重新排序
        personalized_results.sort(key=lambda x: x['personalized_score'], reverse=True)
        
        return personalized_results
    
    def _calculate_retrieval_quality(self,
                                    query: str,
                                    results: List[Dict[str, Any]],
                                    all_nodes: List[Dict[str, Any]],
                                    query_analysis: Dict[str, Any]) -> RetrievalMetrics:
        """计算检索质量"""
        if not results:
            return RetrievalMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
        
        # 1. 相关性
        relevance_scores = []
        for result in results:
            relevance = self._calculate_result_relevance(result, query, query_analysis)
            relevance_scores.append(relevance)
        relevance_score = np.mean(relevance_scores) if relevance_scores else 0.0
        
        # 2. 精确度（需要真实相关结果，这里使用近似）
        precision_score = self._estimate_precision(results, query_analysis)
        
        # 3. 召回率（需要所有相关结果，这里使用近似）
        recall_score = self._estimate_recall(results, all_nodes, query_analysis)
        
        # 4. 多样性
        diversity_score = self._calculate_results_diversity(results)
        
        # 5. 新颖性
        novelty_score = self._calculate_results_novelty(results, query)
        
        return RetrievalMetrics(
            relevance_score=float(relevance_score),
            precision_score=float(precision_score),
            recall_score=float(recall_score),
            diversity_score=float(diversity_score),
            novelty_score=float(novelty_score)
        )
    
    def _semantic_search(self,
                        query: str,
                        nodes: List[Dict[str, Any]],
                        query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """语义搜索"""
        results = []
        
        # 生成查询向量（简化版）
        query_vector = self._generate_query_vector(query)
        
        if query_vector is None:
            return results
        
        for node in nodes:
            node_vector = node.get('vector')
            if node_vector is None:
                continue
            
            # 计算相似度
            similarity = self._calculate_vector_similarity(query_vector, node_vector)
            
            if similarity > 0.1:  # 阈值
                result = node.copy()
                result['search_score'] = similarity
                result['similarity'] = similarity
                results.append(result)
        
        # 按相似度排序
        results.sort(key=lambda x: x['search_score'], reverse=True)
        
        return results[:self.config.max_results * 2]
    
    def _keyword_search(self,
                       query: str,
                       nodes: List[Dict[str, Any]],
                       query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """关键词搜索"""
        results = []
        
        # 提取查询关键词
        keywords = query_analysis['keywords']
        if not keywords:
            return results
        
        for node in nodes:
            content = node.get('content', '')
            if not content:
                continue
            
            # 计算关键词匹配分数
            keyword_score = self._calculate_keyword_match(content, keywords)
            
            if keyword_score > 0:
                result = node.copy()
                result['search_score'] = keyword_score
                result['keyword_matches'] = self._get_matched_keywords(content, keywords)
                results.append(result)
        
        # 按关键词分数排序
        results.sort(key=lambda x: x['search_score'], reverse=True)
        
        return results[:self.config.max_results * 2]
    
    def _temporal_search(self,
                        query: str,
                        nodes: List[Dict[str, Any]],
                        query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """时间相关搜索"""
        results = []
        
        # 提取查询中的时间信息
        time_info = self._extract_time_info(query)
        
        for node in nodes:
            # 获取节点时间
            node_time = node.get('created_at')
            if not node_time:
                continue
            
            # 计算时间相关性
            temporal_score = self._calculate_temporal_relevance(node_time, time_info)
            
            if temporal_score > 0.1:  # 阈值
                result = node.copy()
                result['search_score'] = temporal_score
                result['temporal_relevance'] = temporal_score
                results.append(result)
        
        # 按时间相关性排序
        results.sort(key=lambda x: x['search_score'], reverse=True)
        
        return results[:self.config.max_results]
    
    def _calculate_optimized_score(self,
                                  result: Dict[str, Any],
                                  query: str,
                                  query_analysis: Dict[str, Any],
                                  context: Dict[str, Any],
                                  base_score: float) -> float:
        """计算优化分数"""
        optimized_score = base_score
        
        # 1. 新鲜度提升（新内容更重要）
        freshness_boost = self._calculate_freshness_boost(result)
        optimized_score *= (1 + freshness_boost)
        
        # 2. 重要性提升（重要内容更重要）
        importance_boost = result.get('importance', 0.0) * 0.1
        optimized_score *= (1 + importance_boost)
        
        # 3. 流行度提升（常访问的内容可能更相关）
        popularity_boost = self._calculate_popularity_boost(result)
        optimized_score *= (1 + popularity_boost)
        
        # 4. 上下文相关性
        context_relevance = self._calculate_result_context_relevance(result, context)
        optimized_score *= (1 + context_relevance * 0.2)
        
        # 5. 查询类型适配
        query_type_adjustment = self._adjust_for_query_type(result, query_analysis['query_type'])
        optimized_score *= query_type_adjustment
        
        return min(1.0, optimized_score)  # 确保不超过1.0
    
    def _calculate_diversity_score(self,
                                  candidate: Dict[str, Any],
                                  selected_results: List[Dict[str, Any]],
                                  query_analysis: Dict[str, Any]) -> float:
        """计算多样性分数"""
        if not selected_results:
            return 1.0  # 第一个结果具有最大多样性
        
        diversity_scores = []
        
        for selected in selected_results:
            # 计算与已选结果的差异
            difference = self._calculate_result_difference(candidate, selected, query_analysis)
            diversity_scores.append(difference)
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def _calculate_personalization_score(self,
                                        result: Dict[str, Any],
                                        user_prefs: Dict[str, float],
                                        query: str) -> float:
        """计算个性化分数"""
        score = 0.0
        
        # 1. 基于内容类型的偏好
        content