"""
混合搜索服务
结合向量搜索和关键词搜索，提供更准确的搜索结果
"""

import logging
import time
import re
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import numpy as np

from ..models.vector_models import (
    HybridSearchQuery, SearchResult, SearchResponse,
    DistanceMetric
)
from .vector_encoder import VectorEncoder
from .qdrant_service import QdrantService

logger = logging.getLogger(__name__)


class HybridSearchService:
    """混合搜索服务"""
    
    def __init__(
        self,
        qdrant_service: QdrantService,
        vector_encoder: VectorEncoder,
        collection_name: str = "topology_memory"
    ):
        """
        初始化混合搜索服务
        
        Args:
            qdrant_service: Qdrant服务实例
            vector_encoder: 向量编码器实例
            collection_name: 集合名称
        """
        self.qdrant_service = qdrant_service
        self.vector_encoder = vector_encoder
        self.collection_name = collection_name
        
        logger.info("初始化混合搜索服务")
    
    def hybrid_search(self, query: HybridSearchQuery) -> SearchResponse:
        """
        执行混合搜索
        
        Args:
            query: 混合搜索查询
            
        Returns:
            搜索结果
        """
        start_time = time.time()
        
        try:
            # 1. 向量搜索
            vector_results = self._vector_search(query)
            
            # 2. 关键词搜索（如果有关键词）
            keyword_results = self._keyword_search(query)
            
            # 3. 融合结果
            fused_results = self._fuse_results(
                vector_results, keyword_results, query
            )
            
            # 4. 应用时间衰减和重要性加权
            ranked_results = self._apply_ranking_factors(fused_results, query)
            
            query_time_ms = (time.time() - start_time) * 1000
            
            return SearchResponse(
                results=ranked_results[:query.limit],
                total=len(ranked_results),
                query_time_ms=query_time_ms,
                query_type="hybrid"
            )
            
        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
            query_time_ms = (time.time() - start_time) * 1000
            return SearchResponse(
                results=[],
                total=0,
                query_time_ms=query_time_ms,
                query_type="hybrid"
            )
    
    def _vector_search(self, query: HybridSearchQuery) -> List[Dict[str, Any]]:
        """执行向量搜索"""
        try:
            # 编码查询文本为向量
            query_vector = self.vector_encoder.encode_text(query.query_text)
            
            # 构建向量搜索查询
            from ..models.vector_models import SearchQuery
            vector_query = SearchQuery(
                query_vector=query_vector,
                limit=query.limit * 2,  # 获取更多结果用于融合
                threshold=0.3,  # 较低的阈值以获取更多相关结果
                distance_metric=DistanceMetric.COSINE,
                with_payload=True,
                with_vector=False,
                filter_conditions=query.filter_conditions
            )
            
            # 执行搜索
            response = self.qdrant_service.search(vector_query, self.collection_name)
            
            # 转换结果格式
            results = []
            for result in response.results:
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                    "metadata": result.metadata,
                    "search_type": "vector"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    def _keyword_search(self, query: HybridSearchQuery) -> List[Dict[str, Any]]:
        """执行关键词搜索"""
        if not query.keywords:
            return []
        
        try:
            # 构建关键词过滤条件
            filter_conditions = query.filter_conditions or {}
            
            # 添加关键词匹配条件
            keyword_filter = {
                "must": [
                    {
                        "key": "text",
                        "match": {"any": query.keywords}
                    }
                ]
            }
            
            # 合并过滤条件
            if "must" in filter_conditions:
                filter_conditions["must"].extend(keyword_filter["must"])
            else:
                filter_conditions.update(keyword_filter)
            
            # 使用Qdrant的scroll功能进行关键词搜索
            scroll_results = self.qdrant_service.scroll_points(
                collection_name=self.collection_name,
                limit=query.limit * 2,
                with_payload=True,
                with_vector=False
            )
            
            # 计算关键词匹配分数
            results = []
            for point in scroll_results["points"]:
                if "text" in point.payload:
                    text = point.payload["text"]
                    keyword_score = self._calculate_keyword_score(text, query.keywords)
                    
                    if keyword_score > 0:
                        results.append({
                            "id": point.id,
                            "score": keyword_score,
                            "payload": point.payload,
                            "metadata": {},
                            "search_type": "keyword"
                        })
            
            # 按分数排序
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return results[:query.limit * 2]  # 限制结果数量
            
        except Exception as e:
            logger.error(f"关键词搜索失败: {e}")
            return []
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """计算关键词匹配分数"""
        if not text or not keywords:
            return 0.0
        
        text_lower = text.lower()
        total_score = 0.0
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # 计算出现频率
            count = text_lower.count(keyword_lower)
            if count > 0:
                # 基础分数：出现次数 * 0.1
                base_score = count * 0.1
                
                # 检查是否完整匹配（不是子字符串）
                words = re.findall(r'\b\w+\b', text_lower)
                if keyword_lower in words:
                    base_score += 0.2  # 完整匹配加分
                
                total_score += base_score
        
        # 归一化到0-1
        max_possible_score = len(keywords) * 0.3  # 每个关键词最大0.3分
        if max_possible_score > 0:
            return min(total_score / max_possible_score, 1.0)
        
        return 0.0
    
    def _fuse_results(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        query: HybridSearchQuery
    ) -> List[Dict[str, Any]]:
        """融合向量和关键词搜索结果"""
        # 创建结果字典，按ID索引
        results_dict = {}
        
        # 添加向量搜索结果
        for result in vector_results:
            result_id = result["id"]
            if result_id not in results_dict:
                results_dict[result_id] = {
                    "id": result_id,
                    "vector_score": result["score"],
                    "keyword_score": 0.0,
                    "payload": result["payload"],
                    "metadata": result["metadata"]
                }
            else:
                results_dict[result_id]["vector_score"] = result["score"]
        
        # 添加关键词搜索结果
        for result in keyword_results:
            result_id = result["id"]
            if result_id in results_dict:
                results_dict[result_id]["keyword_score"] = result["score"]
            else:
                results_dict[result_id] = {
                    "id": result_id,
                    "vector_score": 0.0,
                    "keyword_score": result["score"],
                    "payload": result["payload"],
                    "metadata": result["metadata"]
                }
        
        # 计算融合分数
        fused_results = []
        for result_id, result_data in results_dict.items():
            # 加权融合
            fused_score = (
                result_data["vector_score"] * query.vector_weight +
                result_data["keyword_score"] * query.keyword_weight
            )
            
            fused_results.append({
                "id": result_id,
                "score": fused_score,
                "payload": result_data["payload"],
                "metadata": result_data["metadata"],
                "vector_score": result_data["vector_score"],
                "keyword_score": result_data["keyword_score"]
            })
        
        # 按融合分数排序
        fused_results.sort(key=lambda x: x["score"], reverse=True)
        
        return fused_results
    
    def _apply_ranking_factors(
        self,
        results: List[Dict[str, Any]],
        query: HybridSearchQuery
    ) -> List[Dict[str, Any]]:
        """应用时间衰减和重要性加权"""
        ranked_results = []
        current_time = datetime.now()
        
        for result in results:
            # 复制结果
            ranked_result = result.copy()
            base_score = result["score"]
            
            # 应用时间衰减
            time_decayed_score = self._apply_time_decay(
                base_score, result["payload"], current_time, query.time_decay_factor
            )
            
            # 应用重要性加权
            final_score = self._apply_importance_weight(
                time_decayed_score, result["payload"], query.importance_weight
            )
            
            ranked_result["score"] = final_score
            ranked_result["metadata"]["time_decay_factor"] = query.time_decay_factor
            ranked_result["metadata"]["importance_weight"] = query.importance_weight
            
            ranked_results.append(ranked_result)
        
        # 重新排序
        ranked_results.sort(key=lambda x: x["score"], reverse=True)
        
        return ranked_results
    
    def _apply_time_decay(
        self,
        score: float,
        payload: Dict[str, Any],
        current_time: datetime,
        decay_factor: float
    ) -> float:
        """应用时间衰减"""
        if decay_factor <= 0:
            return score
        
        # 从payload中提取时间信息
        timestamp = None
        
        # 尝试不同的时间字段
        time_fields = ["timestamp", "created_at", "updated_at", "date"]
        for field in time_fields:
            if field in payload:
                timestamp_str = payload[field]
                try:
                    if isinstance(timestamp_str, str):
                        # 尝试解析时间字符串
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    elif isinstance(timestamp_str, (int, float)):
                        # 假设是Unix时间戳
                        timestamp = datetime.fromtimestamp(timestamp_str)
                    break
                except (ValueError, TypeError):
                    continue
        
        if not timestamp:
            return score
        
        # 计算时间差（天）
        time_diff = (current_time - timestamp).total_seconds() / (24 * 3600)
        
        # 应用指数衰减
        decayed_score = score * np.exp(-decay_factor * time_diff)
        
        return decayed_score
    
    def _apply_importance_weight(
        self,
        score: float,
        payload: Dict[str, Any],
        importance_weight: float
    ) -> float:
        """应用重要性加权"""
        if importance_weight <= 1.0:
            return score
        
        # 从payload中提取重要性信息
        importance = 1.0
        
        # 检查重要性字段
        importance_fields = ["importance", "priority", "weight", "relevance"]
        for field in importance_fields:
            if field in payload:
                try:
                    field_value = payload[field]
                    if isinstance(field_value, (int, float)):
                        importance = max(0.1, min(field_value, 10.0))  # 限制在0.1-10之间
                    break
                except (ValueError, TypeError):
                    continue
        
        # 应用重要性加权
        weighted_score = score * (importance * importance_weight)
        
        # 归一化到0-1
        return min(weighted_score, 1.0)
    
    def semantic_search(
        self,
        query_text: str,
        limit: int = 10,
        threshold: float = 0.5,
        collection_name: Optional[str] = None
    ) -> SearchResponse:
        """
        语义搜索（简化接口）
        
        Args:
            query_text: 查询文本
            limit: 返回结果数量
            threshold: 相似度阈值
            collection_name: 集合名称
            
        Returns:
            搜索结果
        """
        collection = collection_name or self.collection_name
        
        # 创建混合搜索查询
        query = HybridSearchQuery(
            query_text=query_text,
            limit=limit,
            vector_weight=1.0,  # 只使用向量搜索
            keyword_weight=0.0,
            time_decay_factor=0.0,
            importance_weight=1.0
        )
        
        return self.hybrid_search(query)
    
    def keyword_boosted_search(
        self,
        query_text: str,
        keywords: List[str],
        limit: int = 10,
        keyword_weight: float = 0.4,
        collection_name: Optional[str] = None
    ) -> SearchResponse:
        """
        关键词增强搜索
        
        Args:
            query_text: 查询文本
            keywords: 关键词列表
            limit: 返回结果数量
            keyword_weight: 关键词权重
            collection_name: 集合名称
            
        Returns:
            搜索结果
        """
        collection = collection_name or self.collection_name
        
        # 创建混合搜索查询
        query = HybridSearchQuery(
            query_text=query_text,
            keywords=keywords,
            limit=limit,
            vector_weight=1.0 - keyword_weight,
            keyword_weight=keyword_weight,
            time_decay_factor=0.1,
            importance_weight=1.0
        )
        
        return self.hybrid_search(query)


# 工厂函数
def create_hybrid_search_service(
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    model_type: str = "multilingual_mini",
    collection_name: str = "topology_memory"
) -> HybridSearchService:
    """
    创建混合搜索服务
    
    Args:
        qdrant_host: Qdrant主机地址
        qdrant_port: Qdrant端口
        model_type: 向量编码器模型类型
        collection_name: 集合名称
        
    Returns:
        混合搜索服务实例
    """
    # 创建Qdrant服务
    qdrant_service = QdrantService(host=qdrant_host, port=qdrant_port)
    
    # 创建向量编码器
    vector_encoder = VectorEncoder(
        model_name=model_type,
        device="cpu"
    )
    
    # 创建混合搜索服务
    return HybridSearchService(
        qdrant_service=qdrant_service,
        vector_encoder=vector_encoder,
        collection_name=collection_name
    )