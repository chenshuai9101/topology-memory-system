"""
拓扑记忆上下文管理器 - 核心引擎
集成上下文管理、记忆管理和拓扑算法的统一接口
"""

import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass

from .context_manager import ContextManager, ContextCreate, ContextUpdate, ContextResponse
from .memory_manager import MemoryManager, MemoryNode, MemoryEdge
from .topology_algorithms import TopologyAlgorithms, TopologyConfig
from ..api.schemas import TopologyQuery, TopologyResponse, PaginationParams, PaginatedResponse


logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """引擎配置"""
    # 上下文管理器配置
    max_contexts_per_session: int = 100
    context_cleanup_interval: int = 300
    
    # 记忆管理器配置
    max_memory_nodes: int = 10000
    max_edges_per_node: int = 50
    vector_dimension: int = 384
    memory_similarity_threshold: float = 0.5
    
    # 拓扑算法配置
    topology_min_similarity: float = 0.3
    topology_max_edges_per_node: int = 10
    enable_community_detection: bool = True
    
    # 性能配置
    enable_caching: bool = True
    cache_ttl: int = 300
    batch_size: int = 100


class TopologyMemoryEngine:
    """拓扑记忆核心引擎"""
    
    def __init__(self, config: Optional[EngineConfig] = None):
        """
        初始化核心引擎
        
        Args:
            config: 引擎配置
        """
        self.config = config or EngineConfig()
        
        # 初始化组件
        self.context_manager = ContextManager(
            max_contexts_per_session=self.config.max_contexts_per_session,
            cleanup_interval=self.config.context_cleanup_interval
        )
        
        self.memory_manager = MemoryManager(
            max_nodes=self.config.max_memory_nodes,
            max_edges_per_node=self.config.max_edges_per_node,
            vector_dimension=self.config.vector_dimension,
            similarity_threshold=self.config.memory_similarity_threshold
        )
        
        topology_config = TopologyConfig(
            min_similarity=self.config.topology_min_similarity,
            max_edges_per_node=self.config.topology_max_edges_per_node,
            community_detection=self.config.enable_community_detection
        )
        
        self.topology_algorithms = TopologyAlgorithms(topology_config)
        
        # 缓存
        self._cache = {}
        self._cache_ttl = self.config.cache_ttl
        
        # 性能监控
        self._performance_stats = {
            "total_requests": 0,
            "avg_response_time": 0.0,
            "last_reset": time.time()
        }
        
        logger.info("TopologyMemoryEngine initialized")
    
    # 上下文管理接口
    
    def create_context(self, context_data: ContextCreate) -> ContextResponse:
        """
        创建上下文
        
        Args:
            context_data: 上下文创建数据
            
        Returns:
            ContextResponse: 创建的上下文
        """
        start_time = time.time()
        
        try:
            result = self.context_manager.create_context(context_data)
            
            # 可选：将上下文保存为记忆节点
            if context_data.context_type in ["conversation", "memory", "knowledge"]:
                self._context_to_memory_node(context_data, result.id)
            
            self._update_performance_stats(start_time)
            return result
            
        except Exception as e:
            logger.error(f"Failed to create context: {e}")
            raise
    
    def get_context(self, session_id: str, context_id: str) -> Optional[ContextResponse]:
        """
        获取上下文
        
        Args:
            session_id: 会话ID
            context_id: 上下文ID
            
        Returns:
            Optional[ContextResponse]: 上下文或None
        """
        start_time = time.time()
        
        try:
            result = self.context_manager.get_context(session_id, context_id)
            self._update_performance_stats(start_time)
            return result
            
        except Exception as e:
            logger.error(f"Failed to get context: {e}")
            return None
    
    def update_context(self, session_id: str, context_id: str, 
                      update_data: ContextUpdate) -> Optional[ContextResponse]:
        """
        更新上下文
        
        Args:
            session_id: 会话ID
            context_id: 上下文ID
            update_data: 更新数据
            
        Returns:
            Optional[ContextResponse]: 更新后的上下文或None
        """
        start_time = time.time()
        
        try:
            result = self.context_manager.update_context(session_id, context_id, update_data)
            self._update_performance_stats(start_time)
            return result
            
        except Exception as e:
            logger.error(f"Failed to update context: {e}")
            return None
    
    def delete_context(self, session_id: str, context_id: str) -> bool:
        """
        删除上下文
        
        Args:
            session_id: 会话ID
            context_id: 上下文ID
            
        Returns:
            bool: 是否成功删除
        """
        start_time = time.time()
        
        try:
            result = self.context_manager.delete_context(session_id, context_id)
            self._update_performance_stats(start_time)
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete context: {e}")
            return False
    
    def list_contexts(self, session_id: str, 
                     pagination: PaginationParams) -> PaginatedResponse:
        """
        列出上下文
        
        Args:
            session_id: 会话ID
            pagination: 分页参数
            
        Returns:
            PaginatedResponse: 分页响应
        """
        start_time = time.time()
        
        try:
            contexts, total = self.context_manager.list_contexts(
                session_id, 
                page=pagination.page, 
                page_size=pagination.page_size
            )
            
            total_pages = (total + pagination.page_size - 1) // pagination.page_size
            
            response = PaginatedResponse(
                items=contexts,
                total=total,
                page=pagination.page,
                page_size=pagination.page_size,
                total_pages=total_pages
            )
            
            self._update_performance_stats(start_time)
            return response
            
        except Exception as e:
            logger.error(f"Failed to list contexts: {e}")
            return PaginatedResponse(
                items=[],
                total=0,
                page=pagination.page,
                page_size=pagination.page_size,
                total_pages=0
            )
    
    # 记忆管理接口
    
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
        start_time = time.time()
        
        try:
            node_id = self.memory_manager.create_memory_node(content, vector, metadata)
            self._update_performance_stats(start_time)
            return node_id
            
        except Exception as e:
            logger.error(f"Failed to create memory node: {e}")
            raise
    
    def get_memory_node(self, node_id: str) -> Optional[MemoryNode]:
        """
        获取记忆节点
        
        Args:
            node_id: 节点ID
            
        Returns:
            Optional[MemoryNode]: 记忆节点或None
        """
        start_time = time.time()
        
        try:
            result = self.memory_manager.get_memory_node(node_id)
            self._update_performance_stats(start_time)
            return result
            
        except Exception as e:
            logger.error(f"Failed to get memory node: {e}")
            return None
    
    def link_memory_nodes(self, source_id: str, target_id: str,
                         relationship: str = "related_to",
                         weight: float = 1.0) -> bool:
        """
        链接记忆节点
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            relationship: 关系类型
            weight: 权重
            
        Returns:
            bool: 是否成功链接
        """
        start_time = time.time()
        
        try:
            result = self.memory_manager.link_nodes(
                source_id, target_id, relationship, weight
            )
            self._update_performance_stats(start_time)
            return result
            
        except Exception as e:
            logger.error(f"Failed to link memory nodes: {e}")
            return False
    
    def search_memory(self, query: TopologyQuery) -> TopologyResponse:
        """
        搜索记忆
        
        Args:
            query: 拓扑查询
            
        Returns:
            TopologyResponse: 查询结果
        """
        start_time = time.time()
        
        try:
            # 搜索记忆节点
            memory_nodes = self.memory_manager.search_memory(
                query=query.query,
                limit=query.limit,
                threshold=query.threshold,
                include_vectors=query.include_vectors
            )
            
            # 获取相关边
            edges = []
            if memory_nodes:
                # 获取第一个节点的相关边作为示例
                related_nodes, related_edges = self.memory_manager.get_related_nodes(
                    memory_nodes[0].node_id,
                    depth=1,
                    limit=query.limit
                )
                edges = related_edges
            
            query_time = time.time() - start_time
            
            response = TopologyResponse(
                nodes=memory_nodes,
                edges=edges,
                query_time=query_time,
                total_nodes=len(memory_nodes),
                total_edges=len(edges)
            )
            
            self._update_performance_stats(start_time)
            return response
            
        except Exception as e:
            logger.error(f"Failed to search memory: {e}")
            return TopologyResponse(
                nodes=[],
                edges=[],
                query_time=time.time() - start_time,
                total_nodes=0,
                total_edges=0
            )
    
    # 拓扑算法接口
    
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
        start_time = time.time()
        
        try:
            result = self.memory_manager.build_topology(
                center_node_id, max_nodes, min_similarity
            )
            self._update_performance_stats(start_time)
            return result
            
        except Exception as e:
            logger.error(f"Failed to build topology: {e}")
            return [], []
    
    def analyze_topology(self, center_node_id: str) -> Dict[str, Any]:
        """
        分析拓扑结构
        
        Args:
            center_node_id: 中心节点ID
            
        Returns:
            Dict[str, Any]: 拓扑分析结果
        """
        start_time = time.time()
        
        try:
            # 构建拓扑
            nodes, edges = self.build_topology(center_node_id)
            
            # 构建图
            graph = self.topology_algorithms.build_topology_graph(nodes, edges)
            
            # 分析拓扑
            analysis = self.topology_algorithms.analyze_topology(graph)
            
            self._update_performance_stats(start_time)
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze topology: {e}")
            return {"error": str(e)}
    
    def find_related_contexts(self, context_id: str,
                             session_id: str,
                             limit: int = 10) -> List[ContextResponse]:
        """
        查找相关上下文
        
        Args:
            context_id: 上下文ID
            session_id: 会话ID
            limit: 返回数量限制
            
        Returns:
            List[ContextResponse]: 相关上下文列表
        """
        start_time = time.time()
        
        try:
            # 获取上下文
            context = self.get_context(session_id, context_id)
            if not context:
                return []
            
            # 搜索相关记忆节点
            query = TopologyQuery(
                query=context.content.get("text", "")[:100],  # 使用前100个字符
                limit=limit
            )
            
            memory_results = self.search_memory(query)
            
            # 通过记忆节点找到相关上下文
            related_contexts = []
            for node in memory_results.nodes:
                # 这里需要实现从记忆节点到上下文的映射
                # 简化实现：返回当前会话的其他上下文
                pass
            
            # 简化实现：返回当前会话的其他上下文
            session_contexts = self.context_manager.get_session_contexts(session_id)
            related_contexts = [
                ctx for ctx in session_contexts 
                if ctx.id != context_id
            ][:limit]
            
            self._update_performance_stats(start_time)
            return related_contexts
            
        except Exception as e:
            logger.error(f"Failed to find related contexts: {e}")
            return []
    
    # 系统管理接口
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取系统统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        context_stats = self.context_manager.get_stats()
        memory_stats = self.memory_manager.get_stats()
        
        stats = {
            "context_manager": context_stats,
            "memory_manager": memory_stats,
            "performance": self._performance_stats,
            "engine_config": {
                "max_contexts_per_session": self.config.max_contexts_per_session,
                "max_memory_nodes": self.config.max_memory_nodes,
                "vector_dimension": self.config.vector_dimension
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return stats
    
    def clear_all(self) -> None:
        """清除所有数据"""
        self.context_manager.clear_all()
        self.memory_manager.clear_all()
        self._cache.clear()
        
        logger.info("Cleared all engine data")
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            Dict[str, Any]: 健康状态
        """
        try:
            # 检查组件状态
            context_ok = True  # 上下文管理器总是就绪
            memory_ok = len(self.memory_manager.nodes) >= 0  # 简单检查
            
            # 性能检查
            avg_response_time = self._performance_stats["avg_response_time"]
            performance_ok = avg_response_time < 0.05  # 平均响应时间小于50ms
            
            status = "healthy" if (context_ok and memory_ok and performance_ok) else "degraded"
            
            return {
                "status": status,
                "components": {
                    "context_manager": "healthy" if context_ok else "degraded",
                    "memory_manager": "healthy" if memory_ok else "degraded",
                    "performance": "healthy" if performance_ok else "degraded"
                },
                "performance_metrics": self._performance_stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # 私有辅助方法
    
    def _context_to_memory_node(self, context_data: ContextCreate, context_id: str) -> Optional[str]:
        """将上下文转换为记忆节点"""
        try:
            # 提取文本内容
            content_text = ""
            if isinstance(context_data.content, dict):
                content_text = context_data.content.get("text", "")
            elif isinstance(context_data.content, str):
                content_text = context_data.content
            
            if not content_text:
                return None
            
            # 创建记忆节点
            metadata = {
                "context_id": context_id,
                "session_id": context_data.session_id,
                "user_id": context_data.user_id,
                "context_type": context_data.context_type,
                "priority": context_data.priority
            }
            
            # 合并元数据
            if context_data.metadata:
                metadata.update(context_data.metadata)
            
            node_id = self.memory_manager.create_memory_node(
                content=content_text,
                vector=None,  # 实际应该生成向量
                metadata=metadata
            )
            
            logger.debug(f"Converted context {context_id} to memory node {node_id}")
            return node_id
            
        except Exception as e:
            logger.error(f"Failed to convert context to memory node: {e}")
            return None
    
    def _update_performance_stats(self, start_time: float) -> None:
        """更新性能统计"""
        response_time = time.time() - start_time
        
        self._performance_stats["total_requests"] += 1
        
        # 更新平均响应时间（指数移动平均）
        alpha = 0.1  # 平滑因子
        old_avg = self._performance_stats["avg_response_time"]
        new_avg = old_avg * (1 - alpha) + response_time * alpha
        
        self._performance_stats["avg_response_time"] = new_avg
        
        # 定期重置统计
        current_time = time.time()
        if current_time - self._performance_stats["last_reset"] > 3600:  # 每小时重置
            self._performance_stats["total_requests"] = 0
            self._performance_stats["avg_response_time"] = 0.0
            self._performance_stats["last_reset"] = current_time
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """从缓存获取数据"""
        if not self.config.enable_caching:
            return None
        
        if key in self._cache:
            data, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return data
        
        return None
    
    def _set_to_cache(self, key: str, data: Any) -> None:
        """设置缓存数据"""
        if not self.config.enable_caching:
            return
        
        self._cache[key] = (data, time.time())
        
        # 清理过期缓存
        if len(self._cache) > 1000:  # 缓存大小限制
            current_time = time.time()
            expired_keys = [
                key for key, (_, timestamp) in self._cache.items()
                if current_time - timestamp > self._cache_ttl
            ]
            
            for key in expired_keys[:100]:  # 每次清理100个
                del self._cache[key]