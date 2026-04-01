"""
数据库服务层
整合仓库和缓存，提供业务逻辑
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID, uuid4

from ..repositories import (
    ContextRepository, ContextHistoryRepository, ContextStatsRepository,
    MemoryNodeRepository, NodeClusterRepository, NodeVersionRepository,
    AssociationRepository, AssociationStatsRepository, AssociationPatternRepository
)
from .redis_cache import redis_cache, cached

logger = logging.getLogger(__name__)


class DatabaseService:
    """数据库服务"""
    
    def __init__(self):
        # 初始化仓库
        self.context_repo = ContextRepository()
        self.context_history_repo = ContextHistoryRepository()
        self.context_stats_repo = ContextStatsRepository()
        
        self.memory_node_repo = MemoryNodeRepository()
        self.node_cluster_repo = NodeClusterRepository()
        self.node_version_repo = NodeVersionRepository()
        
        self.association_repo = AssociationRepository()
        self.association_stats_repo = AssociationStatsRepository()
        self.association_pattern_repo = AssociationPatternRepository()
    
    # ========== 上下文服务 ==========
    
    @cached(ttl=300, key_prefix="context")
    def get_context(self, context_id: UUID) -> Optional[Dict[str, Any]]:
        """获取上下文"""
        context = self.context_repo.get_by_id(context_id)
        if context:
            # 更新访问统计
            self.context_repo.increment_access(context_id)
            return context.to_dict()
        return None
    
    def create_context(self, session_id: str, user_id: str, context_type: str, 
                      content: Dict[str, Any], metadata: Dict[str, Any] = None,
                      priority: int = 1, ttl: int = None) -> Dict[str, Any]:
        """创建上下文"""
        if metadata is None:
            metadata = {}
        
        # 创建上下文
        context = self.context_repo.create(
            session_id=session_id,
            user_id=user_id,
            context_type=context_type,
            content=content,
            metadata=metadata,
            priority=priority,
            ttl=ttl
        )
        
        # 计算过期时间
        if ttl:
            context.calculate_expiry()
            self.context_repo.update(context.id, expires_at=context.expires_at)
        
        # 创建历史版本
        self.context_history_repo.create_version(context)
        
        # 更新统计
        self.context_stats_repo.update_daily_stats(session_id, context_type)
        
        # 缓存上下文
        redis_cache.cache_context(context.id, context.to_dict())
        
        return context.to_dict()
    
    def update_context(self, context_id: UUID, **kwargs) -> Optional[Dict[str, Any]]:
        """更新上下文"""
        # 获取原始上下文
        context = self.context_repo.get_by_id(context_id)
        if not context:
            return None
        
        # 创建历史版本
        self.context_history_repo.create_version(
            context, 
            changed_by=kwargs.get('changed_by'),
            change_reason=kwargs.get('change_reason')
        )
        
        # 更新上下文
        updated = self.context_repo.update(context_id, **kwargs)
        if updated:
            # 清除缓存
            redis_cache.delete(f"context:{context_id}")
            
            # 重新缓存
            redis_cache.cache_context(context_id, updated.to_dict())
            
            return updated.to_dict()
        
        return None
    
    @cached(ttl=600, key_prefix="session_contexts")
    def get_session_contexts(self, session_id: str, active_only: bool = True) -> List[Dict[str, Any]]:
        """获取会话上下文"""
        contexts = self.context_repo.get_by_session(session_id, active_only)
        return [ctx.to_dict() for ctx in contexts]
    
    def deactivate_expired_contexts(self) -> int:
        """停用过期的上下文"""
        return self.context_repo.deactivate_expired()
    
    # ========== 记忆节点服务 ==========
    
    @cached(ttl=300, key_prefix="memory_node")
    def get_memory_node(self, node_id: UUID) -> Optional[Dict[str, Any]]:
        """获取记忆节点"""
        node = self.memory_node_repo.get_by_id(node_id)
        if node:
            # 更新访问统计
            self.memory_node_repo.increment_access(node_id)
            redis_cache.add_to_recently_accessed(node_id)
            return node.to_dict()
        return None
    
    def create_memory_node(self, node_type: str, content: str, summary: str = None,
                          metadata: Dict[str, Any] = None, tags: List[str] = None,
                          embedding: List[float] = None) -> Dict[str, Any]:
        """创建记忆节点"""
        if metadata is None:
            metadata = {}
        if tags is None:
            tags = []
        
        # 创建节点
        node = self.memory_node_repo.create(
            node_type=node_type,
            content=content,
            summary=summary,
            metadata=metadata,
            tags=tags,
            embedding=embedding,
            embedding_dim=len(embedding) if embedding else None
        )
        
        # 创建初始版本
        self.node_version_repo.create_version(node)
        
        # 缓存节点
        redis_cache.cache_memory_node(node.id, node.to_dict())
        
        return node.to_dict()
    
    def update_memory_node(self, node_id: UUID, **kwargs) -> Optional[Dict[str, Any]]:
        """更新记忆节点"""
        # 获取原始节点
        node = self.memory_node_repo.get_by_id(node_id)
        if not node:
            return None
        
        # 创建新版本
        self.node_version_repo.create_version(
            node,
            changed_by=kwargs.get('changed_by'),
            change_reason=kwargs.get('change_reason')
        )
        
        # 更新节点
        updated = self.memory_node_repo.update(node_id, **kwargs)
        if updated:
            # 清除缓存
            redis_cache.delete(f"memory_node:{node_id}")
            
            # 重新缓存
            redis_cache.cache_memory_node(node_id, updated.to_dict())
            
            return updated.to_dict()
        
        return None
    
    @cached(ttl=600, key_prefix="important_nodes")
    def get_important_nodes(self, threshold: float = 0.7, limit: int = 50) -> List[Dict[str, Any]]:
        """获取重要节点"""
        nodes = self.memory_node_repo.get_important_nodes(threshold, limit)
        return [node.to_dict() for node in nodes]
    
    @cached(ttl=300, key_prefix="related_nodes")
    def get_related_nodes(self, node_id: UUID, relation_type: str = None, 
                         limit: int = 20) -> List[Dict[str, Any]]:
        """获取相关节点"""
        related = self.memory_node_repo.get_related_nodes(node_id, relation_type, limit)
        
        result = []
        for node, weight in related:
            node_dict = node.to_dict()
            node_dict['relation_weight'] = weight
            result.append(node_dict)
        
        return result
    
    def search_nodes(self, query_text: str, limit: int = 20) -> List[Dict[str, Any]]:
        """搜索节点"""
        nodes = self.memory_node_repo.search_content(query_text, limit)
        return [node.to_dict() for node in nodes]
    
    # ========== 关联服务 ==========
    
    def create_association(self, source_id: UUID, target_id: UUID, relation_type: str,
                          weight: float = 1.0, confidence: float = 1.0,
                          context_id: UUID = None, bidirectional: bool = False,
                          metadata: Dict[str, Any] = None, description: str = None) -> Dict[str, Any]:
        """创建关联"""
        if metadata is None:
            metadata = {}
        
        if bidirectional:
            # 创建双向关联
            forward, reverse = self.association_repo.create_bidirectional(
                source_id, target_id, relation_type, weight, context_id,
                confidence=confidence, metadata=metadata, description=description
            )
            
            # 更新节点统计
            self.association_stats_repo.update_node_stats(source_id)
            self.association_stats_repo.update_node_stats(target_id)
            
            return {
                'forward': forward.to_dict(),
                'reverse': reverse.to_dict()
            }
        else:
            # 创建单向关联
            association = self.association_repo.create(
                source_node_id=source_id,
                target_node_id=target_id,
                relation_type=relation_type,
                weight=weight,
                confidence=confidence,
                context_id=context_id,
                bidirectional=False,
                metadata=metadata,
                description=description
            )
            
            # 更新节点统计
            self.association_stats_repo.update_node_stats(source_id)
            
            return association.to_dict()
    
    def update_association_weight(self, association_id: UUID, new_weight: float,
                                decay_factor: float = 0.9) -> Optional[Dict[str, Any]]:
        """更新关联权重"""
        association = self.association_repo.update_weight(association_id, new_weight, decay_factor)
        if association:
            return association.to_dict()
        return None
    
    def find_paths_between_nodes(self, start_id: UUID, end_id: UUID, 
                                max_depth: int = 3, min_weight: float = 0.3) -> List[Dict[str, Any]]:
        """查找节点间的路径"""
        return self.association_repo.find_paths(start_id, end_id, max_depth, min_weight)
    
    # ========== 聚类服务 ==========
    
    def create_cluster(self, name: str, description: str = None, 
                      centroid_id: UUID = None) -> Dict[str, Any]:
        """创建聚类"""
        cluster = self.node_cluster_repo.create(
            name=name,
            description=description,
            centroid_id=centroid_id
        )
        
        return cluster.to_dict()
    
    def add_node_to_cluster(self, node_id: UUID, cluster_id: UUID, 
                           distance: float = None) -> bool:
        """添加节点到聚类"""
        node = self.memory_node_repo.get_by_id(node_id)
        if not node:
            return False
        
        # 更新节点聚类信息
        updated = self.memory_node_repo.update(
            node_id,
            cluster_id=cluster_id,
            cluster_distance=distance
        )
        
        if updated:
            # 更新聚类统计
            self.node_cluster_repo.update_cluster_statistics(cluster_id)
            return True
        
        return False
    
    @cached(ttl=600, key_prefix="cluster_nodes")
    def get_cluster_nodes(self, cluster_id: UUID, limit: int = 100) -> List[Dict[str, Any]]:
        """获取聚类节点"""
        nodes = self.memory_node_repo.get_by_cluster(cluster_id, limit)
        return [node.to_dict() for node in nodes]
    
    # ========== 统计服务 ==========
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计"""
        stats = {}
        
        # 上下文统计
        context_stats = self.context_repo.get_session_stats("system")
        stats['contexts'] = context_stats
        
        # 节点统计
        node_stats = self.memory_node_repo.get_node_stats()
        stats['memory_nodes'] = node_stats
        
        # 关联统计
        association_stats = self.association_repo.get_association_stats()
        stats['associations'] = association_stats
        
        # 缓存统计
        cache_stats = redis_cache.get_cache_stats()
        stats['cache'] = cache_stats
        
        return stats
    
    def get_performance_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """获取性能指标"""
        metrics = {}
        
        # 最近活跃的上下文
        active_contexts = self.context_repo.get_active_contexts(hours)
        metrics['active_contexts'] = len(active_contexts)
        
        # 最近访问的节点
        recent_nodes = self.memory_node_repo.get_recently_accessed(hours)
        metrics['recently_accessed_nodes'] = len(recent_nodes)
        
        # 频繁访问的节点
        frequent_nodes = self.memory_node_repo.get_frequently_accessed(20)
        metrics['frequently_accessed_nodes'] = len(frequent_nodes)
        
        # 强关联数量
        strong_associations = self.association_repo.get_strong_associations(0.7, 100)
        metrics['strong_associations'] = len(strong_associations)
        
        # 最近创建的关联
        recent_associations = self.association_repo.get_recent_associations(hours)
        metrics['recent_associations'] = len(recent_associations)
        
        return metrics
    
    # ========== 维护服务 ==========
    
    def cleanup_old_data(self, days: int = 30) -> Dict[str, int]:
        """清理旧数据"""
        cleanup_stats = {}
        
        # 清理旧的上下文
        context_count = self.context_repo.cleanup_old_contexts(days)
        cleanup_stats['contexts'] = context_count
        
        # 停用过期的上下文
        expired_count = self.context_repo.deactivate_expired()
        cleanup_stats['expired_contexts'] = expired_count
        
        # 清理缓存
        redis_cache.clear_all_cache()
        cleanup_stats['cache_cleared'] = 1
        
        logger.info(f"数据清理完成: {cleanup_stats}")
        return cleanup_stats
    
    def rebuild_cache(self) -> Dict[str, int]:
        """重建缓存"""
        rebuild_stats = {}
        
        # 清除所有缓存
        redis_cache.clear_all_cache()
        rebuild_stats['cache_cleared'] = 1
        
        # 缓存重要节点
        important_nodes = self.memory_node_repo.get_important_nodes(0.7, 100)
        for node in important_nodes:
            redis_cache.cache_memory_node(node.id, node.to_dict())
        rebuild_stats['important_nodes_cached'] = len(important_nodes)
        
        # 缓存最近访问的节点
        recent_nodes = self.memory_node_repo.get_recently_accessed(24, 50)
        for node in recent_nodes:
            redis_cache.add_to_recently_accessed(node.id)
        rebuild_stats['recent_nodes_cached'] = len(recent_nodes)
        
        logger.info(f"缓存重建完成: {rebuild_stats}")
        return rebuild_stats
    
    def discover_patterns(self) -> List[Dict[str, Any]]:
        """发现关联模式"""
        patterns = self.association_pattern_repo.discover_patterns(min_occurrence=5)
        return [pattern.to_dict() for pattern in patterns]
    
    # ========== 健康检查 ==========
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health = {
            'database': False,
            'cache': False,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # 检查数据库连接
            from ..config.database_manager import db_manager
            db_health = db_manager.health_check()
            health['database'] = all(db_health.values())
            health['database_details'] = db_health
            
            # 检查缓存连接
            redis_cache.redis.ping()
            health['cache'] = True
            
            # 获取连接统计
            connection_stats = db_manager.get_connection_stats()
            health['connection_stats'] = connection_stats
            
            # 获取缓存统计
            cache_stats = redis_cache.get_cache_stats()
            health['cache_stats'] = cache_stats
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            health['error'] = str(e)
        
        return health


# 全局数据库服务实例
db_service = DatabaseService()