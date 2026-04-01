"""
记忆节点仓库
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID

from sqlalchemy import and_, or_, desc, func, text
from sqlalchemy.orm import joinedload

from .base_repository import BaseRepository
from ..models.memory_nodes import MemoryNode, NodeCluster, NodeVersion

logger = logging.getLogger(__name__)


class MemoryNodeRepository(BaseRepository[MemoryNode]):
    """记忆节点仓库"""
    
    def __init__(self):
        super().__init__(MemoryNode)
    
    def get_by_type(self, node_type: str, limit: int = 100) -> List[MemoryNode]:
        """根据类型获取节点"""
        with self.get_session() as session:
            return (
                session.query(MemoryNode)
                .filter(MemoryNode.node_type == node_type)
                .order_by(desc(MemoryNode.importance_score))
                .limit(limit)
                .all()
            )
    
    def get_important_nodes(self, threshold: float = 0.7, limit: int = 50) -> List[MemoryNode]:
        """获取重要节点"""
        with self.get_session() as session:
            return (
                session.query(MemoryNode)
                .filter(MemoryNode.importance_score >= threshold)
                .order_by(desc(MemoryNode.importance_score))
                .limit(limit)
                .all()
            )
    
    def get_recently_accessed(self, hours: int = 24, limit: int = 50) -> List[MemoryNode]:
        """获取最近访问的节点"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self.get_session() as session:
            return (
                session.query(MemoryNode)
                .filter(MemoryNode.last_accessed >= cutoff_time)
                .order_by(desc(MemoryNode.last_accessed))
                .limit(limit)
                .all()
            )
    
    def get_frequently_accessed(self, limit: int = 50) -> List[MemoryNode]:
        """获取频繁访问的节点"""
        with self.get_session() as session:
            return (
                session.query(MemoryNode)
                .order_by(desc(MemoryNode.access_count))
                .limit(limit)
                .all()
            )
    
    def search_content(self, query_text: str, limit: int = 20) -> List[MemoryNode]:
        """搜索节点内容"""
        with self.get_session() as session:
            return (
                session.query(MemoryNode)
                .filter(
                    or_(
                        MemoryNode.content.ilike(f"%{query_text}%"),
                        MemoryNode.summary.ilike(f"%{query_text}%"),
                        MemoryNode.tags.any(query_text)
                    )
                )
                .order_by(desc(MemoryNode.importance_score))
                .limit(limit)
                .all()
            )
    
    def get_by_tags(self, tags: List[str], limit: int = 50) -> List[MemoryNode]:
        """根据标签获取节点"""
        with self.get_session() as session:
            query = session.query(MemoryNode)
            
            for tag in tags:
                query = query.filter(MemoryNode.tags.contains([tag]))
            
            return (
                query.order_by(desc(MemoryNode.importance_score))
                .limit(limit)
                .all()
            )
    
    def get_by_cluster(self, cluster_id: UUID, limit: int = 100) -> List[MemoryNode]:
        """根据聚类获取节点"""
        with self.get_session() as session:
            return (
                session.query(MemoryNode)
                .filter(MemoryNode.cluster_id == cluster_id)
                .order_by(desc(MemoryNode.importance_score))
                .limit(limit)
                .all()
            )
    
    def update_importance(self, node_id: UUID, factors: Dict[str, float] = None) -> Optional[MemoryNode]:
        """更新节点重要性"""
        if factors is None:
            factors = {
                'access_weight': 0.4,
                'association_weight': 0.4,
                'time_weight': 0.2
            }
        
        with self.get_session() as session:
            node = session.query(MemoryNode).filter(MemoryNode.id == node_id).first()
            if node:
                new_importance = node.calculate_importance(factors)
                node.importance_score = new_importance
                session.commit()
                session.refresh(node)
            return node
    
    def increment_access(self, node_id: UUID) -> Optional[MemoryNode]:
        """增加访问计数"""
        with self.get_session() as session:
            node = session.query(MemoryNode).filter(MemoryNode.id == node_id).first()
            if node:
                node.update_access()
                session.commit()
                session.refresh(node)
            return node
    
    def get_node_with_associations(self, node_id: UUID) -> Optional[MemoryNode]:
        """获取节点及其关联"""
        with self.get_session() as session:
            return (
                session.query(MemoryNode)
                .options(
                    joinedload(MemoryNode.source_associations),
                    joinedload(MemoryNode.target_associations)
                )
                .filter(MemoryNode.id == node_id)
                .first()
            )
    
    def get_related_nodes(self, node_id: UUID, relation_type: str = None, 
                         limit: int = 20) -> List[Tuple[MemoryNode, float]]:
        """获取相关节点"""
        with self.get_session() as session:
            # 获取源关联
            query = (
                session.query(MemoryNode, Association.weight)
                .join(Association, Association.target_node_id == MemoryNode.id)
                .filter(Association.source_node_id == node_id)
            )
            
            if relation_type:
                query = query.filter(Association.relation_type == relation_type)
            
            source_results = query.order_by(desc(Association.weight)).limit(limit).all()
            
            # 获取目标关联
            query = (
                session.query(MemoryNode, Association.weight)
                .join(Association, Association.source_node_id == MemoryNode.id)
                .filter(Association.target_node_id == node_id)
            )
            
            if relation_type:
                query = query.filter(Association.relation_type == relation_type)
            
            target_results = query.order_by(desc(Association.weight)).limit(limit).all()
            
            # 合并结果
            all_results = source_results + target_results
            
            # 去重并排序
            seen = set()
            unique_results = []
            for node, weight in all_results:
                if node.id not in seen:
                    seen.add(node.id)
                    unique_results.append((node, weight))
            
            # 按权重排序
            unique_results.sort(key=lambda x: x[1], reverse=True)
            
            return unique_results[:limit]
    
    def get_node_stats(self) -> Dict[str, Any]:
        """获取节点统计"""
        with self.get_session() as session:
            stats = (
                session.query(
                    func.count(MemoryNode.id).label('total_count'),
                    func.avg(MemoryNode.importance_score).label('avg_importance'),
                    func.avg(MemoryNode.access_count).label('avg_access'),
                    func.sum(MemoryNode.access_count).label('total_access')
                )
                .first()
            )
            
            # 按类型统计
            type_stats = (
                session.query(
                    MemoryNode.node_type,
                    func.count(MemoryNode.id).label('count'),
                    func.avg(MemoryNode.importance_score).label('avg_importance')
                )
                .group_by(MemoryNode.node_type)
                .all()
            )
            
            return {
                'total_count': stats.total_count or 0,
                'avg_importance': float(stats.avg_importance or 0),
                'avg_access': float(stats.avg_access or 0),
                'total_access': stats.total_access or 0,
                'by_type': {
                    stat.node_type: {
                        'count': stat.count,
                        'avg_importance': float(stat.avg_importance or 0)
                    }
                    for stat in type_stats
                }
            }


class NodeClusterRepository(BaseRepository[NodeCluster]):
    """节点聚类仓库"""
    
    def __init__(self):
        super().__init__(NodeCluster)
    
    def get_with_nodes(self, cluster_id: UUID) -> Optional[NodeCluster]:
        """获取聚类及其节点"""
        with self.get_session() as session:
            return (
                session.query(NodeCluster)
                .options(joinedload(NodeCluster.nodes))
                .filter(NodeCluster.id == cluster_id)
                .first()
            )
    
    def get_large_clusters(self, min_size: int = 5, limit: int = 20) -> List[NodeCluster]:
        """获取大型聚类"""
        with self.get_session() as session:
            return (
                session.query(NodeCluster)
                .filter(NodeCluster.node_count >= min_size)
                .order_by(desc(NodeCluster.node_count))
                .limit(limit)
                .all()
            )
    
    def get_cohesive_clusters(self, threshold: float = 0.7, limit: int = 20) -> List[NodeCluster]:
        """获取内聚性高的聚类"""
        with self.get_session() as session:
            return (
                session.query(NodeCluster)
                .filter(NodeCluster.cohesion_score >= threshold)
                .order_by(desc(NodeCluster.cohesion_score))
                .limit(limit)
                .all()
            )
    
    def update_cluster_statistics(self, cluster_id: UUID) -> Optional[NodeCluster]:
        """更新聚类统计"""
        with self.get_session() as session:
            cluster = session.query(NodeCluster).filter(NodeCluster.id == cluster_id).first()
            if cluster:
                cluster.update_statistics()
                session.commit()
                session.refresh(cluster)
            return cluster


class NodeVersionRepository(BaseRepository[NodeVersion]):
    """节点版本仓库"""
    
    def __init__(self):
        super().__init__(NodeVersion)
    
    def get_by_node(self, node_id: UUID, limit: int = 50) -> List[NodeVersion]:
        """根据节点ID获取版本"""
        with self.get_session() as session:
            return (
                session.query(NodeVersion)
                .filter(NodeVersion.node_id == node_id)
                .order_by(desc(NodeVersion.version))
                .limit(limit)
                .all()
            )
    
    def get_latest_version(self, node_id: UUID) -> Optional[NodeVersion]:
        """获取最新版本"""
        with self.get_session() as session:
            return (
                session.query(NodeVersion)
                .filter(NodeVersion.node_id == node_id)
                .order_by(desc(NodeVersion.version))
                .first()
            )
    
    def create_version(self, node: MemoryNode, changed_by: str = None, 
                      change_reason: str = None) -> NodeVersion:
        """创建新版本"""
        # 获取当前最大版本号
        latest = self.get_latest_version(node.id)
        next_version = (latest.version + 1) if latest else 1
        
        version = NodeVersion(
            node_id=node.id,
            version=next_version,
            content=node.content,
            summary=node.summary,
            metadata=node.metadata,
            changed_by=changed_by,
            change_reason=change_reason
        )
        
        return self.create(**version.to_dict())