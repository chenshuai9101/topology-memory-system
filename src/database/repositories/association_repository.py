"""
关联关系仓库
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID

from sqlalchemy import and_, or_, desc, func, text
from sqlalchemy.orm import joinedload

from .base_repository import BaseRepository
from ..models.associations import Association, AssociationStats, AssociationPattern

logger = logging.getLogger(__name__)


class AssociationRepository(BaseRepository[Association]):
    """关联关系仓库"""
    
    def __init__(self):
        super().__init__(Association)
    
    def get_by_nodes(self, source_id: UUID, target_id: UUID) -> List[Association]:
        """根据节点获取关联"""
        with self.get_session() as session:
            return (
                session.query(Association)
                .filter(
                    and_(
                        Association.source_node_id == source_id,
                        Association.target_node_id == target_id
                    )
                )
                .all()
            )
    
    def get_by_source(self, source_id: UUID, relation_type: str = None, 
                     limit: int = 50) -> List[Association]:
        """根据源节点获取关联"""
        with self.get_session() as session:
            query = (
                session.query(Association)
                .filter(Association.source_node_id == source_id)
            )
            
            if relation_type:
                query = query.filter(Association.relation_type == relation_type)
            
            return (
                query.order_by(desc(Association.weight))
                .limit(limit)
                .all()
            )
    
    def get_by_target(self, target_id: UUID, relation_type: str = None, 
                     limit: int = 50) -> List[Association]:
        """根据目标节点获取关联"""
        with self.get_session() as session:
            query = (
                session.query(Association)
                .filter(Association.target_node_id == target_id)
            )
            
            if relation_type:
                query = query.filter(Association.relation_type == relation_type)
            
            return (
                query.order_by(desc(Association.weight))
                .limit(limit)
                .all()
            )
    
    def get_by_context(self, context_id: UUID, limit: int = 100) -> List[Association]:
        """根据上下文获取关联"""
        with self.get_session() as session:
            return (
                session.query(Association)
                .filter(Association.context_id == context_id)
                .order_by(desc(Association.weight))
                .limit(limit)
                .all()
            )
    
    def get_strong_associations(self, threshold: float = 0.7, limit: int = 100) -> List[Association]:
        """获取强关联"""
        with self.get_session() as session:
            return (
                session.query(Association)
                .filter(Association.weight >= threshold)
                .order_by(desc(Association.weight))
                .limit(limit)
                .all()
            )
    
    def get_recent_associations(self, hours: int = 24, limit: int = 100) -> List[Association]:
        """获取最近创建的关联"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self.get_session() as session:
            return (
                session.query(Association)
                .filter(Association.created_at >= cutoff_time)
                .order_by(desc(Association.created_at))
                .limit(limit)
                .all()
            )
    
    def create_bidirectional(self, source_id: UUID, target_id: UUID, 
                           relation_type: str, weight: float = 1.0, 
                           context_id: UUID = None, **kwargs) -> Tuple[Association, Association]:
        """创建双向关联"""
        # 正向关联
        forward_assoc = self.create(
            source_node_id=source_id,
            target_node_id=target_id,
            relation_type=relation_type,
            weight=weight,
            context_id=context_id,
            bidirectional=True,
            **kwargs
        )
        
        # 反向关联
        reverse_type = forward_assoc.get_reverse_relation()
        reverse_assoc = self.create(
            source_node_id=target_id,
            target_node_id=source_id,
            relation_type=reverse_type,
            weight=weight,
            context_id=context_id,
            bidirectional=True,
            **kwargs
        )
        
        return forward_assoc, reverse_assoc
    
    def update_weight(self, association_id: UUID, new_weight: float, 
                     decay_factor: float = 0.9) -> Optional[Association]:
        """更新关联权重"""
        with self.get_session() as session:
            association = session.query(Association).filter(Association.id == association_id).first()
            if association:
                association.update_weight(new_weight, decay_factor)
                session.commit()
                session.refresh(association)
            return association
    
    def get_node_degree(self, node_id: UUID, direction: str = "both") -> Dict[str, int]:
        """获取节点度（关联数量）"""
        with self.get_session() as session:
            result = {"in": 0, "out": 0, "total": 0}
            
            if direction in ["both", "out"]:
                out_degree = (
                    session.query(func.count(Association.id))
                    .filter(Association.source_node_id == node_id)
                    .scalar()
                )
                result["out"] = out_degree or 0
            
            if direction in ["both", "in"]:
                in_degree = (
                    session.query(func.count(Association.id))
                    .filter(Association.target_node_id == node_id)
                    .scalar()
                )
                result["in"] = in_degree or 0
            
            result["total"] = result["in"] + result["out"]
            
            return result
    
    def get_association_stats(self) -> Dict[str, Any]:
        """获取关联统计"""
        with self.get_session() as session:
            stats = (
                session.query(
                    func.count(Association.id).label('total_count'),
                    func.avg(Association.weight).label('avg_weight'),
                    func.avg(Association.confidence).label('avg_confidence'),
                    func.count(Association.id).filter(Association.bidirectional == True).label('bidirectional_count')
                )
                .first()
            )
            
            # 按关系类型统计
            type_stats = (
                session.query(
                    Association.relation_type,
                    func.count(Association.id).label('count'),
                    func.avg(Association.weight).label('avg_weight')
                )
                .group_by(Association.relation_type)
                .all()
            )
            
            return {
                'total_count': stats.total_count or 0,
                'avg_weight': float(stats.avg_weight or 0),
                'avg_confidence': float(stats.avg_confidence or 0),
                'bidirectional_count': stats.bidirectional_count or 0,
                'by_type': {
                    stat.relation_type: {
                        'count': stat.count,
                        'avg_weight': float(stat.avg_weight or 0)
                    }
                    for stat in type_stats
                }
            }
    
    def find_paths(self, start_id: UUID, end_id: UUID, max_depth: int = 3, 
                  min_weight: float = 0.3) -> List[List[Association]]:
        """查找节点间的路径"""
        # 使用递归CTE查找路径
        with self.get_session() as session:
            # 递归CTE查询
            cte_query = text("""
                WITH RECURSIVE paths AS (
                    -- 基础情况：直接关联
                    SELECT 
                        source_node_id, 
                        target_node_id, 
                        id as association_id,
                        weight,
                        ARRAY[source_node_id] as path_nodes,
                        1 as depth
                    FROM associations
                    WHERE source_node_id = :start_id
                      AND weight >= :min_weight
                    
                    UNION ALL
                    
                    -- 递归情况：继续查找
                    SELECT 
                        a.source_node_id,
                        a.target_node_id,
                        a.id as association_id,
                        a.weight,
                        p.path_nodes || a.source_node_id,
                        p.depth + 1
                    FROM associations a
                    JOIN paths p ON a.source_node_id = p.target_node_id
                    WHERE p.depth < :max_depth
                      AND a.weight >= :min_weight
                      AND a.source_node_id != ALL(p.path_nodes)  -- 避免循环
                )
                SELECT * FROM paths
                WHERE target_node_id = :end_id
                ORDER BY depth, weight DESC
            """)
            
            result = session.execute(
                cte_query, 
                {
                    'start_id': start_id,
                    'end_id': end_id,
                    'max_depth': max_depth,
                    'min_weight': min_weight
                }
            ).fetchall()
            
            # 将结果转换为关联列表
            paths = []
            for row in result:
                # 这里需要根据association_id获取完整的关联对象
                # 简化处理：返回路径信息
                path_info = {
                    'source_id': row.source_node_id,
                    'target_id': row.target_node_id,
                    'association_id': row.association_id,
                    'weight': row.weight,
                    'path_nodes': row.path_nodes,
                    'depth': row.depth
                }
                paths.append(path_info)
            
            return paths


class AssociationStatsRepository(BaseRepository[AssociationStats]):
    """关联统计仓库"""
    
    def __init__(self):
        super().__init__(AssociationStats)
    
    def get_node_stats(self, node_id: UUID, date: datetime = None) -> List[AssociationStats]:
        """获取节点统计"""
        if date is None:
            date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        with self.get_session() as session:
            return (
                session.query(AssociationStats)
                .filter(
                    and_(
                        AssociationStats.node_id == node_id,
                        AssociationStats.date == date
                    )
                )
                .all()
            )
    
    def update_node_stats(self, node_id: UUID, date: datetime = None) -> List[AssociationStats]:
        """更新节点统计"""
        if date is None:
            date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        with self.get_session() as session:
            # 获取节点的所有关联
            associations = (
                session.query(Association)
                .filter(
                    or_(
                        Association.source_node_id == node_id,
                        Association.target_node_id == node_id
                    )
                )
                .all()
            )
            
            # 按关系和方向分组
            stats_by_key = {}
            
            for assoc in associations:
                if assoc.source_node_id == node_id:
                    direction = "out"
                    relation_type = assoc.relation_type
                else:
                    direction = "in"
                    relation_type = assoc.relation_type
                
                key = (relation_type, direction)
                
                if key not in stats_by_key:
                    stats_by_key[key] = {
                        'count': 0,
                        'total_weight': 0.0,
                        'total_confidence': 0.0
                    }
                
                stats = stats_by_key[key]
                stats['count'] += 1
                stats['total_weight'] += assoc.weight
                stats['total_confidence'] += assoc.confidence
            
            # 更新或创建统计记录
            updated_stats = []
            
            for (relation_type, direction), stats in stats_by_key.items():
                # 查找现有记录
                stat_record = (
                    session.query(AssociationStats)
                    .filter(
                        and_(
                            AssociationStats.node_id == node_id,
                            AssociationStats.date == date,
                            AssociationStats.relation_type == relation_type,
                            AssociationStats.direction == direction
                        )
                    )
                    .first()
                )
                
                if not stat_record:
                    stat_record = AssociationStats(
                        node_id=node_id,
                        date=date,
                        relation_type=relation_type,
                        direction=direction
                    )
                    session.add(stat_record)
                
                # 更新统计
                stat_record.count = stats['count']
                stat_record.total_weight = stats['total_weight']
                stat_record.avg_weight = stats['total_weight'] / stats['count'] if stats['count'] > 0 else 0
                stat_record.avg_confidence = stats['total_confidence'] / stats['count'] if stats['count'] > 0 else 0
                
                updated_stats.append(stat_record)
            
            session.commit()
            
            # 刷新所有记录
            for stat in updated_stats:
                session.refresh(stat)
            
            return updated_stats


class AssociationPatternRepository(BaseRepository[AssociationPattern]):
    """关联模式仓库"""
    
    def __init__(self):
        super().__init__(AssociationPattern)
    
    def get_common_patterns(self, min_support: float = 0.01, min_confidence: float = 0.5, 
                           limit: int = 50) -> List[AssociationPattern]:
        """获取常见模式"""
        with self.get_session() as session:
            return (
                session.query(AssociationPattern)
                .filter(
                    and_(
                        AssociationPattern.support >= min_support,
                        AssociationPattern.confidence >= min_confidence
                    )
                )
                .order_by(desc(AssociationPattern.support))
                .limit(limit)
                .all()
            )
    
    def discover_patterns(self, min_occurrence: int = 5) -> List[AssociationPattern]:
        """发现关联模式"""
        with self.get_session() as session:
            # 获取所有关联
            associations = session.query(Association).all()
            
            # 按源类型、目标类型、关系类型分组
            pattern_counts = {}
            source_counts = {}
            target_counts = {}
            total_associations = len(associations)
            
            for assoc in associations:
                # 获取节点类型（需要查询节点表）
                source_node = (
                    session.query(MemoryNode)
                    .filter(MemoryNode.id == assoc.source_node_id)
                    .first()
                )
                target_node = (
                    session.query(MemoryNode)
                    .filter(MemoryNode.id == assoc.target_node_id)
                    .first()
                )
                
                if source_node and target_node:
                    key = (source_node.node_type, target_node.node_type, assoc.relation_type)
                    
                    # 更新模式计数
                    if key not in pattern_counts:
                        pattern_counts[key] = {
                            'count': 0,
                            'total_weight': 0.0,
                            'total_confidence': 0.0
                        }
                    
                    pattern_counts[key]['count'] += 1
                    pattern_counts[key]['total_weight'] += assoc.weight
                    pattern_counts[key]['total_confidence'] += assoc.confidence
                    
                    # 更新源类型计数
                    source_type = source_node.node_type
                    if source_type not in source_counts:
                        source_counts[source_type] = 0
                    source_counts[source_type] += 1
                    
                    # 更新目标类型计数
                    target_type = target_node.node_type
                    if target_type not in target_counts:
                        target_counts[target_type] = 0
                    target_counts[target_type] += 1
            
            # 创建模式记录
            discovered_patterns = []
            
            for (source_type, target_type, relation_type), stats in pattern_counts.items():
                if stats['count'] >= min_occurrence:
                    pattern = AssociationPattern(
                        source_type=source_type,
                        target_type=target_type,
                        relation_type=relation_type,
                        occurrence_count=stats['count'],
                        avg_weight=stats['total_weight'] / stats['count'],
                        avg_confidence=stats['total_confidence'] / stats['count']
                    )
                    
                    # 计算指标
                    source_count = source_counts.get(source_type, 0)
                    target_count = target_counts.get(target_type, 0)
                    pattern.calculate_metrics(total_associations, source_count, target_count)
                    
                    discovered_patterns.append(pattern)
            
            # 保存模式
            saved_patterns = []
            for pattern in discovered_patterns:
                saved_pattern = self.create(**pattern.to_dict())
                saved_patterns.append(saved_pattern)
            
            return saved_patterns