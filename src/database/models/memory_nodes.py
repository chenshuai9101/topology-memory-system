"""
记忆节点数据模型
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import (
    Column, String, Text, Integer, DateTime, 
    JSON, Boolean, Float, ForeignKey, Index,
    UniqueConstraint, CheckConstraint, ARRAY
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, validates

from ..config.database_manager import Base


class MemoryNode(Base):
    """记忆节点表模型"""
    
    __tablename__ = "memory_nodes"
    __table_args__ = (
        # 复合索引
        Index("idx_memory_nodes_type_importance", "node_type", "importance_score"),
        Index("idx_memory_nodes_created_at", "created_at"),
        Index("idx_memory_nodes_access_count", "access_count"),
        Index("idx_memory_nodes_cluster_id", "cluster_id"),
        
        # 部分索引 - 只索引重要节点
        Index("idx_memory_nodes_important", "importance_score", 
              postgresql_where="importance_score >= 0.7"),
        
        # 检查约束
        CheckConstraint("importance_score >= 0.0 AND importance_score <= 1.0", 
                       name="check_importance_range"),
        CheckConstraint("access_count >= 0", name="check_access_count_positive"),
        CheckConstraint("stability_score >= 0.0 AND stability_score <= 1.0", 
                       name="check_stability_range"),
    )
    
    # 主键
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
        comment="节点ID"
    )
    
    # 基础字段
    node_type = Column(
        String(50),
        nullable=False,
        index=True,
        comment="节点类型: concept, entity, event, relation, pattern"
    )
    
    content = Column(
        Text,
        nullable=False,
        comment="节点内容"
    )
    
    summary = Column(
        Text,
        nullable=True,
        comment="内容摘要"
    )
    
    # 向量表示
    embedding = Column(
        JSONB,
        nullable=True,
        comment="向量嵌入"
    )
    
    embedding_dim = Column(
        Integer,
        nullable=True,
        comment="向量维度"
    )
    
    # 元数据
    metadata = Column(
        JSONB,
        default={},
        comment="元数据"
    )
    
    tags = Column(
        ARRAY(String),
        default=[],
        comment="标签"
    )
    
    # 重要性评分
    importance_score = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="重要性评分(0.0-1.0)"
    )
    
    stability_score = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="稳定性评分(0.0-1.0)"
    )
    
    relevance_score = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="相关性评分(0.0-1.0)"
    )
    
    # 访问统计
    access_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="访问次数"
    )
    
    last_accessed = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="最后访问时间"
    )
    
    # 聚类信息
    cluster_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="聚类ID"
    )
    
    cluster_distance = Column(
        Float,
        nullable=True,
        comment="聚类距离"
    )
    
    # 时间戳
    created_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
        comment="创建时间"
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
        comment="更新时间"
    )
    
    # 关系
    source_associations = relationship(
        "Association",
        foreign_keys="Association.source_node_id",
        back_populates="source_node",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    target_associations = relationship(
        "Association",
        foreign_keys="Association.target_node_id",
        back_populates="target_node",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    def __repr__(self):
        return f"<MemoryNode(id={self.id}, type={self.node_type}, importance={self.importance_score})>"
    
    @validates('node_type')
    def validate_node_type(self, key, value):
        """验证节点类型"""
        valid_types = ['concept', 'entity', 'event', 'relation', 'pattern', 'other']
        if value not in valid_types:
            raise ValueError(f"node_type must be one of {valid_types}")
        return value
    
    @validates('importance_score', 'stability_score', 'relevance_score')
    def validate_score(self, key, value):
        """验证评分范围"""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be between 0.0 and 1.0")
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": str(self.id),
            "node_type": self.node_type,
            "content": self.content,
            "summary": self.summary,
            "embedding_dim": self.embedding_dim,
            "metadata": self.metadata,
            "tags": self.tags,
            "importance_score": self.importance_score,
            "stability_score": self.stability_score,
            "relevance_score": self.relevance_score,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "cluster_id": str(self.cluster_id) if self.cluster_id else None,
            "cluster_distance": self.cluster_distance,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
    
    def update_access(self):
        """更新访问统计"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
    
    def calculate_importance(self, factors: Dict[str, float]) -> float:
        """计算重要性评分"""
        # 基础重要性计算
        importance = 0.0
        
        # 访问频率权重
        if factors.get('access_weight', 0) > 0:
            importance += min(self.access_count / 100, 1.0) * factors['access_weight']
        
        # 关联数量权重
        if factors.get('association_weight', 0) > 0:
            total_associations = self.source_associations.count() + self.target_associations.count()
            importance += min(total_associations / 50, 1.0) * factors['association_weight']
        
        # 时间衰减权重
        if factors.get('time_weight', 0) > 0:
            days_old = (datetime.utcnow() - self.created_at).days
            time_factor = max(0, 1.0 - (days_old / 365))  # 一年衰减
            importance += time_factor * factors['time_weight']
        
        return min(max(importance, 0.0), 1.0)


class NodeCluster(Base):
    """节点聚类表"""
    
    __tablename__ = "node_clusters"
    __table_args__ = (
        Index("idx_node_clusters_centroid", "centroid_id"),
        Index("idx_node_clusters_created_at", "created_at"),
    )
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="聚类ID"
    )
    
    name = Column(
        String(200),
        nullable=True,
        comment="聚类名称"
    )
    
    description = Column(
        Text,
        nullable=True,
        comment="聚类描述"
    )
    
    centroid_id = Column(
        UUID(as_uuid=True),
        ForeignKey("memory_nodes.id"),
        nullable=True,
        comment="质心节点ID"
    )
    
    # 聚类统计
    node_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="节点数量"
    )
    
    avg_importance = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="平均重要性"
    )
    
    avg_stability = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="平均稳定性"
    )
    
    cohesion_score = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="内聚性评分"
    )
    
    # 聚类特征
    keywords = Column(
        ARRAY(String),
        default=[],
        comment="关键词"
    )
    
    metadata = Column(
        JSONB,
        default={},
        comment="聚类元数据"
    )
    
    # 时间戳
    created_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
        comment="创建时间"
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
        comment="更新时间"
    )
    
    # 关系
    centroid = relationship("MemoryNode", foreign_keys=[centroid_id])
    nodes = relationship("MemoryNode", backref="cluster")
    
    def __repr__(self):
        return f"<NodeCluster(id={self.id}, name={self.name}, nodes={self.node_count})>"
    
    def update_statistics(self):
        """更新聚类统计"""
        if self.nodes:
            importances = [node.importance_score for node in self.nodes]
            stabilities = [node.stability_score for node in self.nodes]
            
            self.node_count = len(self.nodes)
            self.avg_importance = sum(importances) / len(importances)
            self.avg_stability = sum(stabilities) / len(stabilities)
            
            # 简单内聚性计算（基于节点间距离）
            # 这里可以扩展为更复杂的计算
            self.cohesion_score = 0.8  # 示例值


class NodeVersion(Base):
    """节点版本表 - 用于节点内容版本控制"""
    
    __tablename__ = "node_versions"
    __table_args__ = (
        Index("idx_node_versions_node_id", "node_id"),
        Index("idx_node_versions_created_at", "created_at"),
        Index("idx_node_versions_version", "version"),
    )
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="版本ID"
    )
    
    node_id = Column(
        UUID(as_uuid=True),
        ForeignKey("memory_nodes.id", ondelete="CASCADE"),
        nullable=False,
        comment="节点ID"
    )
    
    version = Column(
        Integer,
        nullable=False,
        comment="版本号"
    )
    
    content = Column(
        Text,
        nullable=False,
        comment="内容快照"
    )
    
    summary = Column(
        Text,
        nullable=True,
        comment="摘要快照"
    )
    
    metadata = Column(
        JSONB,
        default={},
        comment="元数据快照"
    )
    
    changed_by = Column(
        String(100),
        nullable=True,
        comment="修改者"
    )
    
    change_reason = Column(
        String(500),
        nullable=True,
        comment="修改原因"
    )
    
    created_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
        comment="创建时间"
    )
    
    # 关系
    node = relationship("MemoryNode", backref="versions")
    
    def __repr__(self):
        return f"<NodeVersion(node_id={self.node_id}, version={self.version})>"