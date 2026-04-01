"""
关联关系数据模型
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import (
    Column, String, Text, Integer, DateTime, 
    JSON, Boolean, Float, ForeignKey, Index,
    UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, validates

from ..config.database_manager import Base


class Association(Base):
    """关联关系表模型"""
    
    __tablename__ = "associations"
    __table_args__ = (
        # 复合索引
        Index("idx_associations_source_target", "source_node_id", "target_node_id"),
        Index("idx_associations_context_source", "context_id", "source_node_id"),
        Index("idx_associations_relation_weight", "relation_type", "weight"),
        Index("idx_associations_created_at", "created_at"),
        
        # 唯一约束 - 防止重复关联
        UniqueConstraint("source_node_id", "target_node_id", "context_id", "relation_type",
                        name="uq_source_target_context_relation"),
        
        # 检查约束
        CheckConstraint("weight >= 0.0 AND weight <= 1.0", name="check_weight_range"),
        CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name="check_confidence_range"),
        CheckConstraint("bidirectional IN (true, false)", name="check_bidirectional"),
    )
    
    # 主键
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
        comment="关联ID"
    )
    
    # 关联源和目标
    source_node_id = Column(
        UUID(as_uuid=True),
        ForeignKey("memory_nodes.id", ondelete="CASCADE"),
        nullable=False,
        comment="源节点ID"
    )
    
    target_node_id = Column(
        UUID(as_uuid=True),
        ForeignKey("memory_nodes.id", ondelete="CASCADE"),
        nullable=False,
        comment="目标节点ID"
    )
    
    context_id = Column(
        UUID(as_uuid=True),
        ForeignKey("contexts.id", ondelete="CASCADE"),
        nullable=True,
        comment="上下文ID"
    )
    
    # 关联属性
    relation_type = Column(
        String(50),
        nullable=False,
        index=True,
        comment="关系类型: related_to, similar_to, part_of, causes, precedes, other"
    )
    
    weight = Column(
        Float,
        default=1.0,
        nullable=False,
        comment="关联权重(0.0-1.0)"
    )
    
    confidence = Column(
        Float,
        default=1.0,
        nullable=False,
        comment="置信度(0.0-1.0)"
    )
    
    bidirectional = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="是否双向关联"
    )
    
    # 元数据
    metadata = Column(
        JSONB,
        default={},
        comment="元数据"
    )
    
    description = Column(
        Text,
        nullable=True,
        comment="关联描述"
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
    source_node = relationship(
        "MemoryNode",
        foreign_keys=[source_node_id],
        back_populates="source_associations"
    )
    
    target_node = relationship(
        "MemoryNode",
        foreign_keys=[target_node_id],
        back_populates="target_associations"
    )
    
    context = relationship(
        "Context",
        back_populates="associations"
    )
    
    def __repr__(self):
        return f"<Association(source={self.source_node_id}, target={self.target_node_id}, type={self.relation_type})>"
    
    @validates('relation_type')
    def validate_relation_type(self, key, value):
        """验证关系类型"""
        valid_types = ['related_to', 'similar_to', 'part_of', 'causes', 'precedes', 'other']
        if value not in valid_types:
            raise ValueError(f"relation_type must be one of {valid_types}")
        return value
    
    @validates('weight', 'confidence')
    def validate_score(self, key, value):
        """验证评分范围"""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be between 0.0 and 1.0")
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": str(self.id),
            "source_node_id": str(self.source_node_id),
            "target_node_id": str(self.target_node_id),
            "context_id": str(self.context_id) if self.context_id else None,
            "relation_type": self.relation_type,
            "weight": self.weight,
            "confidence": self.confidence,
            "bidirectional": self.bidirectional,
            "metadata": self.metadata,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
    
    def get_reverse_relation(self) -> str:
        """获取反向关系类型"""
        reverse_map = {
            'related_to': 'related_to',
            'similar_to': 'similar_to',
            'part_of': 'contains',
            'causes': 'caused_by',
            'precedes': 'follows',
            'other': 'other'
        }
        return reverse_map.get(self.relation_type, 'related_to')
    
    def update_weight(self, new_weight: float, decay_factor: float = 0.9):
        """更新权重（带衰减）"""
        self.weight = self.weight * decay_factor + new_weight * (1 - decay_factor)
        self.weight = max(0.0, min(1.0, self.weight))


class AssociationStats(Base):
    """关联统计表"""
    
    __tablename__ = "association_stats"
    __table_args__ = (
        Index("idx_association_stats_node_id", "node_id"),
        Index("idx_association_stats_date", "date"),
        Index("idx_association_stats_relation_type", "relation_type"),
        UniqueConstraint("node_id", "date", "relation_type", "direction",
                        name="uq_node_date_relation_direction"),
    )
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="统计ID"
    )
    
    node_id = Column(
        UUID(as_uuid=True),
        ForeignKey("memory_nodes.id", ondelete="CASCADE"),
        nullable=False,
        comment="节点ID"
    )
    
    date = Column(
        DateTime(timezone=True),
        nullable=False,
        comment="统计日期"
    )
    
    relation_type = Column(
        String(50),
        nullable=False,
        comment="关系类型"
    )
    
    direction = Column(
        String(10),
        nullable=False,
        comment="方向: in, out"
    )
    
    # 统计指标
    count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="关联数量"
    )
    
    avg_weight = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="平均权重"
    )
    
    avg_confidence = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="平均置信度"
    )
    
    total_weight = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="总权重"
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
    node = relationship("MemoryNode", backref="association_stats")
    
    def __repr__(self):
        return f"<AssociationStats(node_id={self.node_id}, date={self.date}, type={self.relation_type})>"


class AssociationPattern(Base):
    """关联模式表 - 用于发现常见关联模式"""
    
    __tablename__ = "association_patterns"
    __table_args__ = (
        Index("idx_association_patterns_support", "support"),
        Index("idx_association_patterns_confidence", "confidence"),
        Index("idx_association_patterns_created_at", "created_at"),
    )
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="模式ID"
    )
    
    # 模式定义
    source_type = Column(
        String(50),
        nullable=False,
        comment="源节点类型"
    )
    
    target_type = Column(
        String(50),
        nullable=False,
        comment="目标节点类型"
    )
    
    relation_type = Column(
        String(50),
        nullable=False,
        comment="关系类型"
    )
    
    # 模式统计
    support = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="支持度"
    )
    
    confidence = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="置信度"
    )
    
    lift = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="提升度"
    )
    
    occurrence_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="出现次数"
    )
    
    # 模式特征
    avg_weight = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="平均权重"
    )
    
    avg_confidence = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="平均置信度"
    )
    
    # 元数据
    metadata = Column(
        JSONB,
        default={},
        comment="元数据"
    )
    
    description = Column(
        Text,
        nullable=True,
        comment="模式描述"
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
    
    def __repr__(self):
        return f"<AssociationPattern({self.source_type}->{self.target_type}:{self.relation_type}, support={self.support})>"
    
    def calculate_metrics(self, total_associations: int, source_count: int, target_count: int):
        """计算模式指标"""
        # 支持度 = 模式出现次数 / 总关联数
        self.support = self.occurrence_count / total_associations if total_associations > 0 else 0
        
        # 置信度 = 模式出现次数 / 源节点出现次数
        self.confidence = self.occurrence_count / source_count if source_count > 0 else 0
        
        # 提升度 = 置信度 / (目标节点出现次数 / 总关联数)
        target_probability = target_count / total_associations if total_associations > 0 else 0
        self.lift = self.confidence / target_probability if target_probability > 0 else 0