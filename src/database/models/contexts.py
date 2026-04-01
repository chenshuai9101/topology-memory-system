"""
上下文数据模型
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import (
    Column, String, Text, Integer, DateTime, 
    JSON, Boolean, Float, ForeignKey, Index,
    UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, validates

from ..config.database_manager import Base


class Context(Base):
    """上下文表模型"""
    
    __tablename__ = "contexts"
    __table_args__ = (
        # 复合索引
        Index("idx_contexts_session_priority", "session_id", "priority"),
        Index("idx_contexts_created_at", "created_at"),
        Index("idx_contexts_user_session", "user_id", "session_id"),
        Index("idx_contexts_type_priority", "context_type", "priority"),
        
        # 部分索引 - 只索引活跃上下文
        Index("idx_contexts_active", "is_active", postgresql_where="is_active = true"),
        
        # 检查约束
        CheckConstraint("priority >= 1 AND priority <= 10", name="check_priority_range"),
        CheckConstraint("ttl >= 0", name="check_ttl_positive"),
    )
    
    # 主键
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
        comment="上下文ID"
    )
    
    # 基础字段
    session_id = Column(
        String(100),
        nullable=False,
        index=True,
        comment="会话ID"
    )
    
    user_id = Column(
        String(100),
        nullable=False,
        index=True,
        comment="用户ID"
    )
    
    context_type = Column(
        String(50),
        nullable=False,
        index=True,
        comment="上下文类型: conversation, memory, task, knowledge, other"
    )
    
    content = Column(
        JSONB,
        nullable=False,
        comment="上下文内容"
    )
    
    metadata = Column(
        JSONB,
        default={},
        comment="元数据"
    )
    
    # 优先级和状态
    priority = Column(
        Integer,
        nullable=False,
        default=1,
        comment="优先级(1-10)"
    )
    
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        comment="是否活跃"
    )
    
    # 时间管理
    ttl = Column(
        Integer,
        nullable=True,
        comment="生存时间(秒)"
    )
    
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
    
    expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="过期时间"
    )
    
    # 性能指标
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
    
    # 向量字段 (用于语义搜索)
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
    
    # 关系
    associations = relationship(
        "Association",
        back_populates="context",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    def __repr__(self):
        return f"<Context(id={self.id}, session_id={self.session_id}, type={self.context_type})>"
    
    @validates('context_type')
    def validate_context_type(self, key, value):
        """验证上下文类型"""
        valid_types = ['conversation', 'memory', 'task', 'knowledge', 'other']
        if value not in valid_types:
            raise ValueError(f"context_type must be one of {valid_types}")
        return value
    
    @validates('priority')
    def validate_priority(self, key, value):
        """验证优先级"""
        if not 1 <= value <= 10:
            raise ValueError("priority must be between 1 and 10")
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": str(self.id),
            "session_id": self.session_id,
            "user_id": self.user_id,
            "context_type": self.context_type,
            "content": self.content,
            "metadata": self.metadata,
            "priority": self.priority,
            "is_active": self.is_active,
            "ttl": self.ttl,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "embedding_dim": self.embedding_dim,
        }
    
    def update_access(self):
        """更新访问统计"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
    
    def calculate_expiry(self):
        """计算过期时间"""
        if self.ttl:
            from datetime import timedelta
            self.expires_at = self.created_at + timedelta(seconds=self.ttl)
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at


class ContextHistory(Base):
    """上下文历史表 - 用于审计和版本控制"""
    
    __tablename__ = "context_history"
    __table_args__ = (
        Index("idx_context_history_context_id", "context_id"),
        Index("idx_context_history_created_at", "created_at"),
        Index("idx_context_history_version", "version"),
    )
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="历史记录ID"
    )
    
    context_id = Column(
        UUID(as_uuid=True),
        ForeignKey("contexts.id", ondelete="CASCADE"),
        nullable=False,
        comment="上下文ID"
    )
    
    version = Column(
        Integer,
        nullable=False,
        comment="版本号"
    )
    
    content = Column(
        JSONB,
        nullable=False,
        comment="内容快照"
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
    context = relationship("Context", backref="history")
    
    def __repr__(self):
        return f"<ContextHistory(context_id={self.context_id}, version={self.version})>"


class ContextStats(Base):
    """上下文统计表 - 用于性能监控"""
    
    __tablename__ = "context_stats"
    __table_args__ = (
        Index("idx_context_stats_session_id", "session_id"),
        Index("idx_context_stats_date", "date"),
        Index("idx_context_stats_type", "context_type"),
        UniqueConstraint("session_id", "date", "context_type", name="uq_session_date_type"),
    )
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="统计ID"
    )
    
    session_id = Column(
        String(100),
        nullable=False,
        comment="会话ID"
    )
    
    date = Column(
        DateTime(timezone=True),
        nullable=False,
        comment="统计日期"
    )
    
    context_type = Column(
        String(50),
        nullable=False,
        comment="上下文类型"
    )
    
    # 统计指标
    total_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="总数量"
    )
    
    active_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="活跃数量"
    )
    
    avg_priority = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="平均优先级"
    )
    
    avg_access_count = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="平均访问次数"
    )
    
    total_access_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="总访问次数"
    )
    
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
        return f"<ContextStats(session_id={self.session_id}, date={self.date}, type={self.context_type})>"