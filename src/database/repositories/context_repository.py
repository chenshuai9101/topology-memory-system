"""
上下文仓库
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from uuid import UUID

from sqlalchemy import and_, or_, desc, func
from sqlalchemy.orm import joinedload

from .base_repository import BaseRepository
from ..models.contexts import Context, ContextHistory, ContextStats

logger = logging.getLogger(__name__)


class ContextRepository(BaseRepository[Context]):
    """上下文仓库"""
    
    def __init__(self):
        super().__init__(Context)
    
    def get_by_session(self, session_id: str, active_only: bool = True) -> List[Context]:
        """根据会话ID获取上下文"""
        with self.get_session() as session:
            query = session.query(Context).filter(Context.session_id == session_id)
            
            if active_only:
                query = query.filter(Context.is_active == True)
            
            query = query.order_by(desc(Context.priority), desc(Context.created_at))
            return query.all()
    
    def get_by_user(self, user_id: str, limit: int = 100) -> List[Context]:
        """根据用户ID获取上下文"""
        with self.get_session() as session:
            return (
                session.query(Context)
                .filter(Context.user_id == user_id)
                .filter(Context.is_active == True)
                .order_by(desc(Context.created_at))
                .limit(limit)
                .all()
            )
    
    def get_by_type(self, context_type: str, limit: int = 50) -> List[Context]:
        """根据类型获取上下文"""
        with self.get_session() as session:
            return (
                session.query(Context)
                .filter(Context.context_type == context_type)
                .filter(Context.is_active == True)
                .order_by(desc(Context.created_at))
                .limit(limit)
                .all()
            )
    
    def get_active_contexts(self, hours: int = 24) -> List[Context]:
        """获取活跃的上下文"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self.get_session() as session:
            return (
                session.query(Context)
                .filter(Context.is_active == True)
                .filter(Context.created_at >= cutoff_time)
                .order_by(desc(Context.priority), desc(Context.created_at))
                .all()
            )
    
    def search_content(self, query_text: str, limit: int = 20) -> List[Context]:
        """搜索上下文内容"""
        with self.get_session() as session:
            return (
                session.query(Context)
                .filter(
                    or_(
                        Context.content.cast(String).ilike(f"%{query_text}%"),
                        Context.metadata.cast(String).ilike(f"%{query_text}%")
                    )
                )
                .filter(Context.is_active == True)
                .order_by(desc(Context.created_at))
                .limit(limit)
                .all()
            )
    
    def get_expired_contexts(self) -> List[Context]:
        """获取已过期的上下文"""
        with self.get_session() as session:
            return (
                session.query(Context)
                .filter(Context.expires_at <= datetime.utcnow())
                .filter(Context.is_active == True)
                .all()
            )
    
    def deactivate_expired(self) -> int:
        """停用已过期的上下文"""
        expired_contexts = self.get_expired_contexts()
        deactivated_count = 0
        
        for context in expired_contexts:
            context.is_active = False
            deactivated_count += 1
        
        if expired_contexts:
            with self.get_session() as session:
                session.bulk_save_objects(expired_contexts)
                session.commit()
        
        return deactivated_count
    
    def update_priority(self, context_id: UUID, new_priority: int) -> Optional[Context]:
        """更新上下文优先级"""
        return self.update(context_id, priority=new_priority)
    
    def increment_access(self, context_id: UUID) -> Optional[Context]:
        """增加访问计数"""
        with self.get_session() as session:
            context = session.query(Context).filter(Context.id == context_id).first()
            if context:
                context.update_access()
                session.commit()
                session.refresh(context)
            return context
    
    def get_context_with_associations(self, context_id: UUID) -> Optional[Context]:
        """获取上下文及其关联"""
        with self.get_session() as session:
            return (
                session.query(Context)
                .options(joinedload(Context.associations))
                .filter(Context.id == context_id)
                .first()
            )
    
    def get_stats_by_session(self, session_id: str) -> Dict[str, Any]:
        """获取会话统计"""
        with self.get_session() as session:
            stats = (
                session.query(
                    func.count(Context.id).label('total_count'),
                    func.sum(Context.access_count).label('total_access'),
                    func.avg(Context.priority).label('avg_priority'),
                    func.max(Context.created_at).label('latest_created')
                )
                .filter(Context.session_id == session_id)
                .filter(Context.is_active == True)
                .first()
            )
            
            return {
                'total_count': stats.total_count or 0,
                'total_access': stats.total_access or 0,
                'avg_priority': float(stats.avg_priority or 0),
                'latest_created': stats.latest_created
            }
    
    def cleanup_old_contexts(self, days: int = 30) -> int:
        """清理旧的上下文"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        with self.get_session() as session:
            # 先标记为不活跃
            result = (
                session.query(Context)
                .filter(Context.created_at < cutoff_time)
                .filter(Context.is_active == True)
                .update({'is_active': False})
            )
            session.commit()
            
            return result


class ContextHistoryRepository(BaseRepository[ContextHistory]):
    """上下文历史仓库"""
    
    def __init__(self):
        super().__init__(ContextHistory)
    
    def get_by_context(self, context_id: UUID, limit: int = 50) -> List[ContextHistory]:
        """根据上下文ID获取历史记录"""
        with self.get_session() as session:
            return (
                session.query(ContextHistory)
                .filter(ContextHistory.context_id == context_id)
                .order_by(desc(ContextHistory.version))
                .limit(limit)
                .all()
            )
    
    def get_latest_version(self, context_id: UUID) -> Optional[ContextHistory]:
        """获取最新版本"""
        with self.get_session() as session:
            return (
                session.query(ContextHistory)
                .filter(ContextHistory.context_id == context_id)
                .order_by(desc(ContextHistory.version))
                .first()
            )
    
    def create_version(self, context: Context, changed_by: str = None, 
                      change_reason: str = None) -> ContextHistory:
        """创建新版本"""
        # 获取当前最大版本号
        latest = self.get_latest_version(context.id)
        next_version = (latest.version + 1) if latest else 1
        
        history = ContextHistory(
            context_id=context.id,
            version=next_version,
            content=context.content,
            metadata=context.metadata,
            changed_by=changed_by,
            change_reason=change_reason
        )
        
        return self.create(**history.to_dict())


class ContextStatsRepository(BaseRepository[ContextStats]):
    """上下文统计仓库"""
    
    def __init__(self):
        super().__init__(ContextStats)
    
    def get_daily_stats(self, date: datetime = None) -> List[ContextStats]:
        """获取每日统计"""
        if date is None:
            date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        with self.get_session() as session:
            return (
                session.query(ContextStats)
                .filter(ContextStats.date == date)
                .all()
            )
    
    def update_daily_stats(self, session_id: str, context_type: str) -> ContextStats:
        """更新每日统计"""
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        with self.get_session() as session:
            # 获取或创建统计记录
            stats = (
                session.query(ContextStats)
                .filter(
                    and_(
                        ContextStats.session_id == session_id,
                        ContextStats.date == today,
                        ContextStats.context_type == context_type
                    )
                )
                .first()
            )
            
            if not stats:
                stats = ContextStats(
                    session_id=session_id,
                    date=today,
                    context_type=context_type
                )
                session.add(stats)
            
            # 更新统计
            session_stats = (
                session.query(
                    func.count(Context.id).label('count'),
                    func.avg(Context.priority).label('avg_priority'),
                    func.sum(Context.access_count).label('total_access'),
                    func.count(Context.id).filter(Context.is_active == True).label('active_count')
                )
                .filter(Context.session_id == session_id)
                .filter(Context.context_type == context_type)
                .filter(Context.created_at >= today)
                .first()
            )
            
            stats.total_count = session_stats.count or 0
            stats.active_count = session_stats.active_count or 0
            stats.avg_priority = float(session_stats.avg_priority or 0)
            stats.total_access_count = session_stats.total_access or 0
            
            if stats.total_count > 0:
                stats.avg_access_count = stats.total_access_count / stats.total_count
            
            session.commit()
            session.refresh(stats)
            
            return stats