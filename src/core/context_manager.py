"""
拓扑记忆上下文管理器 - 核心引擎
负责对话上下文跟踪、记忆关联和优先级管理
"""

import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque

from ..api.schemas import ContextCreate, ContextUpdate, ContextResponse


logger = logging.getLogger(__name__)


class ContextPriority(Enum):
    """上下文优先级枚举"""
    LOW = 1
    MEDIUM = 5
    HIGH = 10


@dataclass
class ContextEntry:
    """上下文条目数据类"""
    id: str
    session_id: str
    user_id: str
    context_type: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    priority: int
    ttl: Optional[int]
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.ttl:
            self.expires_at = self.created_at + timedelta(seconds=self.ttl)
    
    def update(self, update_data: Dict[str, Any]) -> None:
        """更新上下文条目"""
        for key, value in update_data.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        self.updated_at = datetime.now()
        
        # 重新计算过期时间
        if self.ttl:
            self.expires_at = self.updated_at + timedelta(seconds=self.ttl)
    
    def mark_accessed(self) -> None:
        """标记为已访问"""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False
    
    def to_response(self) -> ContextResponse:
        """转换为响应模型"""
        return ContextResponse(
            id=self.id,
            session_id=self.session_id,
            user_id=self.user_id,
            context_type=self.context_type,
            content=self.content,
            metadata=self.metadata,
            priority=self.priority,
            ttl=self.ttl,
            created_at=self.created_at,
            updated_at=self.updated_at,
            expires_at=self.expires_at
        )


class ContextManager:
    """上下文管理器核心类"""
    
    def __init__(self, max_contexts_per_session: int = 100, cleanup_interval: int = 300):
        """
        初始化上下文管理器
        
        Args:
            max_contexts_per_session: 每个会话最大上下文数量
            cleanup_interval: 清理间隔(秒)
        """
        self.max_contexts_per_session = max_contexts_per_session
        self.cleanup_interval = cleanup_interval
        
        # 存储结构: session_id -> {context_id: ContextEntry}
        self.contexts: Dict[str, Dict[str, ContextEntry]] = defaultdict(dict)
        
        # 用户上下文索引: user_id -> [session_id]
        self.user_sessions: Dict[str, List[str]] = defaultdict(list)
        
        # 优先级队列: priority -> [context_id]
        self.priority_queues: Dict[int, deque] = defaultdict(deque)
        
        # 最后清理时间
        self.last_cleanup = time.time()
        
        logger.info(f"ContextManager initialized with max_contexts_per_session={max_contexts_per_session}")
    
    def create_context(self, context_data: ContextCreate) -> ContextResponse:
        """
        创建新的上下文
        
        Args:
            context_data: 上下文创建数据
            
        Returns:
            ContextResponse: 创建的上下文响应
        """
        # 生成唯一ID
        context_id = str(uuid.uuid4())
        now = datetime.now()
        
        # 检查会话上下文数量限制
        session_contexts = self.contexts.get(context_data.session_id, {})
        if len(session_contexts) >= self.max_contexts_per_session:
            # 移除最低优先级的上下文
            self._evict_low_priority_contexts(context_data.session_id)
        
        # 创建上下文条目
        context_entry = ContextEntry(
            id=context_id,
            session_id=context_data.session_id,
            user_id=context_data.user_id,
            context_type=context_data.context_type,
            content=context_data.content,
            metadata=context_data.metadata,
            priority=context_data.priority,
            ttl=context_data.ttl,
            created_at=now,
            updated_at=now
        )
        
        # 存储上下文
        self.contexts[context_data.session_id][context_id] = context_entry
        
        # 更新索引
        if context_data.session_id not in self.user_sessions[context_data.user_id]:
            self.user_sessions[context_data.user_id].append(context_data.session_id)
        
        # 添加到优先级队列
        self.priority_queues[context_data.priority].append(context_id)
        
        logger.info(f"Created context {context_id} for session {context_data.session_id}")
        
        # 定期清理
        self._cleanup_if_needed()
        
        return context_entry.to_response()
    
    def get_context(self, session_id: str, context_id: str) -> Optional[ContextResponse]:
        """
        获取上下文
        
        Args:
            session_id: 会话ID
            context_id: 上下文ID
            
        Returns:
            Optional[ContextResponse]: 上下文响应或None
        """
        session_contexts = self.contexts.get(session_id)
        if not session_contexts:
            return None
        
        context_entry = session_contexts.get(context_id)
        if not context_entry:
            return None
        
        # 检查是否过期
        if context_entry.is_expired():
            self.delete_context(session_id, context_id)
            return None
        
        # 标记为已访问
        context_entry.mark_accessed()
        
        return context_entry.to_response()
    
    def update_context(self, session_id: str, context_id: str, 
                      update_data: ContextUpdate) -> Optional[ContextResponse]:
        """
        更新上下文
        
        Args:
            session_id: 会话ID
            context_id: 上下文ID
            update_data: 更新数据
            
        Returns:
            Optional[ContextResponse]: 更新后的上下文响应或None
        """
        session_contexts = self.contexts.get(session_id)
        if not session_contexts:
            return None
        
        context_entry = session_contexts.get(context_id)
        if not context_entry:
            return None
        
        # 检查是否过期
        if context_entry.is_expired():
            self.delete_context(session_id, context_id)
            return None
        
        # 更新上下文
        update_dict = update_data.dict(exclude_unset=True)
        context_entry.update(update_dict)
        
        # 如果优先级改变，更新优先级队列
        if 'priority' in update_dict:
            self._update_context_priority(context_id, context_entry.priority)
        
        logger.info(f"Updated context {context_id} in session {session_id}")
        
        return context_entry.to_response()
    
    def delete_context(self, session_id: str, context_id: str) -> bool:
        """
        删除上下文
        
        Args:
            session_id: 会话ID
            context_id: 上下文ID
            
        Returns:
            bool: 是否成功删除
        """
        session_contexts = self.contexts.get(session_id)
        if not session_contexts:
            return False
        
        if context_id not in session_contexts:
            return False
        
        # 获取上下文条目以获取优先级
        context_entry = session_contexts[context_id]
        
        # 从存储中删除
        del session_contexts[context_id]
        
        # 从优先级队列中删除
        self._remove_from_priority_queue(context_id, context_entry.priority)
        
        # 如果会话为空，清理索引
        if not session_contexts:
            del self.contexts[session_id]
            self._cleanup_user_sessions(session_id)
        
        logger.info(f"Deleted context {context_id} from session {session_id}")
        
        return True
    
    def list_contexts(self, session_id: str, 
                     page: int = 1, page_size: int = 20) -> Tuple[List[ContextResponse], int]:
        """
        列出会话中的所有上下文
        
        Args:
            session_id: 会话ID
            page: 页码
            page_size: 每页大小
            
        Returns:
            Tuple[List[ContextResponse], int]: 上下文列表和总数量
        """
        session_contexts = self.contexts.get(session_id, {})
        
        # 过滤过期上下文
        valid_contexts = []
        expired_contexts = []
        
        for context_id, context_entry in session_contexts.items():
            if context_entry.is_expired():
                expired_contexts.append(context_id)
            else:
                valid_contexts.append(context_entry)
        
        # 清理过期上下文
        for context_id in expired_contexts:
            self.delete_context(session_id, context_id)
        
        # 按优先级排序
        valid_contexts.sort(key=lambda x: x.priority, reverse=True)
        
        # 分页
        total = len(valid_contexts)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        paginated_contexts = valid_contexts[start_idx:end_idx]
        
        # 转换为响应模型
        responses = [context.to_response() for context in paginated_contexts]
        
        return responses, total
    
    def get_session_contexts(self, session_id: str) -> List[ContextResponse]:
        """
        获取会话的所有上下文
        
        Args:
            session_id: 会话ID
            
        Returns:
            List[ContextResponse]: 上下文响应列表
        """
        session_contexts = self.contexts.get(session_id, {})
        
        responses = []
        expired_contexts = []
        
        for context_id, context_entry in session_contexts.items():
            if context_entry.is_expired():
                expired_contexts.append(context_id)
            else:
                responses.append(context_entry.to_response())
        
        # 清理过期上下文
        for context_id in expired_contexts:
            self.delete_context(session_id, context_id)
        
        return responses
    
    def get_user_contexts(self, user_id: str) -> List[ContextResponse]:
        """
        获取用户的所有上下文
        
        Args:
            user_id: 用户ID
            
        Returns:
            List[ContextResponse]: 上下文响应列表
        """
        session_ids = self.user_sessions.get(user_id, [])
        
        all_contexts = []
        for session_id in session_ids:
            session_contexts = self.get_session_contexts(session_id)
            all_contexts.extend(session_contexts)
        
        return all_contexts
    
    def search_contexts(self, query: Dict[str, Any], 
                       session_id: Optional[str] = None) -> List[ContextResponse]:
        """
        搜索上下文
        
        Args:
            query: 搜索查询
            session_id: 可选的会话ID限制
            
        Returns:
            List[ContextResponse]: 匹配的上下文响应列表
        """
        results = []
        
        # 确定搜索范围
        if session_id:
            search_sessions = [(session_id, self.contexts.get(session_id, {}))]
        else:
            search_sessions = list(self.contexts.items())
        
        for sess_id, contexts in search_sessions:
            for context_entry in contexts.values():
                # 检查是否过期
                if context_entry.is_expired():
                    continue
                
                # 匹配查询条件
                if self._matches_query(context_entry, query):
                    results.append(context_entry.to_response())
        
        # 按优先级排序
        results.sort(key=lambda x: x.priority, reverse=True)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取管理器统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        total_contexts = 0
        total_sessions = len(self.contexts)
        total_users = len(self.user_sessions)
        
        priority_distribution = defaultdict(int)
        type_distribution = defaultdict(int)
        
        for session_contexts in self.contexts.values():
            for context_entry in session_contexts.values():
                if not context_entry.is_expired():
                    total_contexts += 1
                    priority_distribution[context_entry.priority] += 1
                    type_distribution[context_entry.context_type] += 1
        
        return {
            "total_contexts": total_contexts,
            "total_sessions": total_sessions,
            "total_users": total_users,
            "priority_distribution": dict(priority_distribution),
            "type_distribution": dict(type_distribution),
            "max_contexts_per_session": self.max_contexts_per_session,
            "last_cleanup": self.last_cleanup
        }
    
    def _evict_low_priority_contexts(self, session_id: str) -> None:
        """逐出低优先级上下文"""
        session_contexts = self.contexts.get(session_id, {})
        if not session_contexts:
            return
        
        # 找到最低优先级的上下文
        min_priority = min(context.priority for context in session_contexts.values())
        
        # 找到该优先级的所有上下文
        low_priority_contexts = [
            context_id for context_id, context in session_contexts.items()
            if context.priority == min_priority
        ]
        
        # 逐出访问次数最少的上下文
        if low_priority_contexts:
            context_to_evict = min(
                low_priority_contexts,
                key=lambda cid: session_contexts[cid].access_count
            )
            self.delete_context(session_id, context_to_evict)
            logger.debug(f"Evicted low priority context {context_to_evict} from session {session_id}")
    
    def _update_context_priority(self, context_id: str, new_priority: int) -> None:
        """更新上下文优先级"""
        # 从旧优先级队列中删除
        for priority, queue in self.priority_queues.items():
            if context_id in queue:
                queue.remove(context_id)
                break
        
        # 添加到新优先级队列
        self.priority_queues[new_priority].append(context_id)
    
    def _remove_from_priority_queue(self, context_id: str, priority: int) -> None:
        """从优先级队列中删除上下文"""
        queue = self.priority_queues.get(priority)
        if queue and context_id in queue:
            queue.remove(context_id)
    
    def _cleanup_user_sessions(self, session_id: str) -> None:
        """清理用户会话索引"""
        for user_id, sessions in self.user_sessions.items():
            if session_id in sessions:
                sessions.remove(session_id)
                if not sessions:
                    del self.user_sessions[user_id]
                break
    
    def _matches_query(self, context_entry: ContextEntry, query: Dict[str, Any]) -> bool:
        """检查上下文是否匹配查询"""
        for key, value in query.items():
            if key == "context_type":
                if context_entry.context_type != value:
                    return False
            elif key == "priority":
                if context_entry.priority != value:
                    return False
            elif key == "user_id":
                if context_entry.user_id != value:
                    return False
            elif key in context_entry.content:
                if context_entry.content[key] != value:
                    return False
            elif key in context_entry.metadata:
                if context_entry.metadata[key] != value:
                    return False
            else:
                # 键不存在于上下文中
                return False
        
        return True
    
    def _cleanup_if_needed(self) -> None:
        """如果需要则执行清理"""
        current_time = time.time()
        if current_time - self.last_cleanup >= self.cleanup_interval:
            self._cleanup_expired_contexts()
            self.last_cleanup = current_time
    
    def _cleanup_expired_contexts(self) -> None:
        """清理过期上下文"""
        expired_count = 0
        
        for session_id, session_contexts in list(self.contexts.items()):
            expired_contexts = []
            
            for context_id, context_entry in session_contexts.items():
                if context_entry.is_expired():
                    expired_contexts.append(context_id)
            
            # 删除过期上下文
            for context_id in expired_contexts:
                self.delete_context(session_id, context_id)
                expired_count += 1
            
            # 如果会话为空，删除会话
            if not session_contexts:
                del self.contexts[session_id]
        
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired contexts")
    
    def clear_all(self) -> None:
        """清除所有上下文"""
        self.contexts.clear()
        self.user_sessions.clear()
        self.priority_queues.clear()
        logger.info("Cleared all contexts from ContextManager")