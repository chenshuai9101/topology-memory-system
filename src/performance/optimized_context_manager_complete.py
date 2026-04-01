"""
完整优化的上下文管理器
包含所有性能优化措施
"""

import uuid
import time
import threading
import json
import zlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, OrderedDict
from sortedcontainers import SortedDict
import logging

logger = logging.getLogger(__name__)


# ==================== 基础数据结构 ====================

class LRUCache:
    """LRU缓存实现"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache = OrderedDict()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
    
    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self):
        with self._lock:
            self.cache.clear()


class ReadWriteLock:
    """读写锁实现"""
    
    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0
    
    def acquire_read(self):
        self._read_ready.acquire()
        try:
            self._readers += 1
        finally:
            self._read_ready.release()
    
    def release_read(self):
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()
        finally:
            self._read_ready.release()
    
    def acquire_write(self):
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()
    
    def release_write(self):
        self._read_ready.release()


# ==================== 优化后的上下文条目 ====================

@dataclass
class OptimizedContextEntry:
    """优化后的上下文条目"""
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
    _is_expired: bool = False  # 惰性过期标记
    
    def __post_init__(self):
        if self.ttl:
            self.expires_at = self.created_at + timedelta(seconds=self.ttl)
    
    def update(self, update_data: Dict[str, Any]) -> None:
        for key, value in update_data.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        self.updated_at = datetime.now()
        
        if self.ttl:
            self.expires_at = self.updated_at + timedelta(seconds=self.ttl)
        
        self._is_expired = False
    
    def mark_accessed(self) -> None:
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def is_expired(self) -> bool:
        if self._is_expired:
            return True
        if self.expires_at and datetime.now() > self.expires_at:
            self._is_expired = True
            return True
        return False


# ==================== 优化后的上下文管理器 ====================

class OptimizedContextManager:
    """完整优化的上下文管理器"""
    
    def __init__(self, 
                 max_contexts_per_session: int = 100,
                 cleanup_interval: int = 300,
                 enable_cache: bool = True,
                 cache_capacity: int = 1000):
        """
        初始化优化后的上下文管理器
        
        Args:
            max_contexts_per_session: 每个会话最大上下文数量
            cleanup_interval: 清理间隔(秒)
            enable_cache: 是否启用缓存
            cache_capacity: 缓存容量
        """
        self.max_contexts_per_session = max_contexts_per_session
        self.cleanup_interval = cleanup_interval
        
        # 核心数据结构
        self.context_by_id: Dict[str, OptimizedContextEntry] = {}
        self.contexts_by_session: Dict[str, Set[str]] = defaultdict(set)
        self.priority_index: SortedDict[int, Set[str]] = SortedDict()
        
        # 缓存系统
        if enable_cache:
            self.cache = LRUCache(capacity=cache_capacity)
        else:
            self.cache = None
        
        # 并发控制
        self.rw_lock = ReadWriteLock()
        
        # 清理相关
        self.last_cleanup = time.time()
        
        logger.info(f"OptimizedContextManager initialized")
    
    def create_context(self, session_id: str, user_id: str, context_type: str,
                      content: Dict[str, Any], metadata: Dict[str, Any],
                      priority: int, ttl: Optional[int] = None) -> str:
        """
        创建新的上下文
        
        Returns:
            str: 创建的上下文ID
        """
        start_time = time.perf_counter()
        
        try:
            self.rw_lock.acquire_write()
            
            # 生成唯一ID
            context_id = str(uuid.uuid4())
            now = datetime.now()
            
            # 创建上下文条目
            context_entry = OptimizedContextEntry(
                id=context_id,
                session_id=session_id,
                user_id=user_id,
                context_type=context_type,
                content=content,
                metadata=metadata,
                priority=priority,
                ttl=ttl,
                created_at=now,
                updated_at=now
            )
            
            # 检查会话上下文数量限制
            session_contexts = self.contexts_by_session.get(session_id, set())
            if len(session_contexts) >= self.max_contexts_per_session:
                self._evict_low_priority_optimized(session_id)
            
            # 存储上下文
            self.context_by_id[context_id] = context_entry
            self.contexts_by_session[session_id].add(context_id)
            
            # 更新优先级索引
            if priority not in self.priority_index:
                self.priority_index[priority] = set()
            self.priority_index[priority].add(context_id)
            
            # 更新缓存
            if self.cache:
                self.cache.put(context_id, context_entry)
            
            # 定期清理
            self._cleanup_if_needed()
            
            elapsed = (time.perf_counter() - start_time) * 1000
            if elapsed > 50:
                logger.warning(f"Create context took {elapsed:.3f}ms (exceeds 50ms target)")
            
            return context_id
            
        finally:
            self.rw_lock.release_write()
    
    def get_context(self, session_id: str, context_id: str) -> Optional[OptimizedContextEntry]:
        """
        获取上下文
        
        Returns:
            Optional[OptimizedContextEntry]: 上下文条目或None
        """
        start_time = time.perf_counter()
        
        try:
            self.rw_lock.acquire_read()
            
            # 检查缓存
            if self.cache:
                cached = self.cache.get(context_id)
                if cached:
                    elapsed = (time.perf_counter() - start_time) * 1000
                    if elapsed > 10:
                        logger.warning(f"Get cached context took {elapsed:.3f}ms")
                    return cached
            
            # 从主存储获取
            context_entry = self.context_by_id.get(context_id)
            if not context_entry:
                elapsed = (time.perf_counter() - start_time) * 1000
                return None
            
            # 检查是否过期
            if context_entry.is_expired():
                self._delete_context_no_lock(session_id, context_id)
                elapsed = (time.perf_counter() - start_time) * 1000
                return None
            
            # 标记访问
            context_entry.mark_accessed()
            
            # 更新缓存
            if self.cache:
                self.cache.put(context_id, context_entry)
            
            elapsed = (time.perf_counter() - start_time) * 1000
            if elapsed > 10:
                logger.warning(f"Get context took {elapsed:.3f}ms (exceeds 10ms target)")
            
            return context_entry
            
        finally:
            self.rw_lock.release_read()
    
    def update_context(self, session_id: str, context_id: str,
                      content: Optional[Dict[str, Any]] = None,
                      metadata: Optional[Dict[str, Any]] = None,
                      priority: Optional[int] = None) -> bool:
        """
        更新上下文
        
        Returns:
            bool: 是否成功更新
        """
        start_time = time.perf_counter()
        
        try:
            self.rw_lock.acquire_write()
            
            context_entry = self.context_by_id.get(context_id)
            if not context_entry:
                elapsed = (time.perf_counter() - start_time) * 1000
                return False
            
            # 检查是否过期
            if context_entry.is_expired():
                self._delete_context_no_lock(session_id, context_id)
                elapsed = (time.perf_counter() - start_time) * 1000
                return False
            
            # 记录旧优先级
            old_priority = context_entry.priority
            
            # 更新上下文
            update_dict = {}
            if content is not None:
                update_dict['content'] = content
            if metadata is not None:
                update_dict['metadata'] = metadata
            if priority is not None:
                update_dict['priority'] = priority
            
            context_entry.update(update_dict)
            
            # 如果优先级改变，更新索引
            if priority is not None and priority != old_priority:
                self._update_context_priority(context_id, old_priority, priority)
            
            # 更新缓存
            if self.cache:
                self.cache.put(context_id, context_entry)
            
            elapsed = (time.perf_counter() - start_time) * 1000
            if elapsed > 20:
                logger.warning(f"Update context took {elapsed:.3f}ms")
            
            return True
            
        finally:
            self.rw_lock.release_write()
    
    def delete_context(self, session_id: str, context_id: str) -> bool:
        """
        删除上下文
        
        Returns:
            bool: 是否成功删除
        """
        start_time = time.perf_counter()
        
        try:
            self.rw_lock.acquire_write()
            return self._delete_context_no_lock(session_id, context_id)
        finally:
            self.rw_lock.release_write()
            elapsed = (time.perf_counter() - start_time) * 1000
            if elapsed > 10:
                logger.warning(f"Delete context took {elapsed:.3f}ms")
    
    def list_contexts(self, session_id: str, 
                     page: int = 1, page_size: int = 20) -> Tuple[List[OptimizedContextEntry], int]:
        """
        列出会话中的所有上下文
        
        Returns:
            Tuple[List[OptimizedContextEntry], int]: 上下文列表和总数量
        """
        start_time = time.perf_counter()
        
        try:
            self.rw_lock.acquire_read()
            
            session_contexts = self.contexts_by_session.get(session_id, set())
            
            # 过滤过期上下文
            valid_contexts = []
            expired_contexts = []
            
            for context_id in session_contexts:
                context_entry = self.context_by_id.get(context_id)
                if context_entry and not context_entry.is_expired():
                    valid_contexts.append(context_entry)
                elif context_entry:
                    expired_contexts.append(context_id)
            
            # 清理过期上下文
            if expired_contexts:
                self.rw_lock.release_read()
                try:
                    self.rw_lock.acquire_write()
                    for context_id in expired_contexts:
                        self._delete_context_no_lock(session_id, context_id)
                finally:
                    self.rw_lock.release_write()
                    self.rw_lock.acquire_read()
            
            # 按优先级排序
            valid_contexts.sort(key=lambda x: x.priority, reverse=True)
            
            # 分页
            total = len(valid_contexts)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            
            paginated_contexts = valid_contexts[start_idx:end_idx]
            
            elapsed = (time.perf_counter() - start_time) * 1000
            if elapsed > 20:
                logger.warning(f"List contexts took {elapsed:.3f}ms (exceeds 20ms target)")
            
            return paginated_contexts, total
            
        finally:
            self.rw_lock.release_read()
    
    def search_contexts(self, query: Dict[str, Any],
                       session_id: Optional[str] = None) -> List[OptimizedContextEntry]:
        """
        搜索上下文
        
        Returns:
            List[OptimizedContextEntry]: 匹配的上下文列表
        """
        start_time = time.perf_counter()
        
        try:
            self.rw_lock.acquire_read()
            
            results = []
            
            # 确定搜索范围
            if session_id:
                search_sessions = [(session_id, self.contexts_by_session.get(session_id, set()))]
            else:
                search_sessions = list(self.contexts_by_session.items())
            
            for sess_id, context_ids in search_sessions:
                for context_id in context_ids:
                    context_entry = self.context_by_id.get(context_id)
                    if not context_entry:
                        continue
                    
                    # 检查是否过期
                    if context_entry.is_expired():
                        continue
                    
                    # 匹配查询条件
                    if self._matches_query(context_entry, query):
                        results.append(context_entry)
            
            # 按优先级排序
            results.sort(key=lambda x: x.priority, reverse=True)
            
            elapsed = (time.perf_counter() - start_time) * 1000
            if elapsed > 30:
                logger.warning(f"Search contexts took {elapsed:.3f}ms (exceeds 30ms target)")
            
            return results
            
        finally:
            self.rw_lock.release_read()
    
    def _evict_low_priority_optimized(self, session_id: str) -> None:
        """优化的低优先级上下文逐出"""
        session_contexts = self.contexts_by_session.get(session_id, set())
        if not session_contexts:
            return
        
        # 找到最低优先级
        min_priority = None
        for priority in self.priority_index:
            if self.priority_index[priority] & session_contexts:
                min_priority = priority
                break
        
        if min_priority is None:
            return
        
        # 找到该优先级的所有上下文
        low_priority_contexts = self.priority_index[min_priority] & session_contexts
        
        # 找到访问次数最少的上下文
        if low_priority_contexts:
            context_to_evict = min(
                low_priority_contexts,
                key=lambda cid: self.context_by_id[cid].access_count
            )
            self._delete_context_no_lock(session_id, context_to_evict)
    
    def _update_context_priority(self, context_id: str, old_priority: int, new_priority: int) -> None:
        """更新上下文优先级"""
        # 从旧优先级索引中删除
        if old_priority in self.priority_index:
            self.priority_index[old_priority].discard(context_id)
            if not self.priority_index[old_priority]:
                del self.priority_index[old_priority]
        
        # 添加到新优先级索引
        if new_priority not in self.priority_index:
            self.priority_index[new_priority] = set()
        self.priority_index[new_priority].add(context_id)
    
    def _delete_context_no_lock(self, session_id: str, context_id: str) -> bool:
        """无锁删除上下文（内部使用）"""
        context_entry = self.context_by_id.get(context_id)
        if not context_entry:
            return False
        
        # 从存储中删除
        del self.context_by_id[context_id]
        
        # 从会话索引中删除
        if session_id in self.contexts_by_session:
            self.contexts_by_session[session_id].discard(context_id)
            if not self.contexts_by_session[session_id]:
                del self.contexts_by_session[session_id]
        
        # 从优先级索引中删除
        priority = context_entry.priority
        if priority in self.priority_index:
            self.priority_index[priority].discard(context_id)
            if not self.priority_index[priority]:
                del self.priority_index[priority]
        
        # 从缓存中删除
        if self.cache:
            self.cache.delete(context_id)
        
        return True
    
    def _matches_query(self, context_entry: OptimizedContextEntry, query: Dict[str, Any]) -> bool:
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
                return False
        
        return True
    
    def _cleanup_if_needed(self) -> None:
        """如果需要则执行清理"""
        current_time = time.time()
        if current_time - self.last_cleanup >= self.cleanup_interval:
            self