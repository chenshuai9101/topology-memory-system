"""
优化后的上下文管理器
实现高性能的上下文管理，确保管理开销 < 50ms
"""

import uuid
import time
import threading
import json
import zlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque, OrderedDict
from sortedcontainers import SortedDict
import logging

from ..api.schemas import ContextCreate, ContextUpdate, ContextResponse


logger = logging.getLogger(__name__)


# ==================== 基础数据结构 ====================

class LRUCache:
    """LRU缓存实现"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache = OrderedDict()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """设置缓存值"""
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self.cache.clear()
    
    def __len__(self) -> int:
        """获取缓存大小"""
        with self._lock:
            return len(self.cache)


class ReadWriteLock:
    """读写锁实现"""
    
    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0
    
    def acquire_read(self):
        """获取读锁"""
        self._read_ready.acquire()
        try:
            self._readers += 1
        finally:
            self._read_ready.release()
    
    def release_read(self):
        """释放读锁"""
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()
        finally:
            self._read_ready.release()
    
    def acquire_write(self):
        """获取写锁"""
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()
    
    def release_write(self):
        """释放写锁"""
        self._read_ready.release()


class HotspotDetector:
    """热点数据检测器"""
    
    def __init__(self, threshold: int = 10, decay_factor: float = 0.9):
        self.threshold = threshold
        self.decay_factor = decay_factor
        self.access_counts: Dict[str, float] = defaultdict(float)
        self._lock = threading.RLock()
    
    def record_access(self, key: str, weight: float = 1.0):
        """记录访问"""
        with self._lock:
            # 应用衰减
            for k in list(self.access_counts.keys()):
                self.access_counts[k] *= self.decay_factor
            
            # 记录新访问
            self.access_counts[key] += weight
    
    def get_hotspots(self, limit: int = 10) -> List[str]:
        """获取热点数据"""
        with self._lock:
            sorted_items = sorted(
                self.access_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return [k for k, v in sorted_items[:limit] if v >= self.threshold]
    
    def clear(self):
        """清空统计数据"""
        with self._lock:
            self.access_counts.clear()


class PerformanceMetrics:
    """性能指标收集器"""
    
    def __init__(self):
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
    
    def record_operation(self, operation: str, time_ms: float):
        """记录操作性能"""
        with self._lock:
            self.operation_times[operation].append(time_ms)
            self.operation_counts[operation] += 1
    
    def get_stats(self, operation: str = None) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            if operation:
                times = self.operation_times.get(operation, [])
                if not times:
                    return {}
                
                return {
                    'count': len(times),
                    'avg_ms': sum(times) / len(times),
                    'p95_ms': sorted(times)[int(len(times) * 0.95)] if times else 0,
                    'p99_ms': sorted(times)[int(len(times) * 0.99)] if times else 0,
                    'max_ms': max(times) if times else 0,
                    'min_ms': min(times) if times else 0
                }
            else:
                return {
                    op: self.get_stats(op)
                    for op in self.operation_times.keys()
                }
    
    def clear(self):
        """清空统计数据"""
        with self._lock:
            self.operation_times.clear()
            self.operation_counts.clear()


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
        
        # 重置过期标记
        self._is_expired = False
    
    def mark_accessed(self) -> None:
        """标记为已访问"""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def is_expired(self) -> bool:
        """惰性过期检查"""
        if self._is_expired:
            return True
        if self.expires_at and datetime.now() > self.expires_at:
            self._is_expired = True
            return True
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


class CompressedContextEntry:
    """压缩存储的上下文条目"""
    
    def __init__(self, entry: OptimizedContextEntry):
        self._compressed_data = self._compress(entry)
        self.id = entry.id
        self.priority = entry.priority
        self.session_id = entry.session_id
        self.user_id = entry.user_id
        self.context_type = entry.context_type
    
    def _compress(self, entry: OptimizedContextEntry) -> bytes:
        """压缩数据"""
        data = {
            'id': entry.id,
            'session_id': entry.session_id,
            'user_id': entry.user_id,
            'context_type': entry.context_type,
            'content': entry.content,
            'metadata': entry.metadata,
            'priority': entry.priority,
            'ttl': entry.ttl,
            'created_at': entry.created_at.isoformat(),
            'updated_at': entry.updated_at.isoformat(),
            'expires_at': entry.expires_at.isoformat() if entry.expires_at else None,
            'access_count': entry.access_count,
            'last_accessed': entry.last_accessed.isoformat() if entry.last_accessed else None
        }
        json_str = json.dumps(data, separators=(',', ':'))
        return zlib.compress(json_str.encode('utf-8'))
    
    def decompress(self) -> OptimizedContextEntry:
        """解压数据"""
        json_str = zlib.decompress(self._compressed_data).decode('utf-8')
        data = json.loads(json_str)
        
        return OptimizedContextEntry(
            id=data['id'],
            session_id=data['session_id'],
            user_id=data['user_id'],
            context_type=data['context_type'],
            content=data['content'],
            metadata=data['metadata'],
            priority=data['priority'],
            ttl=data['ttl'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            expires_at=datetime.fromisoformat(data['expires_at']) if data['expires_at'] else None,
            access_count=data['access_count'],
            last_accessed=datetime.fromisoformat(data['last_accessed']) if data['last_accessed'] else None
        )
    
    def get_size(self) -> int:
        """获取压缩后的大小"""
        return len(self._compressed_data)


# ==================== 优化后的上下文管理器 ====================

class OptimizedContextManager:
    """优化后的上下文管理器"""
    
    def __init__(self, 
                 max_contexts_per_session: int = 100,
                 cleanup_interval: int = 300,
                 enable_cache: bool = True,
                 enable_compression: bool = False,
                 cache_capacity: int = 1000):
        """
        初始化优化后的上下文管理器
        
        Args:
            max_contexts_per_session: 每个会话最大上下文数量
            cleanup_interval: 清理间隔(秒)
            enable_cache: 是否启用缓存
            enable_compression: 是否启用压缩存储
            cache_capacity: 缓存容量
        """
        self.max_contexts_per_session = max_contexts_per_session
        self.cleanup_interval = cleanup_interval
        self.enable_compression = enable_compression
        
        # 核心数据结构
        self.context_by_id: Dict[str, Any] = {}  # 存储OptimizedContextEntry或CompressedContextEntry
        self.contexts_by_session: Dict[str, Set[str]] = defaultdict(set)
        self.priority_index: SortedDict[int, Set[str]] = SortedDict()
        
        # 缓存系统
        if enable_cache:
            self.cache = LRUCache(capacity=cache_capacity)
            self.hotspot_detector = HotspotDetector()
        else:
            self.cache = None
            self.hotspot_detector = None
        
        # 并发控制
        self.rw_lock = ReadWriteLock()
        
        # 性能监控
        self.metrics = PerformanceMetrics()
        
        # 清理相关
        self.last_cleanup = time.time()
        self._cleanup_lock = threading.Lock()
        
        logger.info(f"OptimizedContextManager initialized with "
                   f"max_contexts_per_session={max_contexts_per_session}, "
                   f"enable_cache={enable_cache}, "
                   f"enable_compression={enable_compression}")
    
    def create_context(self, context_data: ContextCreate) -> ContextResponse:
        """
        创建新的上下文
        
        Args:
            context_data: 上下文创建数据
            
        Returns:
            ContextResponse: 创建的上下文响应
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
            
            # 检查会话上下文数量限制
            session_contexts = self.contexts_by_session.get(context_data.session_id, set())
            if len(session_contexts) >= self.max_contexts_per_session:
                self._evict_low_priority_optimized(context_data.session_id)
            
            # 存储上下文
            if self.enable_compression:
                self.context_by_id[context_id] = CompressedContextEntry(context_entry)
            else:
                self.context_by_id[context_id] = context_entry
            
            # 更新索引
            self.contexts_by_session[context_data.session_id].add(context_id)
            
            # 更新优先级索引
            if context_data.priority not in self.priority_index:
                self.priority_index[context_data.priority] = set()
            self.priority_index[context_data.priority].add(context_id)
            
            # 更新缓存
            if self.cache:
                self.cache.put(context_id, context_entry)
            
            response = context_entry.to_response()
            
            # 记录性能指标
            elapsed = (time.perf_counter() - start_time) * 1000
            self.metrics.record_operation('create', elapsed)
            
            logger.debug(f"Created context {context_id} in {elapsed:.3f}ms")
            
            return response
            
        finally:
            self.rw_lock.release_write()
    
    def get_context(self, session_id: str, context_id: str) -> Optional[ContextResponse]:
        """
        获取上下文
        
        Args:
            session_id: 会话ID
            context_id: 上下文ID
            
        Returns:
            Optional[ContextResponse]: 上下文响应或None
        """
        start_time = time.perf_counter()
        
        try:
            self.rw_lock.acquire_read()
            
            # 检查缓存
            if self.cache:
                cached = self.cache.get(context_id)
                if cached:
                    # 记录热点访问
                    if self.hotspot_detector:
                        self.hotspot_detector.record_access(context_id)
                    
                    elapsed = (time.perf_counter() - start_time) * 1000
                    self.metrics.record_operation('get_cached', elapsed)
                    logger.debug(f"Cache hit for context {context_id} in {elapsed:.3f}ms")
                    return cached.to_response()
            
            # 从主存储获取
            stored_entry = self.context_by_id.get(context_id)
            if not stored_entry:
                elapsed = (time.perf_counter() - start_time) * 1000
                self.metrics.record_operation('get_miss', elapsed)
                logger.debug(f"Context {context_id} not found")
                return None
            
            # 解压数据（如果需要）
            if self.enable_compression:
                context_entry = stored_entry.decompress()
            else:
                context_entry = stored_entry
            
            # 检查是否过期（惰性检查）
            if context_entry.is_expired():
                self._delete_context_no_lock(session_id, context_id)
                elapsed = (time.perf_counter() - start_time) * 1000
                self.metrics.record_operation('get_expired', elapsed)
                logger.debug(f"Context {context_id} expired")
                return None
            
            # 标记访问
            context_entry.mark_accessed()
            
            # 更新缓存
            if self.cache:
                self.cache.put(context_id, context_entry)
                if self.hotspot_detector:
                    self.hotspot_detector.record_access(context_id)
            
            response = context_entry.to_response()
            
            elapsed = (time.perf_counter() - start_time) * 1000
            self.metrics.record_operation('get', elapsed)
            
            logger.debug(f"Retrieved context {context_id} in {elapsed:.3f}ms")
            
            return response
            
        finally:
            self.rw_lock.release_read()
    
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
        start_time = time.perf_counter()
        
        try:
            self.rw_lock.acquire_write()
            
            # 获取上下文
            stored_entry = self.context_by_id.get(context_id)
            if not stored_entry:
                elapsed = (time.perf_counter() - start_time) * 1000
                self.metrics.record_operation('update_miss', elapsed)
                return None
            
            # 解压数据（如果需要）
            if self.enable_compression:
                context_entry = stored_entry.decompress()
            else:
                context_entry = stored_entry
            
            # 检查是否过期
            if context_entry.is_expired():
                self._delete_context_no_lock(session_id, context_id)
                elapsed = (time.perf_counter() - start_time) * 1000
                self.metrics.record_operation('update_expired', elapsed)
                return None
            
            # 记录旧优先级
            old_priority = context_entry.priority
            
            # 更新上下文
            update_dict = {}
            if update_data.content is not None:
                update_dict['