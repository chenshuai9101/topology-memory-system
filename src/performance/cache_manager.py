"""
缓存管理器
实现高性能缓存系统
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict
import logging


logger = logging.getLogger(__name__)


class CacheEntry:
    """缓存条目"""
    
    def __init__(self, value: Any, ttl: Optional[int] = None):
        """
        初始化缓存条目
        
        Args:
            value: 缓存值
            ttl: 生存时间(秒)，None表示永不过期
        """
        self.value = value
        self.created_at = time.time()
        self.access_count = 0
        self.last_accessed = self.created_at
        
        if ttl:
            self.expires_at = self.created_at + ttl
        else:
            self.expires_at = None
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at:
            return time.time() > self.expires_at
        return False
    
    def mark_accessed(self) -> None:
        """标记为已访问"""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def get_age(self) -> float:
        """获取年龄(秒)"""
        return time.time() - self.created_at
    
    def get_time_to_live(self) -> Optional[float]:
        """获取剩余生存时间"""
        if self.expires_at:
            return max(0, self.expires_at - time.time())
        return None


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300, cleanup_interval: int = 60):
        """
        初始化缓存管理器
        
        Args:
            max_size: 最大缓存条目数
            ttl: 默认生存时间(秒)
            cleanup_interval: 清理间隔(秒)
        """
        self.max_size = max_size
        self.default_ttl = ttl
        self.cleanup_interval = cleanup_interval
        
        # 使用OrderedDict实现LRU缓存
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # 统计信息
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
            "cleanups": 0
        }
        
        # 锁用于线程安全
        self._lock = threading.RLock()
        
        # 最后清理时间
        self._last_cleanup = time.time()
        
        logger.info(f"CacheManager initialized with max_size={max_size}, ttl={ttl}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            default: 默认值
            
        Returns:
            Any: 缓存值或默认值
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # 检查是否过期
                if entry.is_expired():
                    del self._cache[key]
                    self._stats["misses"] += 1
                    logger.debug(f"Cache miss (expired): {key}")
                    return default
                
                # 标记为已访问并移动到最近使用位置
                entry.mark_accessed()
                self._cache.move_to_end(key)
                
                self._stats["hits"] += 1
                logger.debug(f"Cache hit: {key}")
                return entry.value
            else:
                self._stats["misses"] += 1
                logger.debug(f"Cache miss: {key}")
                return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间(秒)，None使用默认值
            
        Returns:
            bool: 是否成功设置
        """
        with self._lock:
            # 如果需要，执行清理
            self._cleanup_if_needed()
            
            # 如果缓存已满，移除最旧的条目
            if len(self._cache) >= self.max_size:
                self._evict_oldest()
            
            # 创建缓存条目
            actual_ttl = ttl if ttl is not None else self.default_ttl
            entry = CacheEntry(value, actual_ttl)
            
            # 设置缓存
            self._cache[key] = entry
            self._cache.move_to_end(key)
            
            self._stats["sets"] += 1
            logger.debug(f"Cache set: {key} (ttl={actual_ttl})")
            
            return True
    
    def delete(self, key: str) -> bool:
        """
        删除缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否成功删除
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats["deletes"] += 1
                logger.debug(f"Cache delete: {key}")
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """
        检查缓存键是否存在且未过期
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否存在
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if entry.is_expired():
                    del self._cache[key]
                    return False
                return True
            return False
    
    def clear(self) -> None:
        """清除所有缓存"""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self._lock:
            current_time = time.time()
            
            # 计算命中率
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0
            
            # 计算缓存使用情况
            total_size = len(self._cache)
            expired_count = sum(1 for entry in self._cache.values() if entry.is_expired())
            
            # 计算平均年龄
            ages = [entry.get_age() for entry in self._cache.values() if not entry.is_expired()]
            avg_age = sum(ages) / len(ages) if ages else 0
            
            return {
                "size": total_size,
                "max_size": self.max_size,
                "usage_percentage": (total_size / self.max_size) * 100 if self.max_size > 0 else 0,
                "expired_entries": expired_count,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": hit_rate,
                "sets": self._stats["sets"],
                "deletes": self._stats["deletes"],
                "evictions": self._stats["evictions"],
                "cleanups": self._stats["cleanups"],
                "average_age_seconds": avg_age,
                "last_cleanup": self._last_cleanup
            }
    
    def get_keys(self, pattern: str = None) -> List[str]:
        """
        获取所有缓存键
        
        Args:
            pattern: 可选的正则表达式模式
            
        Returns:
            List[str]: 缓存键列表
        """
        with self._lock:
            keys = list(self._cache.keys())
            
            if pattern:
                import re
                regex = re.compile(pattern)
                keys = [key for key in keys if regex.match(key)]
            
            return keys
    
    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """
        获取缓存条目详细信息
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[Dict[str, Any]]: 条目信息或None
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # 检查是否过期
                if entry.is_expired():
                    del self._cache[key]
                    return None
                
                return {
                    "key": key,
                    "value_type": type(entry.value).__name__,
                    "created_at": datetime.fromtimestamp(entry.created_at).isoformat(),
                    "last_accessed": datetime.fromtimestamp(entry.last_accessed).isoformat(),
                    "access_count": entry.access_count,
                    "age_seconds": entry.get_age(),
                    "ttl_seconds": entry.get_time_to_live(),
                    "is_expired": entry.is_expired()
                }
            return None
    
    def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """
        递增缓存值（用于计数器）
        
        Args:
            key: 缓存键
            amount: 递增数量
            ttl: 生存时间(秒)
            
        Returns:
            int: 递增后的值
        """
        with self._lock:
            current = self.get(key, 0)
            new_value = current + amount
            self.set(key, new_value, ttl)
            return new_value
    
    def decrement(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """
        递减缓存值（用于计数器）
        
        Args:
            key: 缓存键
            amount: 递减数量
            ttl: 生存时间(秒)
            
        Returns:
            int: 递减后的值
        """
        return self.increment(key, -amount, ttl)
    
    def set_multi(self, items: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """
        批量设置缓存值
        
        Args:
            items: 键值对字典
            ttl: 生存时间(秒)
        """
        with self._lock:
            for key, value in items.items():
                self.set(key, value, ttl)
    
    def get_multi(self, keys: List[str]) -> Dict[str, Any]:
        """
        批量获取缓存值
        
        Args:
            keys: 缓存键列表
            
        Returns:
            Dict[str, Any]: 键值对字典
        """
        with self._lock:
            result = {}
            for key in keys:
                value = self.get(key)
                if value is not None:
                    result[key] = value
            return result
    
    def delete_multi(self, keys: List[str]) -> int:
        """
        批量删除缓存值
        
        Args:
            keys: 缓存键列表
            
        Returns:
            int: 成功删除的数量
        """
        with self._lock:
            deleted_count = 0
            for key in keys:
                if self.delete(key):
                    deleted_count += 1
            return deleted_count
    
    def _evict_oldest(self) -> None:
        """逐出最旧的缓存条目"""
        if self._cache:
            # 移除第一个（最旧的）条目
            key, entry = next(iter(self._cache.items()))
            del self._cache[key]
            self._stats["evictions"] += 1
            logger.debug(f"Cache eviction: {key}")
    
    def _cleanup_if_needed(self) -> None:
        """如果需要则执行清理"""
        current_time = time.time()
        if current_time - self._last_cleanup >= self.cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = current_time
    
    def _cleanup_expired(self) -> None:
        """清理过期缓存条目"""
        with self._lock:
            expired_keys = []
            
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            # 删除过期条目
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                self._stats["cleanups"] += 1
                logger.debug(f"Cache cleanup: removed {len(expired_keys)} expired entries")
    
    def clear_stats(self) -> None:
        """清除统计信息"""
        with self._lock:
            self._stats = {
                "hits": 0,
                "misses": 0,
                "sets": 0,
                "deletes": 0,
                "evictions": 0,
                "cleanups": 0
            }
            logger.debug("Cache stats cleared")
    
    def __contains__(self, key: str) -> bool:
        """检查键是否在缓存中"""
        return self.exists(key)
    
    def __len__(self) -> int:
        """获取缓存大小"""
        with self._lock:
            # 先清理过期条目
            self._cleanup_expired()
            return len(self._cache)
    
    def __getitem__(self, key: str) -> Any:
        """获取缓存值（支持下标访问）"""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value
    
    def __setitem__(self, key: str, value: Any) -> None:
        """设置缓存值（支持下标访问）"""
        self.set(key, value)
    
    def __delitem__(self, key: str) -> None:
        """删除缓存值（支持下标访问）"""
        if not self.delete(key):
            raise KeyError(key)


# 全局缓存实例
_global_cache = None


def get_global_cache(max_size: int = 1000, ttl: int = 300) -> CacheManager:
    """
    获取全局缓存实例（单例模式）
    
    Args:
        max_size: 最大缓存条目数
        ttl: 默认生存时间(秒)
        
    Returns:
        CacheManager: 全局缓存管理器实例
    """
    global _global_cache
    
    if _global_cache is None:
        _global_cache = CacheManager(max_size=max_size, ttl=ttl)
    
    return _global_cache