"""
缓存适配器 - 为兼容层提供缓存支持
"""

import time
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Union, List
import logging

logger = logging.getLogger(__name__)


class CacheAdapter:
    """缓存适配器，为兼容层提供缓存支持"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化缓存适配器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 缓存存储
        self.cache_store = {}
        
        # 缓存统计
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0,
            'total_size': 0
        }
        
        # 配置参数
        self.default_ttl = self.config.get('default_ttl', 3600)  # 默认1小时
        self.max_size = self.config.get('max_size', 10000)  # 最大缓存项数
        self.enabled = self.config.get('enabled', True)
        
        # 缓存键前缀
        self.key_prefix = "compatibility:"
        
        logger.info(f"Cache adapter initialized (enabled: {self.enabled})")
    
    def _generate_cache_key(self, category: str, *args, **kwargs) -> str:
        """
        生成缓存键
        
        Args:
            category: 缓存类别
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            缓存键字符串
        """
        # 创建键的字符串表示
        key_parts = [category]
        
        # 添加位置参数
        for arg in args:
            key_parts.append(str(arg))
        
        # 添加关键字参数（排序以确保一致性）
        for key in sorted(kwargs.keys()):
            value = kwargs[key]
            key_parts.append(f"{key}={value}")
        
        # 连接所有部分
        key_string = ":".join(key_parts)
        
        # 生成哈希
        hash_digest = hashlib.md5(key_string.encode()).hexdigest()
        
        return f"{self.key_prefix}{category}:{hash_digest}"
    
    def get(self, category: str, *args, **kwargs) -> Optional[Any]:
        """
        从缓存获取数据
        
        Args:
            category: 缓存类别
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            缓存数据，如果未找到或已过期则返回None
        """
        if not self.enabled:
            self.stats['misses'] += 1
            return None
        
        cache_key = self._generate_cache_key(category, *args, **kwargs)
        
        if cache_key not in self.cache_store:
            self.stats['misses'] += 1
            return None
        
        cache_entry = self.cache_store[cache_key]
        
        # 检查是否过期
        if cache_entry['expires_at'] < time.time():
            # 已过期，删除并返回None
            del self.cache_store[cache_key]
            self.stats['evictions'] += 1
            self.stats['total_size'] -= 1
            self.stats['misses'] += 1
            return None
        
        # 命中缓存
        self.stats['hits'] += 1
        return cache_entry['data']
    
    def set(self, category: str, data: Any, ttl: Optional[int] = None, 
           *args, **kwargs) -> bool:
        """
        设置缓存数据
        
        Args:
            category: 缓存类别
            data: 要缓存的数据
            ttl: 生存时间（秒），如果为None则使用默认值
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            是否成功设置
        """
        if not self.enabled:
            return False
        
        # 检查缓存大小限制
        if len(self.cache_store) >= self.max_size:
            self._evict_oldest()
        
        cache_key = self._generate_cache_key(category, *args, **kwargs)
        
        # 计算过期时间
        if ttl is None:
            ttl = self.default_ttl
        
        expires_at = time.time() + ttl
        
        # 创建缓存条目
        cache_entry = {
            'data': data,
            'expires_at': expires_at,
            'created_at': time.time(),
            'category': category,
            'key': cache_key,
            'ttl': ttl
        }
        
        # 存储到缓存
        self.cache_store[cache_key] = cache_entry
        self.stats['sets'] += 1
        self.stats['total_size'] = len(self.cache_store)
        
        logger.debug(f"Cache set: {cache_key} (ttl: {ttl}s)")
        return True
    
    def delete(self, category: str, *args, **kwargs) -> bool:
        """
        删除缓存数据
        
        Args:
            category: 缓存类别
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            是否成功删除
        """
        if not self.enabled:
            return False
        
        cache_key = self._generate_cache_key(category, *args, **kwargs)
        
        if cache_key in self.cache_store:
            del self.cache_store[cache_key]
            self.stats['total_size'] = len(self.cache_store)
            logger.debug(f"Cache deleted: {cache_key}")
            return True
        
        return False
    
    def clear_category(self, category: str) -> int:
        """
        清除指定类别的所有缓存
        
        Args:
            category: 要清除的缓存类别
            
        Returns:
            清除的缓存项数量
        """
        if not self.enabled:
            return 0
        
        keys_to_delete = []
        prefix = f"{self.key_prefix}{category}:"
        
        for key in list(self.cache_store.keys()):
            if key.startswith(prefix):
                keys_to_delete.append(key)
        
        # 删除所有匹配的键
        for key in keys_to_delete:
            del self.cache_store[key]
        
        deleted_count = len(keys_to_delete)
        self.stats['total_size'] = len(self.cache_store)
        
        if deleted_count > 0:
            logger.info(f"Cleared {deleted_count} cache entries for category: {category}")
        
        return deleted_count
    
    def clear_all(self) -> int:
        """
        清除所有缓存
        
        Returns:
            清除的缓存项数量
        """
        if not self.enabled:
            return 0
        
        deleted_count = len(self.cache_store)
        self.cache_store.clear()
        self.stats['total_size'] = 0
        
        logger.info(f"Cleared all cache entries: {deleted_count} items")
        return deleted_count
    
    def _evict_oldest(self, count: int = 1):
        """
        淘汰最旧的缓存项
        
        Args:
            count: 要淘汰的项数
        """
        if not self.cache_store:
            return
        
        # 按创建时间排序
        sorted_entries = sorted(
            self.cache_store.items(),
            key=lambda x: x[1]['created_at']
        )
        
        # 淘汰最旧的项
        for i in range(min(count, len(sorted_entries))):
            key, entry = sorted_entries[i]
            del self.cache_store[key]
            self.stats['evictions'] += 1
        
        self.stats['total_size'] = len(self.cache_store)
        logger.debug(f"Evicted {count} oldest cache entries")
    
    def cleanup_expired(self) -> int:
        """
        清理过期的缓存项
        
        Returns:
            清理的缓存项数量
        """
        if not self.enabled:
            return 0
        
        current_time = time.time()
        keys_to_delete = []
        
        for key, entry in self.cache_store.items():
            if entry['expires_at'] < current_time:
                keys_to_delete.append(key)
        
        # 删除所有过期的键
        for key in keys_to_delete:
            del self.cache_store[key]
        
        cleaned_count = len(keys_to_delete)
        self.stats['evictions'] += cleaned_count
        self.stats['total_size'] = len(self.cache_store)
        
        if cleaned_count > 0:
            logger.debug(f"Cleaned up {cleaned_count} expired cache entries")
        
        return cleaned_count
    
    # 特定类别的缓存方法
    
    def get_session_mapping(self, openclaw_session_key: str) -> Optional[Dict[str, Any]]:
        """获取会话映射缓存"""
        return self.get('session_mapping', openclaw_session_key)
    
    def set_session_mapping(self, openclaw_session_key: str, mapping: Dict[str, Any]) -> bool:
        """设置会话映射缓存"""
        return self.set('session_mapping', mapping, ttl=86400,  # 24小时
                       openclaw_session_key=openclaw_session_key)
    
    def delete_session_mapping(self, openclaw_session_key: str) -> bool:
        """删除会话映射缓存"""
        return self.delete('session_mapping', openclaw_session_key=openclaw_session_key)
    
    def get_context_conversion(self, message_type: str, message_hash: str) -> Optional[Dict[str, Any]]:
        """获取上下文转换缓存"""
        return self.get('context_conversion', message_type, message_hash)
    
    def set_context_conversion(self, message_type: str, message_hash: str, 
                              conversion: Dict[str, Any]) -> bool:
        """设置上下文转换缓存"""
        return self.set('context_conversion', conversion, ttl=3600,  # 1小时
                       message_type=message_type, message_hash=message_hash)
    
    def get_query_result(self, query_type: str, query_hash: str) -> Optional[Dict[str, Any]]:
        """获取查询结果缓存"""
        return self.get('query_result', query_type, query_hash)
    
    def set_query_result(self, query_type: str, query_hash: str, 
                        result: Dict[str, Any]) -> bool:
        """设置查询结果缓存"""
        return self.set('query_result', result, ttl=1800,  # 30分钟
                       query_type=query_type, query_hash=query_hash)
    
    def clear_session_cache(self, session_key: str) -> int:
        """清除与会话相关的所有缓存"""
        # 清除会话映射缓存
        self.delete_session_mapping(session_key)
        
        # 这里可以添加清除其他与会话相关的缓存
        # 例如：会话上下文缓存、会话查询缓存等
        
        return 1  # 返回清除的缓存项数量
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        hit_rate = 0
        if self.stats['hits'] + self.stats['misses'] > 0:
            hit_rate = self.stats['hits'] / (self.stats['hits'] + self.stats['misses'])
        
        return {
            'enabled': self.enabled,
            'stats': self.stats.copy(),
            'hit_rate': hit_rate,
            'current_size': len(self.cache_store),
            'max_size': self.max_size,
            'default_ttl': self.default_ttl,
            'categories': self._get_cache_categories(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_cache_categories(self) -> List[str]:
        """获取所有缓存类别"""
        categories = set()
        for entry in self.cache_store.values():
            categories.add(entry['category'])
        return list(categories)
    
    def enable(self):
        """启用缓存"""
        self.enabled = True
        logger.info("Cache enabled")
    
    def disable(self):
        """禁用缓存"""
        self.enabled = False
        logger.info("Cache disabled")
    
    def is_enabled(self) -> bool:
        """检查缓存是否启用"""
        return self.enabled


class RedisCacheAdapter(CacheAdapter):
    """Redis缓存适配器（扩展版本）"""
    
    def __init__(self, redis_client, config: Optional[Dict[str, Any]] = None):
        """
        初始化Redis缓存适配器
        
        Args:
            redis_client: Redis客户端实例
            config: 配置字典
        """
        super().__init__(config)
        self.redis = redis_client
        self.serializer = json  # 使用JSON序列化
    
    def get(self, category: str, *args, **kwargs) -> Optional[Any]:
        """从Redis获取缓存数据"""
        if not self.enabled:
            self.stats['misses'] += 1
            return None
        
        cache_key = self._generate_cache_key(category, *args, **kwargs)
        
        try:
            cached_data = self.redis.get(cache_key)
            if cached_data is None:
                self.stats['misses'] += 1
                return None
            
            # 反序列化数据
            data = self.serializer.loads(cached_data)
            self.stats['hits'] += 1
            return data
            
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.stats['misses'] += 1
            return None
    
    def set(self, category: str, data: Any, ttl: Optional[int] = None, 
           *args, **kwargs) -> bool:
        """设置Redis缓存数据"""
        if not self.enabled:
            return False
        
        if ttl is None:
            ttl = self.default_ttl
        
        cache_key = self._generate_cache_key(category, *args, **kwargs)
        
        try:
            # 序列化数据
            serialized_data = self.serializer.dumps(data)
            
            # 存储到Redis
            if ttl > 0:
                self.redis.setex(cache_key, ttl, serialized_data)
            else:
                self.redis.set(cache_key, serialized_data)
            
            self.stats['sets'] += 1
            logger.debug(f"Redis cache set: {cache_key} (ttl: {ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def delete(self, category: str, *args, **kwargs) -> bool:
        """删除Redis缓存数据"""
        if not self.enabled:
            return False
        
        cache_key = self._generate_cache_key(category, *args, **kwargs)
        
        try:
            result = self.redis.delete(cache_key)
            deleted = result > 0
            
            if deleted:
                logger.debug(f"Redis cache deleted: {cache_key}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def clear_all(self) -> int:
        """清除所有Redis缓存（谨慎使用）"""
        if not self.enabled:
            return 0
        
        try:
            # 只清除兼容层相关的键
            keys = self.redis.keys(f"{self.key_prefix}*")
            if keys:
                deleted = self.redis.delete(*keys)
                logger.info(f"Cleared Redis cache: {deleted} items")
                return deleted
            return 0
            
        except Exception as e:
            logger.error(f"Redis clear_all error: {e}")
            return 0