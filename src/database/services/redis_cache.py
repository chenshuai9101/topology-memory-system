"""
Redis缓存服务
"""

import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Optional, Any, Dict, List, Union
from uuid import UUID
from functools import wraps

from ..config.database_manager import db_manager

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis缓存服务"""
    
    def __init__(self):
        self.redis = db_manager.redis_client
        self.default_ttl = 300  # 5分钟
    
    # ========== 基础缓存操作 ==========
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        try:
            value = self.redis.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.error(f"获取缓存失败 key={key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """设置缓存值"""
        try:
            if ttl is None:
                ttl = self.default_ttl
            
            serialized = pickle.dumps(value)
            return self.redis.setex(key, ttl, serialized)
        except Exception as e:
            logger.error(f"设置缓存失败 key={key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            return bool(self.redis.delete(key))
        except Exception as e:
            logger.error(f"删除缓存失败 key={key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        try:
            return bool(self.redis.exists(key))
        except Exception as e:
            logger.error(f"检查缓存失败 key={key}: {e}")
            return False
    
    def expire(self, key: str, ttl: int) -> bool:
        """设置过期时间"""
        try:
            return bool(self.redis.expire(key, ttl))
        except Exception as e:
            logger.error(f"设置过期时间失败 key={key}: {e}")
            return False
    
    def ttl(self, key: str) -> int:
        """获取剩余生存时间"""
        try:
            return self.redis.ttl(key)
        except Exception as e:
            logger.error(f"获取TTL失败 key={key}: {e}")
            return -2  # key不存在
    
    # ========== 哈希表操作 ==========
    
    def hget(self, key: str, field: str) -> Optional[Any]:
        """获取哈希表字段值"""
        try:
            value = self.redis.hget(key, field)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.error(f"获取哈希表字段失败 key={key}, field={field}: {e}")
            return None
    
    def hset(self, key: str, field: str, value: Any) -> bool:
        """设置哈希表字段值"""
        try:
            serialized = pickle.dumps(value)
            return bool(self.redis.hset(key, field, serialized))
        except Exception as e:
            logger.error(f"设置哈希表字段失败 key={key}, field={field}: {e}")
            return False
    
    def hgetall(self, key: str) -> Dict[str, Any]:
        """获取哈希表所有字段"""
        try:
            data = self.redis.hgetall(key)
            return {k.decode(): pickle.loads(v) for k, v in data.items()}
        except Exception as e:
            logger.error(f"获取哈希表所有字段失败 key={key}: {e}")
            return {}
    
    def hdel(self, key: str, *fields: str) -> int:
        """删除哈希表字段"""
        try:
            return self.redis.hdel(key, *fields)
        except Exception as e:
            logger.error(f"删除哈希表字段失败 key={key}, fields={fields}: {e}")
            return 0
    
    def hexists(self, key: str, field: str) -> bool:
        """检查哈希表字段是否存在"""
        try:
            return bool(self.redis.hexists(key, field))
        except Exception as e:
            logger.error(f"检查哈希表字段失败 key={key}, field={field}: {e}")
            return False
    
    # ========== 集合操作 ==========
    
    def sadd(self, key: str, *values: Any) -> int:
        """添加集合元素"""
        try:
            serialized = [pickle.dumps(v) for v in values]
            return self.redis.sadd(key, *serialized)
        except Exception as e:
            logger.error(f"添加集合元素失败 key={key}: {e}")
            return 0
    
    def smembers(self, key: str) -> List[Any]:
        """获取集合所有元素"""
        try:
            members = self.redis.smembers(key)
            return [pickle.loads(m) for m in members]
        except Exception as e:
            logger.error(f"获取集合元素失败 key={key}: {e}")
            return []
    
    def srem(self, key: str, *values: Any) -> int:
        """删除集合元素"""
        try:
            serialized = [pickle.dumps(v) for v in values]
            return self.redis.srem(key, *serialized)
        except Exception as e:
            logger.error(f"删除集合元素失败 key={key}: {e}")
            return 0
    
    def sismember(self, key: str, value: Any) -> bool:
        """检查元素是否在集合中"""
        try:
            serialized = pickle.dumps(value)
            return bool(self.redis.sismember(key, serialized))
        except Exception as e:
            logger.error(f"检查集合元素失败 key={key}: {e}")
            return False
    
    # ========== 有序集合操作 ==========
    
    def zadd(self, key: str, mapping: Dict[Any, float]) -> int:
        """添加有序集合元素"""
        try:
            serialized_mapping = {pickle.dumps(k): v for k, v in mapping.items()}
            return self.redis.zadd(key, serialized_mapping)
        except Exception as e:
            logger.error(f"添加有序集合元素失败 key={key}: {e}")
            return 0
    
    def zrange(self, key: str, start: int, end: int, withscores: bool = False) -> List[Any]:
        """获取有序集合范围"""
        try:
            if withscores:
                result = self.redis.zrange(key, start, end, withscores=True)
                return [(pickle.loads(k), v) for k, v in result]
            else:
                result = self.redis.zrange(key, start, end)
                return [pickle.loads(k) for k in result]
        except Exception as e:
            logger.error(f"获取有序集合范围失败 key={key}: {e}")
            return []
    
    def zrevrange(self, key: str, start: int, end: int, withscores: bool = False) -> List[Any]:
        """获取有序集合范围（倒序）"""
        try:
            if withscores:
                result = self.redis.zrevrange(key, start, end, withscores=True)
                return [(pickle.loads(k), v) for k, v in result]
            else:
                result = self.redis.zrevrange(key, start, end)
                return [pickle.loads(k) for k in result]
        except Exception as e:
            logger.error(f"获取有序集合倒序范围失败 key={key}: {e}")
            return []
    
    def zrem(self, key: str, *values: Any) -> int:
        """删除有序集合元素"""
        try:
            serialized = [pickle.dumps(v) for v in values]
            return self.redis.zrem(key, *serialized)
        except Exception as e:
            logger.error(f"删除有序集合元素失败 key={key}: {e}")
            return 0
    
    # ========== 列表操作 ==========
    
    def lpush(self, key: str, *values: Any) -> int:
        """列表左侧插入"""
        try:
            serialized = [pickle.dumps(v) for v in values]
            return self.redis.lpush(key, *serialized)
        except Exception as e:
            logger.error(f"列表左侧插入失败 key={key}: {e}")
            return 0
    
    def rpush(self, key: str, *values: Any) -> int:
        """列表右侧插入"""
        try:
            serialized = [pickle.dumps(v) for v in values]
            return self.redis.rpush(key, *serialized)
        except Exception as e:
            logger.error(f"列表右侧插入失败 key={key}: {e}")
            return 0
    
    def lrange(self, key: str, start: int, end: int) -> List[Any]:
        """获取列表范围"""
        try:
            result = self.redis.lrange(key, start, end)
            return [pickle.loads(r) for r in result]
        except Exception as e:
            logger.error(f"获取列表范围失败 key={key}: {e}")
            return []
    
    def ltrim(self, key: str, start: int, end: int) -> bool:
        """修剪列表"""
        try:
            return bool(self.redis.ltrim(key, start, end))
        except Exception as e:
            logger.error(f"修剪列表失败 key={key}: {e}")
            return False
    
    # ========== 业务缓存 ==========
    
    def cache_context(self, context_id: UUID, context_data: Dict[str, Any], ttl: int = 300) -> bool:
        """缓存上下文数据"""
        key = f"context:{context_id}"
        return self.set(key, context_data, ttl)
    
    def get_cached_context(self, context_id: UUID) -> Optional[Dict[str, Any]]:
        """获取缓存的上下文数据"""
        key = f"context:{context_id}"
        return self.get(key)
    
    def cache_memory_node(self, node_id: UUID, node_data: Dict[str, Any], ttl: int = 300) -> bool:
        """缓存记忆节点数据"""
        key = f"memory_node:{node_id}"
        return self.set(key, node_data, ttl)
    
    def get_cached_memory_node(self, node_id: UUID) -> Optional[Dict[str, Any]]:
        """获取缓存的记忆节点数据"""
        key = f"memory_node:{node_id}"
        return self.get(key)
    
    def cache_query_result(self, query_hash: str, result: Any, ttl: int = 600) -> bool:
        """缓存查询结果"""
        key = f"query:{query_hash}"
        return self.set(key, result, ttl)
    
    def get_cached_query_result(self, query_hash: str) -> Optional[Any]:
        """获取缓存的查询结果"""
        key = f"query:{query_hash}"
        return self.get(key)
    
    def cache_session_contexts(self, session_id: str, contexts: List[Dict[str, Any]], ttl: int = 600) -> bool:
        """缓存会话上下文列表"""
        key = f"session_contexts:{session_id}"
        return self.set(key, contexts, ttl)
    
    def get_cached_session_contexts(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """获取缓存的会话上下文列表"""
        key = f"session_contexts:{session_id}"
        return self.get(key)
    
    def add_to_recently_accessed(self, node_id: UUID, max_items: int = 100) -> bool:
        """添加到最近访问列表"""
        key = "recently_accessed"
        
        # 使用有序集合，分数为时间戳
        score = datetime.utcnow().timestamp()
        mapping = {str(node_id): score}
        
        # 添加元素
        self.zadd(key, mapping)
        
        # 保持列表大小
        self.redis.zremrangebyrank(key, 0, -(max_items + 1))
        
        return True
    
    def get_recently_accessed(self, limit: int = 20) -> List[UUID]:
        """获取最近访问的节点"""
        key = "recently_accessed"
        
        # 获取分数最高的元素（最近访问的）
        items = self.zrevrange(key, 0, limit - 1)
        
        # 转换为UUID
        return [UUID(item) for item in items]
    
    def increment_access_counter(self, node_id: UUID) -> int:
        """增加访问计数器"""
        key = f"access_counter:{node_id}"
        
        try:
            return self.redis.incr(key)
        except Exception as e:
            logger.error(f"增加访问计数器失败 node_id={node_id}: {e}")
            return 0
    
    def get_access_counter(self, node_id: UUID) -> int:
        """获取访问计数器"""
        key = f"access_counter:{node_id}"
        
        try:
            value = self.redis.get(key)
            return int(value) if value else 0
        except Exception as e:
            logger.error(f"获取访问计数器失败 node_id={node_id}: {e}")
            return 0
    
    def clear_all_cache(self) -> bool:
        """清除所有缓存"""
        try:
            # 只清除业务缓存，保留系统缓存
            pattern = "context:*"
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
            
            pattern = "memory_node:*"
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
            
            pattern = "query:*"
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
            
            pattern = "session_contexts:*"
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
            
            logger.info("业务缓存已清除")
            return True
            
        except Exception as e:
            logger.error(f"清除缓存失败: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        try:
            # 统计各种键的数量
            patterns = [
                "context:*",
                "memory_node:*", 
                "query:*",
                "session_contexts:*",
                "recently_accessed",
                "access_counter:*"
            ]
            
            stats = {}
            total_keys = 0
            
            for pattern in patterns:
                keys = self.redis.keys(pattern)
                count = len(keys)
                stats[pattern] = count
                total_keys += count
            
            stats['total_keys'] = total_keys
            
            # 获取内存使用情况
            info = self.redis.info()
            stats['used_memory'] = info.get('used_memory_human', '0')
            stats['connected_clients'] = info.get('connected_clients', 0)
            stats['total_commands_processed'] = info.get('total_commands_processed', 0)
            
            return stats
            
        except Exception as e:
            logger.error(f"获取缓存统计失败: {e}")
            return {}


# 缓存装饰器
def cached(ttl: int = 300, key_prefix: str = ""):
    """缓存装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{key_prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # 尝试从缓存获取
            cache = RedisCache()
            cached_result = cache.get(cache_key)
            
            if cached_result is not None:
                logger.debug(f"缓存命中: {cache_key}")
                return cached_result
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 缓存结果
            if result is not None:
                cache.set(cache_key, result, ttl)
                logger.debug(f"缓存设置: {cache_key}")
            
            return result
        return wrapper
    return decorator


# 全局缓存实例
redis_cache = RedisCache()