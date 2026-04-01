"""
数据库连接管理器
管理PostgreSQL和Redis连接池
"""

import logging
from typing import Optional
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from redis import Redis, ConnectionPool

from .database_config import db_config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQLAlchemy基础类
Base = declarative_base()


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self._engine = None
        self._session_factory = None
        self._redis_pool = None
        self._redis_client = None
        
    def init_postgres(self) -> None:
        """初始化PostgreSQL连接"""
        try:
            # 创建引擎
            self._engine = create_engine(
                db_config.postgres_url,
                **db_config.postgres_engine_options
            )
            
            # 创建会话工厂
            self._session_factory = scoped_session(
                sessionmaker(
                    autocommit=False,
                    autoflush=False,
                    bind=self._engine,
                    expire_on_commit=False
                )
            )
            
            logger.info("PostgreSQL连接初始化成功")
            
        except Exception as e:
            logger.error(f"PostgreSQL连接初始化失败: {e}")
            raise
    
    def init_redis(self) -> None:
        """初始化Redis连接"""
        try:
            # 创建连接池
            self._redis_pool = ConnectionPool(**db_config.redis_connection_pool)
            
            # 创建Redis客户端
            self._redis_client = Redis(
                connection_pool=self._redis_pool,
                decode_responses=True
            )
            
            # 测试连接
            self._redis_client.ping()
            
            logger.info("Redis连接初始化成功")
            
        except Exception as e:
            logger.error(f"Redis连接初始化失败: {e}")
            raise
    
    def init_all(self) -> None:
        """初始化所有数据库连接"""
        self.init_postgres()
        self.init_redis()
    
    @property
    def engine(self):
        """获取SQLAlchemy引擎"""
        if self._engine is None:
            self.init_postgres()
        return self._engine
    
    @property
    def session_factory(self):
        """获取会话工厂"""
        if self._session_factory is None:
            self.init_postgres()
        return self._session_factory
    
    @property
    def redis_client(self) -> Redis:
        """获取Redis客户端"""
        if self._redis_client is None:
            self.init_redis()
        return self._redis_client
    
    @contextmanager
    def get_session(self) -> Session:
        """获取数据库会话上下文管理器"""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"数据库会话异常: {e}")
            raise
        finally:
            session.close()
    
    def create_tables(self) -> None:
        """创建所有数据库表"""
        try:
            # 导入所有模型
            from ..models import contexts, memory_nodes, associations
            
            # 创建表
            Base.metadata.create_all(bind=self.engine)
            logger.info("数据库表创建成功")
            
        except Exception as e:
            logger.error(f"数据库表创建失败: {e}")
            raise
    
    def drop_tables(self) -> None:
        """删除所有数据库表"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("数据库表删除成功")
            
        except Exception as e:
            logger.error(f"数据库表删除失败: {e}")
            raise
    
    def health_check(self) -> Dict[str, bool]:
        """健康检查"""
        health_status = {
            "postgres": False,
            "redis": False
        }
        
        try:
            # 检查PostgreSQL
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            health_status["postgres"] = True
        except Exception as e:
            logger.error(f"PostgreSQL健康检查失败: {e}")
        
        try:
            # 检查Redis
            self.redis_client.ping()
            health_status["redis"] = True
        except Exception as e:
            logger.error(f"Redis健康检查失败: {e}")
        
        return health_status
    
    def close(self) -> None:
        """关闭所有数据库连接"""
        try:
            # 关闭PostgreSQL连接
            if self._engine:
                self._engine.dispose()
                logger.info("PostgreSQL连接已关闭")
            
            # 关闭Redis连接
            if self._redis_pool:
                self._redis_pool.disconnect()
                logger.info("Redis连接已关闭")
                
        except Exception as e:
            logger.error(f"关闭数据库连接时出错: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        stats = {}
        
        try:
            # PostgreSQL连接统计
            if self._engine:
                pool = self._engine.pool
                stats["postgres"] = {
                    "checked_out": pool.checkedout(),
                    "checked_in": pool.checkedin(),
                    "overflow": pool.overflow(),
                    "size": pool.size(),
                    "max_overflow": pool._max_overflow,
                }
        except Exception as e:
            logger.error(f"获取PostgreSQL连接统计失败: {e}")
        
        try:
            # Redis连接统计
            if self._redis_pool:
                stats["redis"] = {
                    "connection_count": self._redis_pool._in_use_connections + self._redis_pool._available_connections,
                    "in_use_connections": self._redis_pool._in_use_connections,
                    "available_connections": self._redis_pool._available_connections,
                }
        except Exception as e:
            logger.error(f"获取Redis连接统计失败: {e}")
        
        return stats


# 全局数据库管理器实例
db_manager = DatabaseManager()