"""
数据库配置模块
配置PostgreSQL和Redis连接
"""

import os
from typing import Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DatabaseConfig:
    """数据库配置类"""
    
    # PostgreSQL配置
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "topology_memory")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    
    # Redis配置
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    
    # 连接池配置
    POSTGRES_POOL_SIZE: int = int(os.getenv("POSTGRES_POOL_SIZE", "20"))
    POSTGRES_MAX_OVERFLOW: int = int(os.getenv("POSTGRES_MAX_OVERFLOW", "40"))
    POSTGRES_POOL_TIMEOUT: int = int(os.getenv("POSTGRES_POOL_TIMEOUT", "30"))
    POSTGRES_POOL_RECYCLE: int = int(os.getenv("POSTGRES_POOL_RECYCLE", "3600"))
    
    # Redis连接池配置
    REDIS_MAX_CONNECTIONS: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
    REDIS_SOCKET_TIMEOUT: int = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
    REDIS_SOCKET_CONNECT_TIMEOUT: int = int(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5"))
    
    # 性能配置
    QUERY_TIMEOUT: int = int(os.getenv("QUERY_TIMEOUT", "30"))
    CONNECTION_TIMEOUT: int = int(os.getenv("CONNECTION_TIMEOUT", "10"))
    
    # 缓存配置
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "300"))  # 5分钟
    CACHE_MAX_SIZE: int = int(os.getenv("CACHE_MAX_SIZE", "10000"))
    
    @property
    def postgres_url(self) -> str:
        """获取PostgreSQL连接URL"""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def redis_url(self) -> str:
        """获取Redis连接URL"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    @property
    def postgres_engine_options(self) -> Dict[str, Any]:
        """获取SQLAlchemy引擎配置"""
        return {
            "pool_size": self.POSTGRES_POOL_SIZE,
            "max_overflow": self.POSTGRES_MAX_OVERFLOW,
            "pool_timeout": self.POSTGRES_POOL_TIMEOUT,
            "pool_recycle": self.POSTGRES_POOL_RECYCLE,
            "echo": False,
            "future": True,
        }
    
    @property
    def redis_connection_pool(self) -> Dict[str, Any]:
        """获取Redis连接池配置"""
        return {
            "host": self.REDIS_HOST,
            "port": self.REDIS_PORT,
            "db": self.REDIS_DB,
            "password": self.REDIS_PASSWORD if self.REDIS_PASSWORD else None,
            "max_connections": self.REDIS_MAX_CONNECTIONS,
            "socket_timeout": self.REDIS_SOCKET_TIMEOUT,
            "socket_connect_timeout": self.REDIS_SOCKET_CONNECT_TIMEOUT,
            "decode_responses": True,
        }


# 全局配置实例
db_config = DatabaseConfig()