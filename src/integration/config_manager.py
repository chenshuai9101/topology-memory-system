"""
统一配置管理系统
支持环境变量、配置文件、默认值的统一管理
"""

import os
import yaml
import json
from typing import Any, Dict, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum


class Environment(Enum):
    """环境类型枚举"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """数据库配置"""
    host: str = "localhost"
    port: int = 5432
    database: str = "topology_memory"
    username: str = "postgres"
    password: str = "postgres"
    pool_size: int = 20
    max_overflow: int = 30
    echo: bool = False
    
    @property
    def connection_string(self) -> str:
        """获取数据库连接字符串"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class RedisConfig:
    """Redis配置"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    decode_responses: bool = True
    
    @property
    def connection_url(self) -> str:
        """获取Redis连接URL"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


@dataclass
class VectorSearchConfig:
    """向量搜索配置"""
    provider: str = "qdrant"  # qdrant, weaviate, pinecone
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "memory_embeddings"
    embedding_dim: int = 384
    distance_metric: str = "Cosine"
    
    @property
    def qdrant_url(self) -> str:
        """获取Qdrant URL"""
        return f"http://{self.host}:{self.port}"


@dataclass
class MLConfig:
    """机器学习配置"""
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    similarity_threshold: float = 0.7
    max_context_length: int = 512
    batch_size: int = 32
    device: str = "cpu"  # cpu, cuda, mps


@dataclass
class APIConfig:
    """API配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = True
    cors_origins: list = field(default_factory=lambda: ["*"])
    api_prefix: str = "/api/v1"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    
    @property
    def server_url(self) -> str:
        """获取服务器URL"""
        return f"http://{self.host}:{self.port}"


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class MonitoringConfig:
    """监控配置"""
    enabled: bool = True
    prometheus_port: int = 9090
    metrics_path: str = "/metrics"
    health_check_path: str = "/health"
    readiness_path: str = "/ready"
    liveness_path: str = "/live"


class ConfigManager:
    """统一配置管理器"""
    
    def __init__(self, env: Union[str, Environment] = Environment.DEVELOPMENT):
        """
        初始化配置管理器
        
        Args:
            env: 环境类型
        """
        if isinstance(env, str):
            env = Environment(env.lower())
        self.env = env
        
        # 配置文件路径
        self.config_dir = Path(__file__).parent.parent.parent / "config"
        self.config_dir.mkdir(exist_ok=True)
        
        # 加载配置
        self._config = self._load_config()
        
        # 初始化各组件配置
        self.database = self._init_database_config()
        self.redis = self._init_redis_config()
        self.vector_search = self._init_vector_search_config()
        self.ml = self._init_ml_config()
        self.api = self._init_api_config()
        self.logging = self._init_logging_config()
        self.monitoring = self._init_monitoring_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        config = {}
        
        # 1. 加载默认配置
        default_config_path = self.config_dir / "default.yaml"
        if default_config_path.exists():
            with open(default_config_path, 'r') as f:
                config.update(yaml.safe_load(f) or {})
        
        # 2. 加载环境特定配置
        env_config_path = self.config_dir / f"{self.env.value}.yaml"
        if env_config_path.exists():
            with open(env_config_path, 'r') as f:
                env_config = yaml.safe_load(f) or {}
                config.update(env_config)
        
        # 3. 加载环境变量（优先级最高）
        config.update(self._load_env_vars())
        
        return config
    
    def _load_env_vars(self) -> Dict[str, Any]:
        """从环境变量加载配置"""
        env_config = {}
        
        # 数据库配置
        if db_host := os.getenv("DB_HOST"):
            env_config["database"] = env_config.get("database", {})
            env_config["database"]["host"] = db_host
        
        if db_port := os.getenv("DB_PORT"):
            env_config["database"] = env_config.get("database", {})
            env_config["database"]["port"] = int(db_port)
        
        if db_name := os.getenv("DB_NAME"):
            env_config["database"] = env_config.get("database", {})
            env_config["database"]["database"] = db_name
        
        # API配置
        if api_host := os.getenv("API_HOST"):
            env_config["api"] = env_config.get("api", {})
            env_config["api"]["host"] = api_host
        
        if api_port := os.getenv("API_PORT"):
            env_config["api"] = env_config.get("api", {})
            env_config["api"]["port"] = int(api_port)
        
        # Redis配置
        if redis_host := os.getenv("REDIS_HOST"):
            env_config["redis"] = env_config.get("redis", {})
            env_config["redis"]["host"] = redis_host
        
        # 向量搜索配置
        if vs_host := os.getenv("VECTOR_SEARCH_HOST"):
            env_config["vector_search"] = env_config.get("vector_search", {})
            env_config["vector_search"]["host"] = vs_host
        
        return env_config
    
    def _init_database_config(self) -> DatabaseConfig:
        """初始化数据库配置"""
        db_config = self._config.get("database", {})
        return DatabaseConfig(
            host=db_config.get("host", "localhost"),
            port=db_config.get("port", 5432),
            database=db_config.get("database", "topology_memory"),
            username=db_config.get("username", "postgres"),
            password=db_config.get("password", "postgres"),
            pool_size=db_config.get("pool_size", 20),
            max_overflow=db_config.get("max_overflow", 30),
            echo=db_config.get("echo", False)
        )
    
    def _init_redis_config(self) -> RedisConfig:
        """初始化Redis配置"""
        redis_config = self._config.get("redis", {})
        return RedisConfig(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            password=redis_config.get("password"),
            db=redis_config.get("db", 0),
            decode_responses=redis_config.get("decode_responses", True)
        )
    
    def _init_vector_search_config(self) -> VectorSearchConfig:
        """初始化向量搜索配置"""
        vs_config = self._config.get("vector_search", {})
        return VectorSearchConfig(
            provider=vs_config.get("provider", "qdrant"),
            host=vs_config.get("host", "localhost"),
            port=vs_config.get("port", 6333),
            collection_name=vs_config.get("collection_name", "memory_embeddings"),
            embedding_dim=vs_config.get("embedding_dim", 384),
            distance_metric=vs_config.get("distance_metric", "Cosine")
        )
    
    def _init_ml_config(self) -> MLConfig:
        """初始化机器学习配置"""
        ml_config = self._config.get("ml", {})
        return MLConfig(
            embedding_model=ml_config.get("embedding_model", "all-MiniLM-L6-v2"),
            embedding_dim=ml_config.get("embedding_dim", 384),
            similarity_threshold=ml_config.get("similarity_threshold", 0.7),
            max_context_length=ml_config.get("max_context_length", 512),
            batch_size=ml_config.get("batch_size", 32),
            device=ml_config.get("device", "cpu")
        )
    
    def _init_api_config(self) -> APIConfig:
        """初始化API配置"""
        api_config = self._config.get("api", {})
        return APIConfig(
            host=api_config.get("host", "0.0.0.0"),
            port=api_config.get("port", 8000),
            workers=api_config.get("workers", 4),
            reload=api_config.get("reload", True),
            cors_origins=api_config.get("cors_origins", ["*"]),
            api_prefix=api_config.get("api_prefix", "/api/v1"),
            docs_url=api_config.get("docs_url", "/docs"),
            redoc_url=api_config.get("redoc_url", "/redoc")
        )
    
    def _init_logging_config(self) -> LoggingConfig:
        """初始化日志配置"""
        logging_config = self._config.get("logging", {})
        return LoggingConfig(
            level=logging_config.get("level", "INFO"),
            format=logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            file_path=logging_config.get("file_path"),
            max_bytes=logging_config.get("max_bytes", 10 * 1024 * 1024),
            backup_count=logging_config.get("backup_count", 5)
        )
    
    def _init_monitoring_config(self) -> MonitoringConfig:
        """初始化监控配置"""
        monitoring_config = self._config.get("monitoring", {})
        return MonitoringConfig(
            enabled=monitoring_config.get("enabled", True),
            prometheus_port=monitoring_config.get("prometheus_port", 9090),
            metrics_path=monitoring_config.get("metrics_path", "/metrics"),
            health_check_path=monitoring_config.get("health_check_path", "/health"),
            readiness_path=monitoring_config.get("readiness_path", "/ready"),
            liveness_path=monitoring_config.get("liveness_path", "/live")
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            "environment": self.env.value,
            "database": self.database.__dict__,
            "redis": {k: v for k, v in self.redis.__dict__.items() if v is not None},
            "vector_search": self.vector_search.__dict__,
            "ml": self.ml.__dict__,
            "api": self.api.__dict__,
            "logging": {k: v for k, v in self.logging.__dict__.items() if v is not None},
            "monitoring": self.monitoring.__dict__
        }
    
    def save(self, path: Optional[Path] = None) -> None:
        """保存配置到文件"""
        if path is None:
            path = self.config_dir / f"{self.env.value}.json"
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def validate(self) -> bool:
        """验证配置是否有效"""
        try:
            # 验证数据库配置
            if not self.database.host:
                raise ValueError("Database host is required")
            
            # 验证API配置
            if self.api.port < 1 or self.api.port > 65535:
                raise ValueError(f"Invalid API port: {self.api.port}")
            
            # 验证向量搜索配置
            if self.vector_search.embedding_dim <= 0:
                raise ValueError(f"Invalid embedding dimension: {self.vector_search.embedding_dim}")
            
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False
    
    def __str__(self) -> str:
        """字符串表示"""
        return json.dumps(self.to_dict(), indent=2, default=str)


# 全局配置实例
_config_manager: Optional[ConfigManager] = None


def get_config(env: Union[str, Environment] = Environment.DEVELOPMENT) -> ConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(env)
    return _config_manager