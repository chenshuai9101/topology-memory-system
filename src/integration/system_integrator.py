"""
系统集成器 - 集成所有组件并提供统一接口
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from .config_manager import ConfigManager, get_config
from .service_container import ServiceContainer, get_service_container


class IntegrationStatus(Enum):
    """集成状态"""
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    CONFIG_LOADED = "config_loaded"
    SERVICES_REGISTERED = "services_registered"
    DATABASE_CONNECTED = "database_connected"
    REDIS_CONNECTED = "redis_connected"
    VECTOR_SEARCH_CONNECTED = "vector_search_connected"
    ML_SERVICES_READY = "ml_services_ready"
    API_SERVER_STARTED = "api_server_started"
    READY = "ready"
    ERROR = "error"


@dataclass
class IntegrationMetrics:
    """集成指标"""
    startup_time: float = 0.0
    config_load_time: float = 0.0
    service_registration_time: float = 0.0
    database_connection_time: float = 0.0
    redis_connection_time: float = 0.0
    vector_search_connection_time: float = 0.0
    ml_services_init_time: float = 0.0
    total_services: int = 0
    healthy_services: int = 0
    last_health_check: float = 0.0


class SystemIntegrator:
    """系统集成器"""
    
    def __init__(self, env: str = "development"):
        """
        初始化系统集成器
        
        Args:
            env: 环境类型
        """
        self.env = env
        self.status = IntegrationStatus.NOT_STARTED
        self.metrics = IntegrationMetrics()
        self.start_time = time.time()
        
        # 初始化日志
        self.logger = logging.getLogger(__name__)
        
        # 核心组件
        self.config: Optional[ConfigManager] = None
        self.container: Optional[ServiceContainer] = None
        
        # 服务实例缓存
        self._services: Dict[str, Any] = {}
        
        # 错误信息
        self.errors: List[str] = []
    
    def initialize(self) -> bool:
        """
        初始化系统
        
        Returns:
            是否初始化成功
        """
        try:
            self.logger.info(f"Starting system integration for environment: {self.env}")
            
            # 1. 加载配置
            self.status = IntegrationStatus.INITIALIZING
            config_start = time.time()
            
            self.config = get_config(self.env)
            if not self.config.validate():
                raise ValueError("Configuration validation failed")
            
            self.metrics.config_load_time = time.time() - config_start
            self.status = IntegrationStatus.CONFIG_LOADED
            self.logger.info(f"Configuration loaded in {self.metrics.config_load_time:.2f}s")
            
            # 2. 初始化服务容器
            service_start = time.time()
            self.container = get_service_container(self.config)
            self.metrics.service_registration_time = time.time() - service_start
            self.status = IntegrationStatus.SERVICES_REGISTERED
            self.logger.info(f"Services registered in {self.metrics.service_registration_time:.2f}s")
            
            # 3. 连接数据库
            if self._connect_database():
                self.status = IntegrationStatus.DATABASE_CONNECTED
                self.logger.info("Database connected successfully")
            
            # 4. 连接Redis
            if self._connect_redis():
                self.status = IntegrationStatus.REDIS_CONNECTED
                self.logger.info("Redis connected successfully")
            
            # 5. 连接向量搜索
            if self._connect_vector_search():
                self.status = IntegrationStatus.VECTOR_SEARCH_CONNECTED
                self.logger.info("Vector search connected successfully")
            
            # 6. 初始化ML服务
            if self._initialize_ml_services():
                self.status = IntegrationStatus.ML_SERVICES_READY
                self.logger.info("ML services initialized successfully")
            
            # 7. 运行健康检查
            self._run_health_check()
            
            # 计算总启动时间
            self.metrics.startup_time = time.time() - self.start_time
            
            if self.errors:
                self.status = IntegrationStatus.ERROR
                self.logger.error(f"System integration completed with {len(self.errors)} errors")
                return False
            else:
                self.status = IntegrationStatus.READY
                self.logger.info(f"System integration completed successfully in {self.metrics.startup_time:.2f}s")
                return True
                
        except Exception as e:
            self.status = IntegrationStatus.ERROR
            self.errors.append(f"Initialization failed: {str(e)}")
            self.logger.error(f"System integration failed: {e}", exc_info=True)
            return False
    
    def _connect_database(self) -> bool:
        """连接数据库"""
        try:
            start_time = time.time()
            
            # 获取数据库引擎
            from sqlalchemy import create_engine, text
            
            engine = create_engine(
                self.config.database.connection_string,
                pool_size=self.config.database.pool_size,
                max_overflow=self.config.database.max_overflow,
                echo=self.config.database.echo
            )
            
            # 测试连接
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # 存储引擎
            self._services["database_engine"] = engine
            
            self.metrics.database_connection_time = time.time() - start_time
            self.logger.debug(f"Database connection established in {self.metrics.database_connection_time:.2f}s")
            return True
            
        except Exception as e:
            self.errors.append(f"Database connection failed: {str(e)}")
            self.logger.error(f"Failed to connect to database: {e}")
            return False
    
    def _connect_redis(self) -> bool:
        """连接Redis"""
        try:
            start_time = time.time()
            
            import redis
            
            redis_client = redis.Redis(
                host=self.config.redis.host,
                port=self.config.redis.port,
                password=self.config.redis.password,
                db=self.config.redis.db,
                decode_responses=self.config.redis.decode_responses
            )
            
            # 测试连接
            redis_client.ping()
            
            # 存储客户端
            self._services["redis_client"] = redis_client
            
            self.metrics.redis_connection_time = time.time() - start_time
            self.logger.debug(f"Redis connection established in {self.metrics.redis_connection_time:.2f}s")
            return True
            
        except ImportError:
            self.logger.warning("Redis not available, skipping Redis connection")
            return False
        except Exception as e:
            self.errors.append(f"Redis connection failed: {str(e)}")
            self.logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    def _connect_vector_search(self) -> bool:
        """连接向量搜索"""
        try:
            start_time = time.time()
            
            if self.config.vector_search.provider == "qdrant":
                from qdrant_client import QdrantClient
                
                qdrant_client = QdrantClient(
                    host=self.config.vector_search.host,
                    port=self.config.vector_search.port
                )
                
                # 测试连接
                collections = qdrant_client.get_collections()
                
                # 确保集合存在
                collection_names = [c.name for c in collections.collections]
                if self.config.vector_search.collection_name not in collection_names:
                    self.logger.info(f"Creating collection: {self.config.vector_search.collection_name}")
                    qdrant_client.create_collection(
                        collection_name=self.config.vector_search.collection_name,
                        vectors_config={
                            "size": self.config.vector_search.embedding_dim,
                            "distance": self.config.vector_search.distance_metric
                        }
                    )
                
                # 存储客户端
                self._services["vector_search_client"] = qdrant_client
                
                self.metrics.vector_search_connection_time = time.time() - start_time
                self.logger.debug(f"Vector search connection established in {self.metrics.vector_search_connection_time:.2f}s")
                return True
            else:
                self.logger.warning(f"Vector search provider {self.config.vector_search.provider} not implemented")
                return False
                
        except ImportError:
            self.logger.warning("Qdrant client not available, skipping vector search connection")
            return False
        except Exception as e:
            self.errors.append(f"Vector search connection failed: {str(e)}")
            self.logger.error(f"Failed to connect to vector search: {e}")
            return False
    
    def _initialize_ml_services(self) -> bool:
        """初始化ML服务"""
        try:
            start_time = time.time()
            
            from sentence_transformers import SentenceTransformer
            
            # 加载嵌入模型
            embedding_model = SentenceTransformer(
                self.config.ml.embedding_model,
                device=self.config.ml.device
            )
            
            # 测试模型
            test_embedding = embedding_model.encode(["test sentence"])
            if len(test_embedding) == 0:
                raise ValueError("Failed to generate embedding")
            
            # 存储模型
            self._services["embedding_model"] = embedding_model
            
            self.metrics.ml_services_init_time = time.time() - start_time
            self.logger.debug(f"ML services initialized in {self.metrics.ml_services_init_time:.2f}s")
            return True
            
        except ImportError:
            self.logger.warning("SentenceTransformers not available, skipping ML services")
            return False
        except Exception as e:
            self.errors.append(f"ML services initialization failed: {str(e)}")
            self.logger.error(f"Failed to initialize ML services: {e}")
            return False
    
    def _run_health_check(self) -> None:
        """运行健康检查"""
        try:
            health_status = self.container.health_check() if self.container else {}
            
            self.metrics.total_services = len(health_status)
            self.metrics.healthy_services = sum(1 for healthy in health_status.values() if healthy)
            self.metrics.last_health_check = time.time()
            
            self.logger.info(f"Health check: {self.metrics.healthy_services}/{self.metrics.total_services} services healthy")
            
            for service, healthy in health_status.items():
                if not healthy:
                    self.logger.warning(f"Service {service} is not healthy")
                    
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
    
    def get_service(self, service_type: type) -> Any:
        """
        获取服务实例
        
        Args:
            service_type: 服务类型
            
        Returns:
            服务实例
        """
        if not self.container:
            raise RuntimeError("System not initialized")
        
        return self.container.get(service_type)
    
    def get_config(self) -> ConfigManager:
        """获取配置管理器"""
        if not self.config:
            raise RuntimeError("System not initialized")
        
        return self.config
    
    def get_container(self) -> ServiceContainer:
        """获取服务容器"""
        if not self.container:
            raise RuntimeError("System not initialized")
        
        return self.container
    
    def start_api_server(self) -> bool:
        """
        启动API服务器
        
        Returns:
            是否启动成功
        """
        try:
            self.logger.info("Starting API server...")
            
            # 导入API模块
            from ..api.main import app
            
            import uvicorn
            
            # 在后台启动服务器
            import threading
            
            def run_server():
                uvicorn.run(
                    app,
                    host=self.config.api.host,
                    port=self.config.api.port,
                    workers=self.config.api.workers,
                    reload=self.config.api.reload
                )
            
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            # 等待服务器启动
            import time
            time.sleep(2)
            
            self.status = IntegrationStatus.API_SERVER_STARTED
            self.logger.info(f"API server started on {self.config.api.server_url}")
            return True
            
        except Exception as e:
            self.errors.append(f"API server startup failed: {str(e)}")
            self.logger.error(f"Failed to start API server: {e}")
            return False
    
    def shutdown(self) -> None:
        """关闭系统"""
        self.logger.info("Shutting down system...")
        
        # 关闭数据库连接
        if "database_engine" in self._services:
            self._services["database_engine"].dispose()
            self.logger.debug("Database connections closed")
        
        # 关闭Redis连接
        if "redis_client" in self._services:
            self._services["redis_client"].close()
            self.logger.debug("Redis connection closed")
        
        self.status = IntegrationStatus.NOT_STARTED
        self.logger.info("System shutdown completed")
    
    def get_status_report(self) -> Dict[str, Any]:
        """获取状态报告"""
        return {
            "environment": self.env,
            "status": self.status.value,
            "startup_time": self.metrics.startup_time,
            "metrics": {
                "config_load_time": self.metrics.config_load_time,
                "service_registration_time": self.metrics.service_registration_time,
                "database_connection_time": self.metrics.database_connection_time,
                "redis_connection_time": self.metrics.redis_connection_time,
                "vector_search_connection_time": self.metrics.vector_search_connection_time,
                "ml_services_init_time": self.metrics.ml_services_init_time
            },
            "health": {
                "total_services": self.metrics.total_services,
                "healthy_services": self.metrics.healthy_services,
                "health_check_age": time.time() - self.metrics.last_health_check
            },
            "errors": self.errors,
            "services_available": list(self._services.keys()) if self._services else []
        }
    
    def __enter__(self):
        """上下文管理器入口"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.shutdown()
    
    def __str__(self) -> str:
        """字符串表示"""
        report = self.get_status_report()
        
        lines = [
            f"System Integrator - {self.env.upper()}",
            f"Status: {report['status']}",
            f"Startup Time: {report['startup_time']:.2f}s",
            f"Healthy Services: {report['health']['healthy_services']}/{report['health']['total_services']}",
        ]
        
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            for error in self.errors[:3]:  # 只显示前3个错误
                lines.append(f"  - {error}")
        
        return "\n".join(lines)


# 全局系统集成器实例
_system_integrator: Optional[SystemIntegrator] = None


def get_system_integrator(env: str = "development") -> SystemIntegrator:
    """获取全局系统集成器实例"""
    global _system_integrator
    if _system_integrator is None:
        _system_integrator = SystemIntegrator(env)
        _system_integrator.initialize()
    return _system_integrator


def initialize_system(env: str = "development") -> bool:
    """初始化系统（便捷函数）"""
    integrator = get_system_integrator(env)
    return integrator.status == IntegrationStatus.READY