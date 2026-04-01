"""
服务容器 - 依赖注入和管理
提供统一的依赖管理和服务生命周期管理
"""

import logging
from typing import Any, Dict, Type, Optional, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum

from .config_manager import ConfigManager, get_config


class ServiceLifecycle(Enum):
    """服务生命周期"""
    TRANSIENT = "transient"  # 每次请求都创建新实例
    SINGLETON = "singleton"  # 单例，全局共享
    SCOPED = "scoped"       # 作用域内单例（如请求作用域）


@dataclass
class ServiceDescriptor:
    """服务描述符"""
    service_type: Type
    implementation: Optional[Type] = None
    factory: Optional[Callable] = None
    lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON
    instance: Optional[Any] = None
    dependencies: Dict[str, Type] = field(default_factory=dict)


class ServiceContainer:
    """服务容器"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        初始化服务容器
        
        Args:
            config: 配置管理器
        """
        self.config = config or get_config()
        self._services: Dict[str, ServiceDescriptor] = {}
        self._scoped_instances: Dict[str, Any] = {}
        self._logger = logging.getLogger(__name__)
        
        # 注册核心服务
        self._register_core_services()
    
    def _register_core_services(self) -> None:
        """注册核心服务"""
        # 注册配置服务
        self.register_singleton(ConfigManager, instance=self.config)
        
        # 注册日志服务
        self.register_transient(logging.Logger, factory=self._create_logger)
    
    def _create_logger(self, name: str = __name__) -> logging.Logger:
        """创建日志器"""
        logger = logging.getLogger(name)
        
        # 配置日志级别
        logger.setLevel(getattr(logging, self.config.logging.level))
        
        # 如果已经有处理器，直接返回
        if logger.handlers:
            return logger
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config.logging.level))
        
        # 创建格式化器
        formatter = logging.Formatter(self.config.logging.format)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        # 如果需要文件日志
        if self.config.logging.file_path:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.config.logging.file_path,
                maxBytes=self.config.logging.max_bytes,
                backupCount=self.config.logging.backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def register(
        self,
        service_type: Type,
        implementation: Optional[Type] = None,
        factory: Optional[Callable] = None,
        lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON
    ) -> None:
        """
        注册服务
        
        Args:
            service_type: 服务类型（接口/抽象类）
            implementation: 实现类（如果与service_type相同可省略）
            factory: 工厂函数
            lifecycle: 服务生命周期
        """
        service_name = service_type.__name__
        
        if service_name in self._services:
            self._logger.warning(f"Service {service_name} already registered, overwriting")
        
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation=implementation or service_type,
            factory=factory,
            lifecycle=lifecycle
        )
        
        self._services[service_name] = descriptor
        self._logger.debug(f"Registered service: {service_name} with lifecycle {lifecycle.value}")
    
    def register_singleton(
        self,
        service_type: Type,
        implementation: Optional[Type] = None,
        instance: Optional[Any] = None
    ) -> None:
        """
        注册单例服务
        
        Args:
            service_type: 服务类型
            implementation: 实现类
            instance: 已有实例（如果提供，将直接使用）
        """
        service_name = service_type.__name__
        
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation=implementation or service_type,
            lifecycle=ServiceLifecycle.SINGLETON,
            instance=instance
        )
        
        self._services[service_name] = descriptor
        self._logger.debug(f"Registered singleton: {service_name}")
    
    def register_transient(
        self,
        service_type: Type,
        implementation: Optional[Type] = None,
        factory: Optional[Callable] = None
    ) -> None:
        """
        注册瞬时服务
        
        Args:
            service_type: 服务类型
            implementation: 实现类
            factory: 工厂函数
        """
        self.register(
            service_type=service_type,
            implementation=implementation,
            factory=factory,
            lifecycle=ServiceLifecycle.TRANSIENT
        )
    
    def register_scoped(
        self,
        service_type: Type,
        implementation: Optional[Type] = None,
        factory: Optional[Callable] = None
    ) -> None:
        """
        注册作用域服务
        
        Args:
            service_type: 服务类型
            implementation: 实现类
            factory: 工厂函数
        """
        self.register(
            service_type=service_type,
            implementation=implementation,
            factory=factory,
            lifecycle=ServiceLifecycle.SCOPED
        )
    
    def get(self, service_type: Type) -> Any:
        """
        获取服务实例
        
        Args:
            service_type: 服务类型
            
        Returns:
            服务实例
        """
        service_name = service_type.__name__
        
        if service_name not in self._services:
            raise KeyError(f"Service {service_name} not registered")
        
        descriptor = self._services[service_name]
        
        # 根据生命周期返回实例
        if descriptor.lifecycle == ServiceLifecycle.SINGLETON:
            if descriptor.instance is None:
                descriptor.instance = self._create_instance(descriptor)
            return descriptor.instance
        
        elif descriptor.lifecycle == ServiceLifecycle.TRANSIENT:
            return self._create_instance(descriptor)
        
        elif descriptor.lifecycle == ServiceLifecycle.SCOPED:
            if service_name not in self._scoped_instances:
                self._scoped_instances[service_name] = self._create_instance(descriptor)
            return self._scoped_instances[service_name]
        
        else:
            raise ValueError(f"Unknown lifecycle: {descriptor.lifecycle}")
    
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """创建服务实例"""
        try:
            if descriptor.factory:
                # 使用工厂函数
                return descriptor.factory(self)
            
            # 使用构造函数
            implementation = descriptor.implementation
            
            # 获取构造函数的参数
            import inspect
            if hasattr(implementation, '__init__'):
                signature = inspect.signature(implementation.__init__)
                parameters = list(signature.parameters.keys())[1:]  # 跳过self
            else:
                parameters = []
            
            # 解析依赖
            dependencies = {}
            for param in parameters:
                # 尝试从容器中获取依赖
                try:
                    # 这里简化处理，实际应该根据类型注解来解析
                    if param in descriptor.dependencies:
                        dep_type = descriptor.dependencies[param]
                        dependencies[param] = self.get(dep_type)
                except KeyError:
                    # 依赖未注册，使用默认值
                    pass
            
            # 创建实例
            if dependencies:
                return implementation(**dependencies)
            else:
                return implementation()
                
        except Exception as e:
            self._logger.error(f"Failed to create instance of {descriptor.service_type.__name__}: {e}")
            raise
    
    def resolve_dependencies(self, target_type: Type) -> Dict[str, Any]:
        """
        解析目标类型的所有依赖
        
        Args:
            target_type: 目标类型
            
        Returns:
            依赖字典
        """
        dependencies = {}
        
        import inspect
        if hasattr(target_type, '__init__'):
            signature = inspect.signature(target_type.__init__)
            for name, param in signature.parameters.items():
                if name == 'self':
                    continue
                
                # 获取参数的类型注解
                annotation = param.annotation
                if annotation != inspect.Parameter.empty:
                    try:
                        dependencies[name] = self.get(annotation)
                    except KeyError:
                        # 依赖未注册，尝试使用默认值
                        if param.default != inspect.Parameter.empty:
                            dependencies[name] = param.default
                        else:
                            raise ValueError(f"Dependency {name} of type {annotation} not registered")
        
        return dependencies
    
    def create_with_dependencies(self, target_type: Type) -> Any:
        """
        使用依赖注入创建实例
        
        Args:
            target_type: 目标类型
            
        Returns:
            创建的实例
        """
        dependencies = self.resolve_dependencies(target_type)
        return target_type(**dependencies)
    
    @contextmanager
    def create_scope(self):
        """
        创建作用域上下文
        
        Usage:
            with container.create_scope():
                service = container.get(SomeScopedService)
                # 在作用域内，service是单例的
        """
        old_scoped_instances = self._scoped_instances.copy()
        self._scoped_instances = {}
        
        try:
            yield self
        finally:
            self._scoped_instances = old_scoped_instances
    
    def register_database_services(self) -> None:
        """注册数据库相关服务"""
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker, Session
            
            # 创建数据库引擎
            engine = create_engine(
                self.config.database.connection_string,
                pool_size=self.config.database.pool_size,
                max_overflow=self.config.database.max_overflow,
                echo=self.config.database.echo
            )
            
            # 注册引擎
            self.register_singleton(type(engine), instance=engine)
            
            # 注册Session工厂
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            self.register_singleton(type(SessionLocal), instance=SessionLocal)
            
            self._logger.info("Database services registered successfully")
            
        except ImportError:
            self._logger.warning("SQLAlchemy not available, skipping database services")
        except Exception as e:
            self._logger.error(f"Failed to register database services: {e}")
    
    def register_redis_services(self) -> None:
        """注册Redis相关服务"""
        try:
            import redis
            
            # 创建Redis连接
            redis_client = redis.Redis(
                host=self.config.redis.host,
                port=self.config.redis.port,
                password=self.config.redis.password,
                db=self.config.redis.db,
                decode_responses=self.config.redis.decode_responses
            )
            
            # 测试连接
            redis_client.ping()
            
            # 注册Redis客户端
            self.register_singleton(type(redis_client), instance=redis_client)
            
            self._logger.info("Redis services registered successfully")
            
        except ImportError:
            self._logger.warning("Redis not available, skipping Redis services")
        except Exception as e:
            self._logger.error(f"Failed to register Redis services: {e}")
    
    def register_vector_search_services(self) -> None:
        """注册向量搜索相关服务"""
        try:
            if self.config.vector_search.provider == "qdrant":
                from qdrant_client import QdrantClient
                
                # 创建Qdrant客户端
                qdrant_client = QdrantClient(
                    host=self.config.vector_search.host,
                    port=self.config.vector_search.port
                )
                
                # 注册Qdrant客户端
                self.register_singleton(QdrantClient, instance=qdrant_client)
                
                self._logger.info("Qdrant vector search services registered successfully")
            else:
                self._logger.warning(f"Vector search provider {self.config.vector_search.provider} not implemented")
                
        except ImportError:
            self._logger.warning("Qdrant client not available, skipping vector search services")
        except Exception as e:
            self._logger.error(f"Failed to register vector search services: {e}")
    
    def register_ml_services(self) -> None:
        """注册机器学习相关服务"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # 创建嵌入模型
            embedding_model = SentenceTransformer(
                self.config.ml.embedding_model,
                device=self.config.ml.device
            )
            
            # 注册嵌入模型
            self.register_singleton(SentenceTransformer, instance=embedding_model)
            
            self._logger.info("ML services registered successfully")
            
        except ImportError:
            self._logger.warning("SentenceTransformers not available, skipping ML services")
        except Exception as e:
            self._logger.error(f"Failed to register ML services: {e}")
    
    def register_all_services(self) -> None:
        """注册所有服务"""
        self._logger.info("Starting service registration...")
        
        # 注册基础设施服务
        self.register_database_services()
        self.register_redis_services()
        self.register_vector_search_services()
        self.register_ml_services()
        
        # 注册业务服务
        self._register_business_services()
        
        self._logger.info("All services registered successfully")
    
    def _register_business_services(self) -> None:
        """注册业务服务"""
        try:
            # 导入业务服务
            from ..database.services.database_service import DatabaseService
            from ..database.services.redis_cache import RedisCache
            from ..core.engine import TopologyEngine
            from ..core.context_manager import ContextManager
            from ..core.memory_manager import MemoryManager
            
            # 注册数据库服务
            self.register_singleton(DatabaseService)
            
            # 注册Redis缓存
            self.register_singleton(RedisCache)
            
            # 注册拓扑引擎
            self.register_singleton(TopologyEngine)
            
            # 注册上下文管理器
            self.register_singleton(ContextManager)
            
            # 注册内存管理器
            self.register_singleton(MemoryManager)
            
            self._logger.info("Business services registered successfully")
            
        except ImportError as e:
            self._logger.warning(f"Failed to import business services: {e}")
        except Exception as e:
            self._logger.error(f"Failed to register business services: {e}")
    
    def health_check(self) -> Dict[str, bool]:
        """
        健康检查
        
        Returns:
            各服务健康状态字典
        """
        health_status = {
            "container": True,
            "database": False,
            "redis": False,
            "vector_search": False,
            "ml": False
        }
        
        try:
            # 检查数据库
            from sqlalchemy import text
            engine = self.get(type(self.get(ConfigManager).database))
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            health_status["database"] = True
        except:
            pass
        
        try:
            # 检查Redis
            redis_client = self.get(type(self.get(ConfigManager).redis))
            redis_client.ping()
            health_status["redis"] = True
        except:
            pass
        
        try:
            # 检查向量搜索
            if self.config.vector_search.provider == "qdrant":
                qdrant_client = self.get(type(self.get(ConfigManager).vector_search))
                qdrant_client.get_collections()
                health_status["vector_search"] = True
        except:
            pass
        
        try:
            # 检查ML服务
            embedding_model = self.get(type(self.get(ConfigManager).ml))
            # 简单测试
            embedding_model.encode(["test"])
            health_status["ml"] = True
        except:
            pass
        
        return health_status
    
    def __str__(self) -> str:
        """字符串表示"""
        services_info = []
        for name, descriptor in self._services.items():
            services_info.append(f"{name}: {descriptor.lifecycle.value}")
        
        return f"ServiceContainer with {len(self._services)} services:\n" + "\n".join(services_info)


# 全局服务容器实例
_service_container: Optional[ServiceContainer] = None


def get_service_container(config: Optional[ConfigManager] = None) -> ServiceContainer:
    """获取全局服务容器实例"""
    global _service_container
    if _service_container is None:
        _service_container = ServiceContainer(config)
        _service_container.register_all_services()
    return _service_container