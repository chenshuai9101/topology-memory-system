"""
健康检查器 - 全系统健康检查和状态报告
"""

import time
import logging
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from .config_manager import ConfigManager, get_config
from .service_container import ServiceContainer, get_service_container


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CheckSeverity(Enum):
    """检查严重性"""
    CRITICAL = "critical"  # 系统无法运行
    HIGH = "high"          # 主要功能受影响
    MEDIUM = "medium"      # 次要功能受影响
    LOW = "low"            # 不影响功能，需要关注


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    name: str
    status: HealthStatus
    severity: CheckSeverity
    message: str
    duration: float
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "status": self.status.value,
            "severity": self.severity.value,
            "message": self.message,
            "duration": self.duration,
            "timestamp": self.timestamp,
            "details": self.details
        }


@dataclass
class HealthCheck:
    """健康检查定义"""
    name: str
    check_func: Callable[[], HealthCheckResult]
    interval: int = 30  # 检查间隔（秒）
    timeout: int = 10   # 超时时间（秒）
    enabled: bool = True
    last_run: Optional[float] = None
    last_result: Optional[HealthCheckResult] = None


class HealthChecker:
    """健康检查器"""
    
    def __init__(self, config: Optional[ConfigManager] = None, container: Optional[ServiceContainer] = None):
        """
        初始化健康检查器
        
        Args:
            config: 配置管理器
            container: 服务容器
        """
        self.config = config or get_config()
        self.container = container or get_service_container(self.config)
        self.logger = logging.getLogger(__name__)
        
        # 健康检查注册表
        self.checks: Dict[str, HealthCheck] = {}
        
        # 监控线程
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_running = False
        
        # 注册默认检查
        self._register_default_checks()
    
    def _register_default_checks(self) -> None:
        """注册默认健康检查"""
        # 配置检查
        self.register_check(
            name="configuration",
            check_func=self._check_configuration,
            interval=60,
            severity=CheckSeverity.CRITICAL
        )
        
        # 数据库检查
        self.register_check(
            name="database",
            check_func=self._check_database,
            interval=30,
            severity=CheckSeverity.CRITICAL
        )
        
        # Redis检查
        self.register_check(
            name="redis",
            check_func=self._check_redis,
            interval=30,
            severity=CheckSeverity.HIGH
        )
        
        # 向量搜索检查
        self.register_check(
            name="vector_search",
            check_func=self._check_vector_search,
            interval=30,
            severity=CheckSeverity.HIGH
        )
        
        # ML服务检查
        self.register_check(
            name="ml_services",
            check_func=self._check_ml_services,
            interval=60,
            severity=CheckSeverity.MEDIUM
        )
        
        # API服务检查
        self.register_check(
            name="api_service",
            check_func=self._check_api_service,
            interval=30,
            severity=CheckSeverity.CRITICAL
        )
        
        # 内存使用检查
        self.register_check(
            name="memory_usage",
            check_func=self._check_memory_usage,
            interval=60,
            severity=CheckSeverity.MEDIUM
        )
        
        # 磁盘空间检查
        self.register_check(
            name="disk_space",
            check_func=self._check_disk_space,
            interval=300,  # 5分钟
            severity=CheckSeverity.MEDIUM
        )
        
        # 系统负载检查
        self.register_check(
            name="system_load",
            check_func=self._check_system_load,
            interval=60,
            severity=CheckSeverity.LOW
        )
    
    def register_check(
        self,
        name: str,
        check_func: Callable[[], HealthCheckResult],
        interval: int = 30,
        timeout: int = 10,
        severity: CheckSeverity = CheckSeverity.MEDIUM,
        enabled: bool = True
    ) -> None:
        """
        注册健康检查
        
        Args:
            name: 检查名称
            check_func: 检查函数
            interval: 检查间隔（秒）
            timeout: 超时时间（秒）
            severity: 严重性
            enabled: 是否启用
        """
        # 包装检查函数以添加超时和错误处理
        def wrapped_check() -> HealthCheckResult:
            start_time = time.time()
            
            try:
                # 设置超时
                result = self._run_with_timeout(check_func, timeout)
                duration = time.time() - start_time
                
                # 确保结果是HealthCheckResult类型
                if not isinstance(result, HealthCheckResult):
                    result = HealthCheckResult(
                        name=name,
                        status=HealthStatus.UNKNOWN,
                        severity=severity,
                        message=f"Check function returned invalid result type: {type(result)}",
                        duration=duration,
                        timestamp=time.time()
                    )
                
                return result
                
            except TimeoutError:
                duration = time.time() - start_time
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    severity=severity,
                    message=f"Check timed out after {timeout} seconds",
                    duration=duration,
                    timestamp=time.time()
                )
            except Exception as e:
                duration = time.time() - start_time
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    severity=severity,
                    message=f"Check failed with error: {str(e)}",
                    duration=duration,
                    timestamp=time.time(),
                    details={"error": str(e)}
                )
        
        self.checks[name] = HealthCheck(
            name=name,
            check_func=wrapped_check,
            interval=interval,
            timeout=timeout,
            enabled=enabled
        )
        
        self.logger.debug(f"Registered health check: {name} (interval: {interval}s)")
    
    def _run_with_timeout(self, func: Callable, timeout: int) -> Any:
        """带超时运行函数"""
        result = [None]
        exception = [None]
        
        def worker():
            try:
                result[0] = func()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            raise TimeoutError(f"Function execution timed out after {timeout} seconds")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]
    
    def _check_configuration(self) -> HealthCheckResult:
        """检查配置"""
        start_time = time.time()
        
        try:
            # 验证配置
            if not self.config.validate():
                return HealthCheckResult(
                    name="configuration",
                    status=HealthStatus.UNHEALTHY,
                    severity=CheckSeverity.CRITICAL,
                    message="Configuration validation failed",
                    duration=time.time() - start_time,
                    timestamp=time.time()
                )
            
            # 检查必要配置项
            required_configs = [
                ("database.host", self.config.database.host),
                ("database.database", self.config.database.database),
                ("api.port", self.config.api.port)
            ]
            
            missing = []
            for name, value in required_configs:
                if not value:
                    missing.append(name)
            
            if missing:
                return HealthCheckResult(
                    name="configuration",
                    status=HealthStatus.UNHEALTHY,
                    severity=CheckSeverity.CRITICAL,
                    message=f"Missing required configuration: {', '.join(missing)}",
                    duration=time.time() - start_time,
                    timestamp=time.time(),
                    details={"missing_configs": missing}
                )
            
            return HealthCheckResult(
                name="configuration",
                status=HealthStatus.HEALTHY,
                severity=CheckSeverity.CRITICAL,
                message="Configuration is valid",
                duration=time.time() - start_time,
                timestamp=time.time(),
                details={"environment": self.config.env.value}
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="configuration",
                status=HealthStatus.UNHEALTHY,
                severity=CheckSeverity.CRITICAL,
                message=f"Configuration check failed: {str(e)}",
                duration=time.time() - start_time,
                timestamp=time.time(),
                details={"error": str(e)}
            )
    
    def _check_database(self) -> HealthCheckResult:
        """检查数据库"""
        start_time = time.time()
        
        try:
            from sqlalchemy import text
            
            # 获取数据库引擎
            engine = self.container.get(type(self.config.database))
            
            # 测试连接和查询
            with engine.connect() as conn:
                # 简单查询
                result = conn.execute(text("SELECT 1 as test, version() as version"))
                row = result.fetchone()
                
                # 检查表是否存在
                table_check = conn.execute(text("""
                    SELECT COUNT(*) as table_count 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """))
                table_count = table_check.fetchone()[0]
            
            duration = time.time() - start_time
            
            return HealthCheckResult(
                name="database",
                status=HealthStatus.HEALTHY,
                severity=CheckSeverity.CRITICAL,
                message="Database is accessible",
                duration=duration,
                timestamp=time.time(),
                details={
                    "connection_time": duration,
                    "table_count": table_count,
                    "test_result": row[0] if row else None
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return HealthCheckResult(
                name="database",
                status=HealthStatus.UNHEALTHY,
                severity=CheckSeverity.CRITICAL,
                message=f"Database check failed: {str(e)}",
                duration=duration,
                timestamp=time.time(),
                details={"error": str(e)}
            )
    
    def _check_redis(self) -> HealthCheckResult:
        """检查Redis"""
        start_time = time.time()
        
        try:
            import redis
            
            # 获取Redis客户端
            redis_client = self.container.get(type(self.config.redis))
            
            # 测试连接
            pong = redis_client.ping()
            
            # 测试读写
            test_key = f"health_check:{int(time.time())}"
            redis_client.set(test_key, "test", ex=10)
            value = redis_client.get(test_key)
            redis_client.delete(test_key)
            
            # 获取信息
            info = redis_client.info()
            
            duration = time.time() - start_time
            
            return HealthCheckResult(
                name="redis",
                status=HealthStatus.HEALTHY,
                severity=CheckSeverity.HIGH,
                message="Redis is accessible",
                duration=duration,
                timestamp=time.time(),
                details={
                    "connection_time": duration,
                    "ping": pong,
                    "test_read_write": value == b"test",
                    "used_memory": info.get('used_memory', 0),
                    "connected_clients": info.get('connected_clients', 0)
                }
            )
            
        except ImportError:
            duration = time.time() - start_time
            return HealthCheckResult(
                name="redis",
                status=HealthStatus.UNKNOWN,
                severity=CheckSeverity.LOW,
                message="Redis client not available",
                duration=duration,
                timestamp=time.time()
            )
        except Exception as e:
            duration = time.time() - start_time
            return HealthCheckResult(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                severity=CheckSeverity.HIGH,
                message=f"Redis check failed: {str(e)}",
                duration=duration,
                timestamp=time.time(),
                details={"error": str(e)}
            )
    
    def _check_vector_search(self) -> HealthCheckResult:
        """检查向量搜索"""
        start_time = time.time()
        
        try:
            if self.config.vector_search.provider == "qdrant":
                from qdrant_client import QdrantClient
                
                # 获取Qdrant客户端
                qdrant_client = self.container.get(QdrantClient)
                
                # 测试连接
                collections = qdrant_client.get_collections()
                
                # 检查目标集合
                target_collection = None
                for collection in collections.collections:
                    if collection.name == self.config.vector_search.collection_name:
                        target_collection = collection
                        break
                
                duration = time.time() - start_time
                
                if target_collection:
                    return HealthCheckResult(
                        name="vector_search",
                        status=HealthStatus.HEALTHY,
                        severity=CheckSeverity.HIGH,
                        message="Vector search is accessible",
                        duration=duration,
                        timestamp=time.time(),
                        details={
                            "connection_time": duration,
                            "provider": self.config.vector_search.provider,
                            "collection_exists": True,
                            "collection_name": target_collection.name,
                            "total_collections": len(collections.collections)
                        }
                    )
                else:
                    return HealthCheckResult(
                        name="vector_search",
                        status=HealthStatus.DEGRADED,
                        severity=CheckSeverity.HIGH,
                        message=f"Vector search collection '{self.config.vector_search.collection_name}' not found",
                        duration=duration,
                        timestamp=time.time(),
                        details={
                            "connection_time": duration,
                            "provider": self.config.vector_search.provider,
                            "collection_exists": False,
                            "available_collections": [c.name for c in collections.collections]
                        }
                    )
            else:
                duration = time.time() - start_time
                return HealthCheckResult(
                    name="vector_search",
                    status=HealthStatus.UNKNOWN,
                    severity=CheckSeverity.LOW,
                    message=f"Vector search provider '{self.config.vector_search.provider}' not implemented",
                    duration=duration,
                    timestamp=time.time()
                )
                
        except ImportError:
            duration = time.time() - start_time
            return HealthCheckResult(
                name="vector_search",
                status=HealthStatus.UNKNOWN,
                severity=CheckSeverity.LOW,
                message="Vector search client not available",
                duration=duration,
                timestamp=time.time()
            )
        except Exception as e:
            duration = time.time() - start_time
            return HealthCheckResult(
                name="vector_search",
                status=HealthStatus.UNHEALTHY,
                severity=CheckSeverity.HIGH,
                message=f"Vector search check failed: {str(e)}",
                duration=duration,
                timestamp=time.time(),
                details={"error": str(e)}
            )
    
    def _check_ml_services(self) -> HealthCheckResult:
        """检查ML服务"""
        start_time = time.time()
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # 获取嵌入模型
            embedding_model = self.container.get(SentenceTransformer)
            
            # 测试模型
            test_sentences = ["This is a test sentence for health check."]
            embeddings = embedding_model.encode(test_sentences)
            
            duration = time.time() - start_time
            
            return HealthCheckResult(
                name="ml_services",
                status=HealthStatus.HEALTHY,
                severity=CheckSeverity.MEDIUM,
                message="ML services are working",
                duration=duration,
                timestamp=time.time(),
                details={
                    "connection_time": duration,
                    "model": self.config.ml.embedding_model,
                    "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings),
                    "device": self.config.ml.device
                }
            )
            
        except ImportError:
            duration = time.time() - start_time
            return HealthCheckResult(
                name="ml_services",
                status=HealthStatus.UNKNOWN,
                severity=CheckSeverity.LOW,
                message="ML libraries not available",
                duration=duration,
                timestamp=time.time()
            )
        except Exception as e:
            duration = time.time() - start_time
            return HealthCheckResult(
                name="ml_services",
                status=HealthStatus.UNHEALTHY,
                severity=CheckSeverity.MEDIUM,
                message=f"ML services check failed: {str(e)}",
                duration=duration,
                timestamp=time.time(),
                details={"error": str(e)}
            )
    
    def _check_api_service(self) -> HealthCheckResult:
        """检查API服务"""
        start_time = time.time()
        
        try:
            import requests
            
            # 尝试访问健康检查端点
            health_url = f"{self.config.api.server_url}{self.config.monitoring.health_check_path}"
            
            response = requests.get(health_url, timeout=5)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                return HealthCheckResult(
                    name="api_service",
                    status=HealthStatus.HEALTHY if data.get("status") == "healthy" else HealthStatus.DEGRADED,
                    severity=CheckSeverity.CRITICAL,
                    message="API service is responding",
                    duration=duration,
                    timestamp=time.time(),
                    details={
                        "response_time": duration,
                        "status_code": response.status_code,
                        "api_status": data.get("status", "unknown"),
                        "version": data.get("version", "unknown")
                    }
                )
            else:
                return HealthCheckResult(
                    name="api_service",
                    status=HealthStatus.UNHEALTHY,
                    severity=CheckSeverity.CRITICAL,
                    message=f"API service returned status {response.status_code}",
                    duration=duration,
                    timestamp=time.time(),
                    details={
                        "response_time": duration,
                        "status_code": response.status_code,
                        "response_text": response.text[:100]
                    }
                )
                
        except requests.exceptions.ConnectionError:
            duration = time.time() - start_time
            return HealthCheckResult(
                name="api_service",
                status=HealthStatus.UNHEALTHY,
                severity=CheckSeverity.CRITICAL,
                message="API service is not reachable",
                duration=duration,
                timestamp=time.time()
            )
        except Exception as e:
            duration = time.time() - start_time
            return HealthCheckResult(
                name="api_service",
                status=HealthStatus.UNHEALTHY,
                severity=CheckSeverity.CRITICAL,
                message=f"API service check failed: {str(e)}",
                duration=duration,
                timestamp=time.time(),
                details={"error": str(e)}
            )
    
    def _check_memory_usage(self) -> HealthCheckResult:
        """检查内存使用"""
        start_time = time.time()
        
        try:
            import psutil
            
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # 获取系统内存信息
            system_memory = psutil.virtual_memory()
            
            duration = time.time() - start_time
            
            process_memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = process_memory_percent = (memory_info.rss / system_memory.total) * 100
            
            status = HealthStatus.HEALTHY
            if memory_percent > 80:
                status = HealthStatus.DEGRADED
            elif memory_percent > 95:
                status = HealthStatus.UNHEALTHY
            
            return HealthCheckResult(
                name="memory_usage",
                status=status,
                severity=CheckSeverity.MEDIUM,
                message=f"Memory usage: {process_memory_mb:.1f}MB ({memory_percent:.1f}%)",
                duration=duration,
                timestamp=time.time(),
                details={
                    "process_memory_mb": process_memory_mb,
                    "process_memory_percent": memory_percent,
                    "system_memory_total_mb": system_memory.total / 1024 / 1024,
                    "system_memory_available_mb": system_memory.available / 1024 / 1024,
                    "system_memory_percent": system_memory.percent
                }
            )
            
        except ImportError:
            duration = time.time() - start_time
            return HealthCheckResult(
                name="memory_usage",
                status=HealthStatus.UNKNOWN,
                severity=CheckSeverity.LOW,
                message="psutil not available for memory check",
                duration=duration,
                timestamp=time.time()
            )
        except Exception as e:
            duration = time.time() - start_time
            return HealthCheckResult(
                name="memory_usage",
                status=HealthStatus.UNKNOWN,
                severity=CheckSeverity.MEDIUM,
                message=f"Memory check failed: {str(e)}",
                duration=duration,
                timestamp=time.time(),
                details={"error": str(e)}
            )
    
    def _check_disk_space(self) -> HealthCheckResult:
        """检查磁盘空间"""
        start_time = time.time()
        
        try:
            import psutil
            import os
            
            # 检查当前工作目录的磁盘空间
            disk_usage = psutil.disk_usage(os.getcwd())
            
            duration = time.time() - start_time
            
            free_gb = disk_usage.free / 1024 / 1024 / 1024
            free_percent = disk_usage.free / disk_usage.total * 100
            
            status = HealthStatus.HEALTHY
            if free_percent < 20:
                status = HealthStatus.DEGRADED
            elif free_percent < 5:
                status = HealthStatus.UNHEALTHY
            
            return HealthCheckResult(
                name="disk_space",
                status=status,
                severity=CheckSeverity.MEDIUM,
                message=f"Disk space: {free_gb:.1f}GB free ({free_percent:.1f}%)",
                duration=duration,
                timestamp=time.time(),
                details={
                    "free_gb": free_gb,
                    "free_percent": free_percent,
                    "total_gb": disk_usage.total / 1024 / 1024 / 1024,
                    "used_gb": disk_usage.used / 1024 / 1024 / 1024,
                    "used_percent": disk_usage.percent
                }
            )
            
        except ImportError:
            duration = time.time() - start_time
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                severity=CheckSeverity.LOW,
                message="psutil not available for disk space check",
                duration=duration,
                timestamp=time.time()
            )
        except Exception as e:
            duration = time.time() - start_time
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                severity=CheckSeverity.MEDIUM,
                message=f"Disk space check failed: {str(e)}",
                duration=duration,
                timestamp=time.time(),
                details={"error": str(e)}
            )
    
    def _check_system_load(self) -> HealthCheckResult:
        """检查系统负载"""
        start_time = time.time()
        
        try:
            import psutil
            
            # 获取系统负载
            load_avg = psutil.getloadavg()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            duration = time.time() - start_time
            
            # 简单的负载判断
            status = HealthStatus.HEALTHY
            if load_avg[0] > psutil.cpu_count() * 0.8:  # 1分钟负载超过CPU核心数的80%
                status = HealthStatus.DEGRADED
            if cpu_percent > 90:
                status = HealthStatus.DEGRADED
            
            return HealthCheckResult(
                name="system_load",
                status=status,
                severity=CheckSeverity.LOW,
                message=f"System load: {load_avg[0]:.2f} (1min), CPU: {cpu_percent:.1f}%",
                duration=duration,
                timestamp=time.time(),
                details={
                    "load_1min": load_avg[0],
                    "load_5min": load_avg[1],
                    "load_15min": load_avg[2],
                    "cpu_percent": cpu_percent,
                    "cpu_count": psutil.cpu_count()
                }
            )
            
        except ImportError:
            duration = time.time() - start_time
            return HealthCheckResult(
                name="system_load",
                status=HealthStatus.UNKNOWN,
                severity=CheckSeverity.LOW,
                message="psutil not available for system load check",
                duration=duration,
                timestamp=time.time()
            )
        except Exception as e:
            duration = time.time() - start_time
            return HealthCheckResult(
                name="system_load",
                status=HealthStatus.UNKNOWN,
                severity=CheckSeverity.LOW,
                message=f"System load check failed: {str(e)}",
                duration=duration,
                timestamp=time.time(),
                details={"error": str(e)}
            )
    
    def run_check(self, name: str) -> HealthCheckResult:
        """
        运行指定健康检查
        
        Args:
            name: 检查名称
            
        Returns:
            健康检查结果
        """
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                severity=CheckSeverity.LOW,
                message=f"Health check '{name}' not found",
                duration=0.0,
                timestamp=time.time()
            )
        
        check = self.checks[name]
        if not check.enabled:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                severity=CheckSeverity.LOW,
                message=f"Health check '{name}' is disabled",
                duration=0.0,
                timestamp=time.time()
            )
        
        result = check.check_func()
        check.last_run = time.time()
        check.last_result = result
        
        # 根据严重性记录日志
        if result.status == HealthStatus.UNHEALTHY:
            if result.severity == CheckSeverity.CRITICAL:
                self.logger.error(f"Critical health check failed: {name} - {result.message}")
            elif result.severity == CheckSeverity.HIGH:
                self.logger.warning(f"High severity health check failed: {name} - {result.message}")
            else:
                self.logger.info(f"Health check failed: {name} - {result.message}")
        
        return result
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """
        运行所有健康检查
        
        Returns:
            所有检查结果字典
        """
        results = {}
        
        for name in self.checks:
            if self.checks[name].enabled:
                results[name] = self.run_check(name)
        
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """
        获取整体健康状态
        
        Returns:
            整体健康状态
        """
        results = self.run_all_checks()
        
        if not results:
            return HealthStatus.UNKNOWN
        
        # 根据最严重的问题确定整体状态
        critical_failed = any(
            r.status == HealthStatus.UNHEALTHY and r.severity == CheckSeverity.CRITICAL
            for r in results.values()
        )
        
        high_failed = any(
            r.status == HealthStatus.UNHEALTHY and r.severity == CheckSeverity.HIGH
            for r in results.values()
        )
        
        any_failed = any(
            r.status == HealthStatus.UNHEALTHY
            for r in results.values()
        )
        
        any_degraded = any(
            r.status == HealthStatus.DEGRADED
            for r in results.values()
        )
        
        if critical_failed:
            return HealthStatus.UNHEALTHY
        elif high_failed:
            return HealthStatus.UNHEALTHY
        elif any_failed:
            return HealthStatus.DEGRADED
        elif any_degraded:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def start_monitoring(self, interval: int = 30) -> None:
        """
        启动健康监控
        
        Args:
            interval: 监控间隔（秒）
        """
        if self.monitor_running:
            self.logger.warning("Health monitor is already running")
            return
        
        self.monitor_running = True
        
        def monitor_loop():
            self.logger.info(f"Starting health monitor with {interval}s interval")
            
            while self.monitor_running:
                try:
                    # 运行所有检查
                    results = self.run_all_checks()
                    
                    # 检查是否有严重问题
                    critical_issues = [
                        name for name, result in results.items()
                        if result.status == HealthStatus.UNHEALTHY and result.severity in [CheckSeverity.CRITICAL, CheckSeverity.HIGH]
                    ]
                    
                    if critical_issues:
                        self.logger.error(f"Critical health issues detected: {', '.join(critical_issues)}")
                    
                except Exception as e:
                    self.logger.error(f"Health monitor error: {e}")
                
                # 等待下一次检查
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Health monitor started")
    
    def stop_monitoring(self) -> None:
        """停止健康监控"""
        if not self.monitor_running:
            return
        
        self.monitor_running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Health monitor stopped")
    
    def get_status_report(self) -> Dict[str, Any]:
        """获取状态报告"""
        results = self.run_all_checks()
        overall_status = self.get_overall_status()
        
        # 统计
        total_checks = len(results)
        healthy_checks = sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY)
        degraded_checks = sum(1 for r in results.values() if r.status == HealthStatus.DEGRADED)
        unhealthy_checks = sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY)
        
        # 按严重性分组
        critical_issues = [
            name for name, result in results.items()
            if result.status == HealthStatus.UNHEALTHY and result.severity == CheckSeverity.CRITICAL
        ]
        
        high_issues = [
            name for name, result in results.items()
            if result.status == HealthStatus.UNHEALTHY and result.severity == CheckSeverity.HIGH
        ]
        
        return {
            "timestamp": time.time(),
            "overall_status": overall_status.value,
            "summary": {
                "total_checks": total_checks,
                "healthy_checks": healthy_checks,
                "degraded_checks": degraded_checks,
                "unhealthy_checks": unhealthy_checks,
                "critical_issues": len(critical_issues),
                "high_issues": len(high_issues)
            },
            "checks": {name: result.to_dict() for name, result in results.items()},
            "issues": {
                "critical": critical_issues,
                "high": high_issues
            },
            "monitoring": {
                "running": self.monitor_running,
                "interval": next((c.interval for c in self.checks.values() if c.enabled), 30)
            }
        }
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_monitoring()
    
    def __str__(self) -> str:
        """字符串表示"""
        report = self.get_status_report()
        
        lines = [
            f"Health Checker - Overall Status: {report['overall_status'].upper()}",
            f"Checks: {report['summary']['healthy_checks']} healthy, "
            f"{report['summary']['degraded_checks']} degraded, "
            f"{report['summary']['unhealthy_checks']} unhealthy",
        ]
        
        if report['summary']['critical_issues'] > 0:
            lines.append(f"Critical Issues: {', '.join(report['issues']['critical'])}")
        
        if report['summary']['high_issues'] > 0:
            lines.append(f"High Issues: {', '.join(report['issues']['high'])}")
        
        return "\n".join(lines)


# 全局健康检查器实例
_health_checker: Optional[HealthChecker] = None


def get_health_checker(
    config: Optional[ConfigManager] = None,
    container: Optional[ServiceContainer] = None
) -> HealthChecker:
    """获取全局健康检查器实例"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker(config, container)
    return _health_checker


def check_system_health() -> Dict[str, Any]:
    """检查系统健康（便捷函数）"""
    checker = get_health_checker()
    return checker.get_status_report()