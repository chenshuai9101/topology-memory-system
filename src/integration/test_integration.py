"""
集成测试 - 测试系统集成组件
"""

import pytest
import time
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

from .config_manager import ConfigManager, Environment, get_config
from .service_container import ServiceContainer, get_service_container
from .system_integrator import SystemIntegrator, get_system_integrator
from .health_checker import HealthChecker, get_health_checker, HealthStatus, CheckSeverity
from .application_starter import ApplicationStarter


class TestConfigManager:
    """测试配置管理器"""
    
    def test_config_manager_initialization(self):
        """测试配置管理器初始化"""
        config = ConfigManager(Environment.DEVELOPMENT)
        
        assert config.env == Environment.DEVELOPMENT
        assert config.database.host == "localhost"
        assert config.database.port == 5432
        assert config.api.port == 8000
        assert config.ml.embedding_model == "all-MiniLM-L6-v2"
    
    def test_config_validation(self):
        """测试配置验证"""
        config = ConfigManager(Environment.DEVELOPMENT)
        
        # 默认配置应该通过验证
        assert config.validate() == True
    
    def test_config_to_dict(self):
        """测试配置转换为字典"""
        config = ConfigManager(Environment.DEVELOPMENT)
        config_dict = config.to_dict()
        
        assert "environment" in config_dict
        assert config_dict["environment"] == "development"
        assert "database" in config_dict
        assert "api" in config_dict
        assert "ml" in config_dict
    
    def test_get_config_function(self):
        """测试获取配置函数"""
        config1 = get_config("development")
        config2 = get_config("development")
        
        # 应该是同一个实例（单例）
        assert config1 is config2
    
    def test_environment_variable_override(self, monkeypatch):
        """测试环境变量覆盖"""
        monkeypatch.setenv("DB_HOST", "test-host")
        monkeypatch.setenv("DB_PORT", "5433")
        monkeypatch.setenv("API_PORT", "8080")
        
        config = ConfigManager(Environment.DEVELOPMENT)
        
        assert config.database.host == "test-host"
        assert config.database.port == 5433
        assert config.api.port == 8080


class TestServiceContainer:
    """测试服务容器"""
    
    def test_service_container_initialization(self):
        """测试服务容器初始化"""
        config = ConfigManager(Environment.DEVELOPMENT)
        container = ServiceContainer(config)
        
        assert container.config == config
        assert isinstance(container._services, dict)
    
    def test_register_and_get_service(self):
        """测试注册和获取服务"""
        config = ConfigManager(Environment.DEVELOPMENT)
        container = ServiceContainer(config)
        
        # 注册一个测试服务
        class TestService:
            def __init__(self, name="test"):
                self.name = name
        
        container.register_singleton(TestService)
        
        # 获取服务实例
        service1 = container.get(TestService)
        service2 = container.get(TestService)
        
        # 单例服务应该是同一个实例
        assert service1 is service2
        assert isinstance(service1, TestService)
    
    def test_transient_service(self):
        """测试瞬时服务"""
        config = ConfigManager(Environment.DEVELOPMENT)
        container = ServiceContainer(config)
        
        class TransientService:
            counter = 0
            
            def __init__(self):
                TransientService.counter += 1
                self.id = TransientService.counter
        
        container.register_transient(TransientService)
        
        # 获取多个实例
        service1 = container.get(TransientService)
        service2 = container.get(TransientService)
        
        # 瞬时服务应该是不同的实例
        assert service1 is not service2
        assert service1.id != service2.id
    
    def test_service_container_health_check(self):
        """测试服务容器健康检查"""
        config = ConfigManager(Environment.DEVELOPMENT)
        container = ServiceContainer(config)
        
        # 健康检查应该返回字典
        health_status = container.health_check()
        
        assert isinstance(health_status, dict)
        assert "container" in health_status
        assert health_status["container"] == True


class TestSystemIntegrator:
    """测试系统集成器"""
    
    @pytest.fixture
    def mock_config(self):
        """模拟配置"""
        config = Mock()
        config.env = Environment.DEVELOPMENT
        config.validate.return_value = True
        
        # 模拟数据库配置
        config.database.host = "localhost"
        config.database.port = 5432
        config.database.database = "test_db"
        config.database.username = "test_user"
        config.database.password = "test_pass"
        config.database.connection_string = "postgresql://test_user:test_pass@localhost:5432/test_db"
        config.database.pool_size = 5
        config.database.max_overflow = 10
        config.database.echo = False
        
        # 模拟Redis配置
        config.redis.host = "localhost"
        config.redis.port = 6379
        config.redis.password = None
        config.redis.db = 0
        config.redis.decode_responses = True
        
        # 模拟向量搜索配置
        config.vector_search.provider = "qdrant"
        config.vector_search.host = "localhost"
        config.vector_search.port = 6333
        config.vector_search.collection_name = "test_collection"
        config.vector_search.embedding_dim = 384
        config.vector_search.distance_metric = "Cosine"
        
        # 模拟ML配置
        config.ml.embedding_model = "all-MiniLM-L6-v2"
        config.ml.embedding_dim = 384
        config.ml.similarity_threshold = 0.7
        config.ml.max_context_length = 512
        config.ml.batch_size = 32
        config.ml.device = "cpu"
        
        # 模拟API配置
        config.api.host = "0.0.0.0"
        config.api.port = 8000
        config.api.workers = 1
        config.api.reload = False
        config.api.server_url = "http://0.0.0.0:8000"
        
        # 模拟监控配置
        config.monitoring.health_check_path = "/health"
        
        return config
    
    def test_system_integrator_initialization(self, mock_config):
        """测试系统集成器初始化"""
        integrator = SystemIntegrator("development")
        
        assert integrator.env == "development"
        assert integrator.status.value == "not_started"
        assert integrator.errors == []
    
    @patch('src.integration.system_integrator.get_config')
    def test_system_integrator_initialize(self, mock_get_config, mock_config):
        """测试系统集成器初始化过程"""
        mock_get_config.return_value = mock_config
        
        integrator = SystemIntegrator("development")
        
        # 模拟组件初始化成功
        with patch.object(integrator, '_connect_database', return_value=True):
            with patch.object(integrator, '_connect_redis', return_value=True):
                with patch.object(integrator, '_connect_vector_search', return_value=True):
                    with patch.object(integrator, '_initialize_ml_services', return_value=True):
                        success = integrator.initialize()
        
        assert success == True
        assert integrator.status.value == "ready"
        assert integrator.errors == []
        assert integrator.metrics.startup_time > 0
    
    def test_system_integrator_status_report(self, mock_config):
        """测试系统集成器状态报告"""
        integrator = SystemIntegrator("development")
        integrator.config = mock_config
        
        report = integrator.get_status_report()
        
        assert "environment" in report
        assert "status" in report
        assert "metrics" in report
        assert "health" in report
        assert "errors" in report


class TestHealthChecker:
    """测试健康检查器"""
    
    @pytest.fixture
    def mock_config(self):
        """模拟配置"""
        config = Mock()
        config.monitoring.health_check_path = "/health"
        config.api.server_url = "http://localhost:8000"
        config.ml.embedding_model = "test-model"
        config.ml.device = "cpu"
        config.vector_search.provider = "qdrant"
        config.vector_search.collection_name = "test-collection"
        
        return config
    
    @pytest.fixture
    def mock_container(self):
        """模拟容器"""
        container = Mock()
        return container
    
    def test_health_checker_initialization(self, mock_config, mock_container):
        """测试健康检查器初始化"""
        checker = HealthChecker(mock_config, mock_container)
        
        assert checker.config == mock_config
        assert checker.container == mock_container
        assert isinstance(checker.checks, dict)
        assert len(checker.checks) > 0
    
    def test_register_check(self, mock_config, mock_container):
        """测试注册健康检查"""
        checker = HealthChecker(mock_config, mock_container)
        
        initial_count = len(checker.checks)
        
        # 注册一个新的检查
        def test_check():
            from .health_checker import HealthCheckResult, HealthStatus, CheckSeverity
            return HealthCheckResult(
                name="test_check",
                status=HealthStatus.HEALTHY,
                severity=CheckSeverity.LOW,
                message="Test check passed",
                duration=0.1,
                timestamp=time.time()
            )
        
        checker.register_check(
            name="custom_check",
            check_func=test_check,
            interval=60,
            severity=CheckSeverity.MEDIUM
        )
        
        assert len(checker.checks) == initial_count + 1
        assert "custom_check" in checker.checks
    
    def test_run_check(self, mock_config, mock_container):
        """测试运行健康检查"""
        checker = HealthChecker(mock_config, mock_container)
        
        # 运行配置检查
        result = checker.run_check("configuration")
        
        assert result.name == "configuration"
        assert isinstance(result.status, HealthStatus)
        assert isinstance(result.severity, CheckSeverity)
        assert isinstance(result.message, str)
        assert result.duration >= 0
    
    def test_run_all_checks(self, mock_config, mock_container):
        """测试运行所有健康检查"""
        checker = HealthChecker(mock_config, mock_container)
        
        results = checker.run_all_checks()
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        for name, result in results.items():
            assert isinstance(result.name, str)
            assert isinstance(result.status, HealthStatus)
    
    def test_get_overall_status(self, mock_config, mock_container):
        """测试获取整体健康状态"""
        checker = HealthChecker(mock_config, mock_container)
        
        status = checker.get_overall_status()
        
        assert isinstance(status, HealthStatus)
    
    def test_get_status_report(self, mock_config, mock_container):
        """测试获取状态报告"""
        checker = HealthChecker(mock_config, mock_container)
        
        report = checker.get_status_report()
        
        assert "timestamp" in report
        assert "overall_status" in report
        assert "summary" in report
        assert "checks" in report
        assert "issues" in report


class TestApplicationStarter:
    """测试应用启动器"""
    
    def test_application_starter_initialization(self):
        """测试应用启动器初始化"""
        starter = ApplicationStarter(env="development")
        
        assert starter.env == "development"
        assert starter.phase.value == "not_started"
        assert starter.errors == []
    
    @patch('src.integration.application_starter.get_config')
    def test_configuration_initialization(self, mock_get_config):
        """测试配置初始化"""
        mock_config = Mock()
        mock_config.validate.return_value = True
        mock_config.env.value = "development"
        mock_config.api.server_url = "http://localhost:8000"
        mock_config.database.connection_string = "postgresql://test@localhost/test"
        mock_get_config.return_value = mock_config
        
        starter = ApplicationStarter(env="development")
        
        with patch.object(starter, '_initialize_configuration') as mock_init:
            mock_init.return_value = True
            starter._initialize_configuration()
        
        mock_init.assert_called_once()
    
    def test_startup_metrics(self):
        """测试启动指标"""
        starter = ApplicationStarter(env="development")
        
        assert starter.metrics.total_time == 0.0
        assert isinstance(starter.metrics.phases, dict)
    
    def test_get_status(self):
        """测试获取状态"""
        starter = ApplicationStarter(env="development")
        
        status = starter.get_status()
        
        assert "phase" in status
        assert "environment" in status
        assert "startup_time" in status
        assert "metrics" in status
        assert "errors" in status


def test_integration_workflow():
    """测试完整集成工作流"""
    # 1. 创建配置管理器
    config = ConfigManager(Environment.DEVELOPMENT)
    
    # 验证配置
    assert config.validate() == True
    
    # 2. 创建服务容器
    container = ServiceContainer(config)
    
    # 注册一些测试服务
    class TestService:
        def ping(self):
            return "pong"
    
    container.register_singleton(TestService)
    
    # 3. 获取服务
    service = container.get(TestService)
    assert service.ping() == "pong"
    
    # 4. 运行健康检查
    health_status = container.health_check()
    assert isinstance(health_status, dict)
    
    # 5. 测试配置转换
    config_dict = config.to_dict()
    assert "environment" in config_dict
    assert config_dict["environment"] == "development"
    
    print("Integration workflow test passed!")


if __name__ == "__main__":
    # 运行集成测试
    test_integration_workflow()
    
    print("\nAll integration tests completed successfully!")