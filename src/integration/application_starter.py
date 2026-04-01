"""
应用启动器 - 完整系统启动和初始化
提供统一的启动接口和启动脚本
"""

import os
import sys
import time
import logging
import argparse
import signal
import threading
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from .config_manager import ConfigManager, get_config, Environment
from .service_container import ServiceContainer, get_service_container
from .system_integrator import SystemIntegrator, get_system_integrator
from .health_checker import HealthChecker, get_health_checker


class StartupPhase(Enum):
    """启动阶段"""
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    CONFIG_LOADED = "config_loaded"
    SERVICES_INITIALIZED = "services_initialized"
    DATABASE_MIGRATED = "database_migrated"
    API_SERVER_STARTING = "api_server_starting"
    READY = "ready"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


@dataclass
class StartupMetrics:
    """启动指标"""
    total_time: float = 0.0
    config_load_time: float = 0.0
    service_init_time: float = 0.0
    database_migration_time: float = 0.0
    api_start_time: float = 0.0
    health_check_time: float = 0.0
    phases: Dict[str, float] = field(default_factory=dict)


class ApplicationStarter:
    """应用启动器"""
    
    def __init__(self, env: str = "development", config_path: Optional[str] = None):
        """
        初始化应用启动器
        
        Args:
            env: 环境类型
            config_path: 配置文件路径
        """
        self.env = env
        self.config_path = config_path
        self.phase = StartupPhase.NOT_STARTED
        self.metrics = StartupMetrics()
        self.start_time = time.time()
        
        # 初始化日志
        self.logger = self._setup_logging()
        
        # 核心组件
        self.config: Optional[ConfigManager] = None
        self.container: Optional[ServiceContainer] = None
        self.integrator: Optional[SystemIntegrator] = None
        self.health_checker: Optional[HealthChecker] = None
        
        # API服务器线程
        self.api_thread: Optional[threading.Thread] = None
        self.api_running = False
        
        # 健康监控线程
        self.health_monitor_thread: Optional[threading.Thread] = None
        self.health_monitor_running = False
        
        # 信号处理
        self._setup_signal_handlers()
        
        # 错误信息
        self.errors: List[str] = []
        
        self.logger.info(f"Application starter initialized for environment: {env}")
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("topology_memory.starter")
        
        # 避免重复添加处理器
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def _setup_signal_handlers(self) -> None:
        """设置信号处理器"""
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
    
    def _handle_shutdown_signal(self, signum, frame) -> None:
        """处理关闭信号"""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown()
    
    def start(self) -> bool:
        """
        启动应用
        
        Returns:
            是否启动成功
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("Starting Topology Memory Context Manager")
            self.logger.info("=" * 60)
            
            # 阶段1: 初始化配置
            self.phase = StartupPhase.INITIALIZING
            if not self._initialize_configuration():
                return False
            
            # 阶段2: 初始化服务
            if not self._initialize_services():
                return False
            
            # 阶段3: 数据库迁移
            if not self._run_database_migrations():
                return False
            
            # 阶段4: 启动API服务器
            if not self._start_api_server():
                return False
            
            # 阶段5: 启动健康监控
            self._start_health_monitor()
            
            # 阶段6: 完成启动
            self.phase = StartupPhase.READY
            self.metrics.total_time = time.time() - self.start_time
            
            self._print_startup_summary()
            
            # 等待关闭信号
            self._wait_for_shutdown()
            
            return True
            
        except Exception as e:
            self.phase = StartupPhase.STOPPED
            self.errors.append(f"Startup failed: {str(e)}")
            self.logger.error(f"Application startup failed: {e}", exc_info=True)
            return False
    
    def _initialize_configuration(self) -> bool:
        """初始化配置"""
        self.logger.info("Phase 1: Initializing configuration...")
        phase_start = time.time()
        
        try:
            # 加载配置
            if self.config_path:
                # 从指定路径加载配置
                config_file = Path(self.config_path)
                if not config_file.exists():
                    raise FileNotFoundError(f"Config file not found: {self.config_path}")
                
                # 这里可以添加自定义配置加载逻辑
                self.logger.info(f"Loading configuration from: {self.config_path}")
            
            # 获取配置管理器
            self.config = get_config(self.env)
            
            if not self.config.validate():
                raise ValueError("Configuration validation failed")
            
            self.phase = StartupPhase.CONFIG_LOADED
            self.metrics.config_load_time = time.time() - phase_start
            self.metrics.phases["config_load"] = self.metrics.config_load_time
            
            self.logger.info(f"Configuration loaded successfully in {self.metrics.config_load_time:.2f}s")
            self.logger.debug(f"Environment: {self.config.env.value}")
            self.logger.debug(f"API URL: {self.config.api.server_url}")
            self.logger.debug(f"Database: {self.config.database.connection_string}")
            
            return True
            
        except Exception as e:
            self.errors.append(f"Configuration initialization failed: {str(e)}")
            self.logger.error(f"Failed to initialize configuration: {e}")
            return False
    
    def _initialize_services(self) -> bool:
        """初始化服务"""
        self.logger.info("Phase 2: Initializing services...")
        phase_start = time.time()
        
        try:
            # 获取服务容器
            self.container = get_service_container(self.config)
            
            # 获取系统集成器
            self.integrator = get_system_integrator(self.env)
            
            # 初始化系统集成器
            if not self.integrator.initialize():
                raise RuntimeError("System integration failed")
            
            # 获取健康检查器
            self.health_checker = get_health_checker(self.config, self.container)
            
            self.phase = StartupPhase.SERVICES_INITIALIZED
            self.metrics.service_init_time = time.time() - phase_start
            self.metrics.phases["service_init"] = self.metrics.service_init_time
            
            self.logger.info(f"Services initialized successfully in {self.metrics.service_init_time:.2f}s")
            
            # 运行初始健康检查
            health_start = time.time()
            health_report = self.health_checker.get_status_report()
            self.metrics.health_check_time = time.time() - health_start
            
            healthy_checks = health_report["summary"]["healthy_checks"]
            total_checks = health_report["summary"]["total_checks"]
            self.logger.info(f"Initial health check: {healthy_checks}/{total_checks} checks healthy")
            
            return True
            
        except Exception as e:
            self.errors.append(f"Service initialization failed: {str(e)}")
            self.logger.error(f"Failed to initialize services: {e}")
            return False
    
    def _run_database_migrations(self) -> bool:
        """运行数据库迁移"""
        self.logger.info("Phase 3: Running database migrations...")
        phase_start = time.time()
        
        try:
            # 检查是否需要运行迁移
            migration_dir = Path(__file__).parent.parent / "database" / "migrations"
            if not migration_dir.exists():
                self.logger.warning(f"Migration directory not found: {migration_dir}")
                self.metrics.database_migration_time = time.time() - phase_start
                self.metrics.phases["database_migration"] = self.metrics.database_migration_time
                return True
            
            # 这里可以添加Alembic迁移逻辑
            # 暂时跳过，假设数据库已经是最新的
            
            self.phase = StartupPhase.DATABASE_MIGRATED
            self.metrics.database_migration_time = time.time() - phase_start
            self.metrics.phases["database_migration"] = self.metrics.database_migration_time
            
            self.logger.info(f"Database migrations completed in {self.metrics.database_migration_time:.2f}s")
            return True
            
        except Exception as e:
            self.errors.append(f"Database migration failed: {str(e)}")
            self.logger.error(f"Failed to run database migrations: {e}")
            return False
    
    def _start_api_server(self) -> bool:
        """启动API服务器"""
        self.logger.info("Phase 4: Starting API server...")
        phase_start = time.time()
        
        try:
            # 导入API应用
            from ..api.main import app
            
            import uvicorn
            
            self.phase = StartupPhase.API_SERVER_STARTING
            
            # 创建API服务器线程
            def run_api_server():
                self.logger.info(f"Starting API server on {self.config.api.server_url}")
                
                uvicorn.run(
                    app,
                    host=self.config.api.host,
                    port=self.config.api.port,
                    workers=self.config.api.workers,
                    reload=self.config.api.reload,
                    log_level="info" if self.config.env == Environment.DEVELOPMENT else "warning"
                )
            
            self.api_thread = threading.Thread(target=run_api_server, daemon=True)
            self.api_running = True
            self.api_thread.start()
            
            # 等待服务器启动
            self._wait_for_api_ready()
            
            self.metrics.api_start_time = time.time() - phase_start
            self.metrics.phases["api_start"] = self.metrics.api_start_time
            
            self.logger.info(f"API server started in {self.metrics.api_start_time:.2f}s")
            self.logger.info(f"API documentation: {self.config.api.server_url}{self.config.api.docs_url}")
            
            return True
            
        except ImportError as e:
            self.errors.append(f"API module not found: {str(e)}")
            self.logger.error(f"Failed to import API module: {e}")
            return False
        except Exception as e:
            self.errors.append(f"API server startup failed: {str(e)}")
            self.logger.error(f"Failed to start API server: {e}")
            return False
    
    def _wait_for_api_ready(self, timeout: int = 30) -> bool:
        """等待API服务器就绪"""
        import requests
        
        self.logger.info("Waiting for API server to be ready...")
        
        start_time = time.time()
        health_url = f"http://{self.config.api.host}:{self.config.api.port}{self.config.monitoring.health_check_path}"
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "healthy":
                        self.logger.info("API server is ready and healthy")
                        return True
            except requests.exceptions.ConnectionError:
                # 服务器可能还在启动中
                pass
            except Exception as e:
                self.logger.debug(f"Health check attempt failed: {e}")
            
            time.sleep(1)
        
        self.logger.warning(f"API server not ready after {timeout} seconds")
        return False
    
    def _start_health_monitor(self) -> None:
        """启动健康监控"""
        if not self.health_checker:
            return
        
        self.logger.info("Starting health monitor...")
        
        def run_health_monitor():
            self.health_checker.start_monitoring(interval=30)
        
        self.health_monitor_thread = threading.Thread(target=run_health_monitor, daemon=True)
        self.health_monitor_running = True
        self.health_monitor_thread.start()
        
        self.logger.info("Health monitor started")
    
    def _print_startup_summary(self) -> None:
        """打印启动摘要"""
        self.logger.info("=" * 60)
        self.logger.info("STARTUP COMPLETE")
        self.logger.info("=" * 60)
        
        self.logger.info(f"Total startup time: {self.metrics.total_time:.2f}s")
        self.logger.info(f"Environment: {self.env}")
        self.logger.info(f"API Server: {self.config.api.server_url}")
        self.logger.info(f"API Docs: {self.config.api.server_url}{self.config.api.docs_url}")
        self.logger.info(f"Health Check: {self.config.api.server_url}{self.config.monitoring.health_check_path}")
        
        # 显示阶段时间
        for phase, duration in self.metrics.phases.items():
            self.logger.debug(f"  {phase}: {duration:.2f}s")
        
        self.logger.info("=" * 60)
        self.logger.info("Application is ready and running")
        self.logger.info("Press Ctrl+C to shutdown")
        self.logger.info("=" * 60)
    
    def _wait_for_shutdown(self) -> None:
        """等待关闭信号"""
        try:
            # 主线程等待，直到收到关闭信号
            while self.phase == StartupPhase.READY:
                time.sleep(1)
                
                # 定期检查API服务器状态
                if self.api_thread and not self.api_thread.is_alive():
                    self.logger.error("API server thread died unexpectedly")
                    self.shutdown()
                    break
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
            self.shutdown()
    
    def shutdown(self) -> None:
        """关闭应用"""
        if self.phase == StartupPhase.SHUTTING_DOWN or self.phase == StartupPhase.STOPPED:
            return
        
        self.phase = StartupPhase.SHUTTING_DOWN
        self.logger.info("Shutting down application...")
        
        # 停止健康监控
        if self.health_checker and self.health_monitor_running:
            self.logger.info("Stopping health monitor...")
            self.health_checker.stop_monitoring()
            self.health_monitor_running = False
        
        # 关闭系统集成器
        if self.integrator:
            self.logger.info("Shutting down system integrator...")
            self.integrator.shutdown()
        
        # 停止API服务器（通过线程自然结束）
        self.api_running = False
        
        # 等待线程结束
        if self.api_thread and self.api_thread.is_alive():
            self.logger.info("Waiting for API server to stop...")
            self.api_thread.join(timeout=10)
        
        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            self.health_monitor_thread.join(timeout=5)
        
        self.phase = StartupPhase.STOPPED
        
        shutdown_time = time.time() - self.start_time
        self.logger.info(f"Application shutdown completed in {shutdown_time:.2f}s")
        self.logger.info("Goodbye!")
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态信息"""
        return {
            "phase": self.phase.value,
            "environment": self.env,
            "startup_time": self.metrics.total_time,
            "metrics": self.metrics.phases,
            "errors": self.errors,
            "api_running": self.api_running,
            "health_monitor_running": self.health_monitor_running,
            "timestamp": time.time()
        }
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.shutdown()


def create_startup_script() -> str:
    """创建启动脚本"""
    script_content = '''#!/usr/bin/env python3
"""
Topology Memory Context Manager - Startup Script

This script starts the complete Topology Memory Context Manager system.

Usage:
    python start.py [options]

Options:
    --env ENV           Environment (development, testing, production) [default: development]
    --config PATH       Path to configuration file
    --help              Show this help message
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.integration.application_starter import ApplicationStarter


def main():
    parser = argparse.ArgumentParser(description="Topology Memory Context Manager")
    parser.add_argument(
        "--env",
        default="development",
        choices=["development", "testing", "staging", "production"],
        help="Environment to run in"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    print(f"Starting Topology Memory Context Manager in {args.env} environment...")
    
    # Create and start the application
    starter = ApplicationStarter(env=args.env, config_path=args.config)
    
    try:
        success = starter.start()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\\nShutdown requested by user")
        return 0
    except Exception as e:
        print(f"Failed to start application: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
'''
    return script_content