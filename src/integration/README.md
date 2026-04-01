# 拓扑记忆上下文管理器 - 系统集成模块

## 概述

系统集成模块提供了完整的拓扑记忆上下文管理器系统的集成、配置管理、服务容器、健康检查和启动脚本。

## 核心组件

### 1. 配置管理器 (ConfigManager)

统一配置管理系统，支持：
- 环境变量覆盖
- 配置文件管理（YAML/JSON）
- 多环境支持（开发/测试/生产）
- 配置验证

### 2. 服务容器 (ServiceContainer)

依赖注入容器，支持：
- 服务注册（单例/瞬时/作用域）
- 自动依赖解析
- 服务生命周期管理
- 健康检查集成

### 3. 系统集成器 (SystemIntegrator)

核心集成组件，提供：
- 系统初始化流程
- 组件连接管理（数据库/Redis/向量搜索/ML服务）
- 状态监控和报告
- 错误处理和恢复

### 4. 健康检查器 (HealthChecker)

全系统健康检查，包括：
- 组件健康状态监控
- 定期健康检查
- 严重性分级（关键/高/中/低）
- 状态报告和告警

### 5. 应用启动器 (ApplicationStarter)

完整系统启动和初始化：
- 分阶段启动流程
- 启动时间监控
- 优雅关闭
- 命令行接口

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动系统

```bash
# 开发环境
python start.py --env development

# 生产环境
python start.py --env production --config /path/to/config.yaml
```

### 使用API

```bash
# API文档
http://localhost:8000/docs

# 健康检查
http://localhost:8000/health

# 就绪检查
http://localhost:8000/ready

# 存活检查
http://localhost:8000/live
```

## 配置系统

### 配置文件结构

```
config/
├── default.yaml      # 默认配置（所有环境继承）
├── development.yaml  # 开发环境配置
├── testing.yaml     # 测试环境配置
└── production.yaml  # 生产环境配置
```

### 环境变量覆盖

支持通过环境变量覆盖配置：

```bash
export DB_HOST=postgres.example.com
export DB_PORT=5432
export API_PORT=8080
export REDIS_PASSWORD=secret
```

### 配置优先级

1. 环境变量（最高优先级）
2. 环境特定配置文件
3. 默认配置文件
4. 代码默认值（最低优先级）

## 服务容器使用

### 注册服务

```python
from src.integration.service_container import get_service_container

# 获取服务容器
container = get_service_container()

# 注册单例服务
class DatabaseService:
    def __init__(self, config):
        self.config = config

container.register_singleton(DatabaseService)

# 注册瞬时服务
class RequestProcessor:
    def process(self, data):
        return processed_data

container.register_transient(RequestProcessor)
```

### 获取服务

```python
# 获取服务实例
db_service = container.get(DatabaseService)
processor = container.get(RequestProcessor)

# 使用服务
result = processor.process(data)
```

### 依赖注入

```python
class UserService:
    def __init__(self, db_service: DatabaseService, cache_service: CacheService):
        self.db = db_service
        self.cache = cache_service
    
    def get_user(self, user_id):
        # 使用注入的服务
        return self.db.get_user(user_id)

# 容器会自动解析依赖
user_service = container.create_with_dependencies(UserService)
```

## 健康检查系统

### 内置检查

系统包含以下健康检查：

1. **配置检查** - 验证配置有效性
2. **数据库检查** - 检查数据库连接和查询
3. **Redis检查** - 检查Redis连接和读写
4. **向量搜索检查** - 检查向量搜索服务
5. **ML服务检查** - 检查机器学习模型
6. **API服务检查** - 检查API端点响应
7. **内存使用检查** - 监控内存使用情况
8. **磁盘空间检查** - 检查磁盘空间
9. **系统负载检查** - 监控系统负载

### 自定义检查

```python
from src.integration.health_checker import get_health_checker, HealthCheckResult, HealthStatus, CheckSeverity

def check_external_api():
    import requests
    
    start_time = time.time()
    try:
        response = requests.get("https://api.example.com/health", timeout=5)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            return HealthCheckResult(
                name="external_api",
                status=HealthStatus.HEALTHY,
                severity=CheckSeverity.MEDIUM,
                message="External API is accessible",
                duration=duration,
                timestamp=time.time()
            )
        else:
            return HealthCheckResult(
                name="external_api",
                status=HealthStatus.UNHEALTHY,
                severity=CheckSeverity.MEDIUM,
                message=f"External API returned {response.status_code}",
                duration=duration,
                timestamp=time.time()
            )
    except Exception as e:
        duration = time.time() - start_time
        return HealthCheckResult(
            name="external_api",
            status=HealthStatus.UNHEALTHY,
            severity=CheckSeverity.MEDIUM,
            message=f"External API check failed: {str(e)}",
            duration=duration,
            timestamp=time.time()
        )

# 注册自定义检查
checker = get_health_checker()
checker.register_check(
    name="external_api",
    check_func=check_external_api,
    interval=60,
    severity=CheckSeverity.MEDIUM
)
```

### 健康监控

```python
# 启动健康监控（每30秒检查一次）
checker.start_monitoring(interval=30)

# 获取状态报告
report = checker.get_status_report()
print(f"Overall status: {report['overall_status']}")
print(f"Healthy checks: {report['summary']['healthy_checks']}/{report['summary']['total_checks']}")

# 停止监控
checker.stop_monitoring()
```

## 系统集成

### 初始化系统

```python
from src.integration.system_integrator import get_system_integrator

# 初始化系统
integrator = get_system_integrator("development")

# 获取系统状态
status = integrator.get_status_report()
print(f"System status: {status['status']}")
print(f"Startup time: {status['startup_time']:.2f}s")

# 获取服务
config = integrator.get_config()
container = integrator.get_container()

# 启动API服务器
integrator.start_api_server()
```

### 使用上下文管理器

```python
from src.integration.system_integrator import SystemIntegrator

with SystemIntegrator("development") as integrator:
    # 系统已初始化
    config = integrator.get_config()
    container = integrator.get_container()
    
    # 使用系统服务
    service = container.get(SomeService)
    result = service.do_something()
    
    # 系统会在退出上下文时自动关闭
```

## 启动脚本

### 命令行参数

```bash
# 显示帮助
python start.py --help

# 开发环境
python start.py --env development

# 生产环境（使用自定义配置）
python start.py --env production --config /etc/topology-memory/config.yaml

# 测试环境
python start.py --env testing
```

### 启动流程

1. **初始化配置** - 加载和验证配置
2. **初始化服务** - 注册和初始化所有服务
3. **数据库迁移** - 运行数据库迁移（如果配置了）
4. **启动API服务器** - 启动FastAPI服务器
5. **启动健康监控** - 开始定期健康检查
6. **等待关闭信号** - 保持运行直到收到关闭信号

### 优雅关闭

系统支持优雅关闭：
- 处理SIGINT和SIGTERM信号
- 停止健康监控
- 关闭系统集成器
- 等待API服务器停止

## 测试

### 运行集成测试

```bash
# 运行所有集成测试
pytest src/integration/test_integration.py -v

# 运行特定测试类
pytest src/integration/test_integration.py::TestConfigManager -v

# 运行单个测试
pytest src/integration/test_integration.py::TestConfigManager::test_config_manager_initialization -v
```

### 测试覆盖率

```bash
# 生成测试覆盖率报告
pytest src/integration/test_integration.py --cov=src.integration --cov-report=html
```

## 性能指标

### 启动时间目标
- 总启动时间: < 30秒
- 配置加载: < 1秒
- 服务初始化: < 5秒
- 数据库连接: < 3秒
- API服务器启动: < 10秒

### 健康检查性能
- 单个检查: < 10秒（可配置）
- 所有检查: < 30秒
- 监控间隔: 30秒（可配置）

## 故障排除

### 常见问题

1. **配置加载失败**
   - 检查配置文件格式（YAML/JSON）
   - 验证环境变量名称
   - 检查文件权限

2. **服务初始化失败**
   - 检查依赖服务（数据库/Redis）是否运行
   - 验证连接字符串和凭据
   - 检查网络连接

3. **健康检查失败**
   - 查看健康检查详情
   - 检查服务日志
   - 验证服务配置

4. **API服务器无法启动**
   - 检查端口是否被占用
   - 验证主机绑定配置
   - 查看API服务器日志

### 日志查看

```bash
# 查看应用日志
tail -f /var/log/topology-memory/app.log

# 查看系统日志
journalctl -u topology-memory -f

# 查看健康检查日志
grep "health_check" /var/log/topology-memory/app.log
```

## 扩展开发

### 添加新服务

1. 创建服务类
2. 在服务容器中注册
3. 添加健康检查（如果需要）
4. 更新配置（如果需要）

### 添加新配置

1. 更新配置管理器类
2. 添加配置数据类
3. 更新配置文件模板
4. 添加配置验证

### 添加新健康检查

1. 创建检查函数
2. 在健康检查器中注册
3. 配置检查间隔和严重性
4. 测试检查功能

## 部署指南

### 开发环境

```bash
# 克隆代码
git clone <repository-url>
cd topology-memory-context-manager

# 安装依赖
pip install -r requirements.txt

# 启动服务
python start.py --env development
```

### 生产环境

```bash
# 使用系统服务
sudo cp systemd/topology-memory.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable topology-memory
sudo systemctl start topology-memory

# 使用Docker
docker build -t topology-memory .
docker run -d -p 8000:8000 --name topology-memory topology-memory

# 使用Docker Compose
docker-compose up -d
```

### 监控和告警

1. **健康检查端点** - 集成到监控系统
2. **指标端点** - Prometheus指标收集
3. **日志聚合** - ELK栈或类似方案
4. **告警规则** - 基于健康状态和指标

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 支持

如有问题或建议，请：
1. 查看文档和FAQ
2. 提交GitHub Issue
3. 联系开发团队