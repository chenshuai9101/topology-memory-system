# 拓扑记忆数据库模块

## 概述

本模块提供拓扑记忆上下文管理器的完整数据库解决方案，包括数据模型、仓库模式、缓存服务和性能优化。

## 功能特性

- ✅ **完整的数据模型**: 上下文、记忆节点、关联关系
- ✅ **Redis缓存**: 热点数据缓存，查询结果缓存
- ✅ **连接池管理**: PostgreSQL和Redis连接池
- ✅ **数据迁移**: Alembic迁移脚本
- ✅ **性能优化**: 索引优化，查询优化
- ✅ **监控统计**: 系统统计，性能指标
- ✅ **健康检查**: 数据库健康状态监控
- ✅ **性能测试**: 并发测试，压力测试

## 目录结构

```
src/database/
├── config/                    # 配置模块
│   ├── database_config.py    # 数据库配置
│   └── database_manager.py   # 数据库连接管理器
├── models/                   # 数据模型
│   ├── contexts.py          # 上下文模型
│   ├── memory_nodes.py      # 记忆节点模型
│   ├── associations.py      # 关联关系模型
│   └── __init__.py
├── repositories/            # 数据仓库
│   ├── base_repository.py   # 基础仓库
│   ├── context_repository.py # 上下文仓库
│   ├── memory_node_repository.py # 记忆节点仓库
│   ├── association_repository.py # 关联仓库
│   └── __init__.py
├── services/               # 服务层
│   ├── redis_cache.py     # Redis缓存服务
│   ├── database_service.py # 数据库服务
│   └── __init__.py
├── migrations/            # 数据库迁移
│   ├── versions/         # 迁移脚本
│   ├── env.py           # Alembic环境配置
│   └── script.py.mako   # 迁移模板
├── init_database.py      # 数据库初始化脚本
├── performance_test.py   # 性能测试脚本
├── DATABASE_ARCHITECTURE.md # 架构文档
└── README.md            # 本文档
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install sqlalchemy alembic psycopg2-binary redis psutil

# 启动PostgreSQL和Redis
# 确保PostgreSQL和Redis服务正在运行
```

### 2. 数据库配置

创建 `.env` 文件:

```env
# PostgreSQL配置
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=topology_memory
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# 连接池配置
POSTGRES_POOL_SIZE=20
POSTGRES_MAX_OVERFLOW=40
REDIS_MAX_CONNECTIONS=50
```

### 3. 初始化数据库

```bash
# 初始化数据库表
python src/database/init_database.py init

# 测试数据库功能
python src/database/init_database.py test

# 重置数据库
python src/database/init_database.py reset
```

### 4. 运行性能测试

```bash
# 运行性能测试 (100并发，每个用户10个请求)
python src/database/performance_test.py --concurrent 100 --requests 10

# 自定义测试参数
python src/database/performance_test.py --concurrent 50 --requests 20 --workers 50
```

## 使用示例

### 1. 基本使用

```python
from database.services.database_service import db_service

# 创建上下文
context = db_service.create_context(
    session_id="session_001",
    user_id="user_001",
    context_type="conversation",
    content={"message": "Hello, world!"},
    priority=5,
    ttl=3600
)

# 获取上下文
retrieved = db_service.get_context(context["id"])

# 创建记忆节点
node = db_service.create_memory_node(
    node_type="concept",
    content="人工智能",
    summary="人工智能相关概念",
    tags=["AI", "technology"]
)

# 创建关联
association = db_service.create_association(
    source_id=node["id"],
    target_id=node["id"],
    relation_type="related_to",
    weight=0.8,
    description="自关联测试"
)

# 获取系统统计
stats = db_service.get_system_stats()

# 健康检查
health = db_service.health_check()
```

### 2. 高级功能

```python
# 获取会话的所有上下文
session_contexts = db_service.get_session_contexts("session_001")

# 搜索记忆节点
search_results = db_service.search_nodes("人工智能", limit=20)

# 获取重要节点
important_nodes = db_service.get_important_nodes(threshold=0.7, limit=50)

# 获取相关节点
related_nodes = db_service.get_related_nodes(node_id, limit=20)

# 清理旧数据
cleanup_stats = db_service.cleanup_old_data(days=30)

# 重建缓存
cache_stats = db_service.rebuild_cache()

# 发现关联模式
patterns = db_service.discover_patterns()
```

## 数据模型

### 上下文 (Context)

存储会话上下文信息，支持优先级、TTL和向量搜索。

```python
{
    "id": "uuid",
    "session_id": "string",
    "user_id": "string",
    "context_type": "conversation|memory|task|knowledge|other",
    "content": {"json": "data"},
    "priority": 1-10,
    "is_active": true,
    "ttl": 3600,
    "created_at": "datetime",
    "updated_at": "datetime"
}
```

### 记忆节点 (MemoryNode)

存储记忆节点，支持聚类、向量化和重要性评分。

```python
{
    "id": "uuid",
    "node_type": "concept|entity|event|relation|pattern|other",
    "content": "string",
    "summary": "string",
    "importance_score": 0.0-1.0,
    "stability_score": 0.0-1.0,
    "tags": ["tag1", "tag2"],
    "cluster_id": "uuid",
    "created_at": "datetime"
}
```

### 关联关系 (Association)

存储节点间的关联关系，支持权重、置信度和双向关联。

```python
{
    "id": "uuid",
    "source_node_id": "uuid",
    "target_node_id": "uuid",
    "relation_type": "related_to|similar_to|part_of|causes|precedes|other",
    "weight": 0.0-1.0,
    "confidence": 0.0-1.0,
    "bidirectional": false,
    "description": "string",
    "created_at": "datetime"
}
```

## 性能优化

### 索引优化

- 所有外键字段都有索引
- 常用查询字段有复合索引
- 部分索引减少索引大小
- 定期分析索引使用情况

### 缓存策略

- 热点数据自动缓存 (TTL: 5分钟)
- 查询结果缓存
- 最近访问列表缓存
- 缓存穿透保护

### 连接池

- PostgreSQL连接池: 20-40连接
- Redis连接池: 50连接
- 连接健康检查
- 连接泄漏检测

## 监控与维护

### 监控指标

```python
# 获取系统统计
stats = db_service.get_system_stats()

# 获取性能指标
metrics = db_service.get_performance_metrics(hours=24)

# 健康检查
health = db_service.health_check()
```

### 维护任务

```python
# 清理过期数据
db_service.cleanup_old_data(days=30)

# 重建缓存
db_service.rebuild_cache()

# 发现关联模式
db_service.discover_patterns()
```

## 迁移管理

### 生成迁移脚本

```bash
# 自动生成迁移脚本
alembic revision --autogenerate -m "description"

# 应用迁移
alembic upgrade head

# 回滚迁移
alembic downgrade -1
```

### 迁移文件位置

迁移脚本位于 `src/database/migrations/versions/`

## 性能测试

### 测试命令

```bash
# 基本测试
python src/database/performance_test.py

# 自定义测试
python src/database/performance_test.py --concurrent 100 --requests 10 --workers 100

# 指定输出文件
python src/database/performance_test.py --output my_test_report.json
```

### 测试报告

测试报告包含:
- 各操作性能指标
- 系统资源使用情况
- 成功率统计
- 是否达到性能目标

## 故障排除

### 常见问题

1. **数据库连接失败**
   - 检查PostgreSQL服务状态
   - 验证连接配置
   - 检查防火墙设置

2. **Redis连接失败**
   - 检查Redis服务状态
   - 验证Redis配置
   - 检查内存使用情况

3. **性能问题**
   - 检查索引使用情况
   - 分析慢查询日志
   - 调整连接池配置

### 调试工具

```python
# 启用SQL日志
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# 获取连接统计
from database.config.database_manager import db_manager
stats = db_manager.get_connection_stats()

# 获取缓存统计
from database.services.redis_cache import redis_cache
cache_stats = redis_cache.get_cache_stats()
```

## 部署建议

### 开发环境
- PostgreSQL单节点
- Redis单节点
- 本地连接池

### 生产环境
- PostgreSQL集群 (主从+读写分离)
- Redis集群 (分片+复制)
- 连接池监控
- 负载均衡

## 版本历史

### v1.0.0 (2026-03-31)
- 初始版本发布
- 完整的数据模型
- Redis缓存支持
- 性能测试框架
- 文档和示例

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue或联系维护者。

---

*最后更新: 2026-03-31*