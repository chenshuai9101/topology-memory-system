# 拓扑记忆上下文管理器 - 核心引擎

## 概述

拓扑记忆上下文管理器的核心引擎，提供上下文管理、记忆管理和拓扑算法功能。基于Python 3.11 + FastAPI + SQLAlchemy技术栈开发。

## 核心功能

### 1. 上下文管理
- `create_context`: 创建上下文
- `get_context`: 获取上下文
- `update_context`: 更新上下文
- `delete_context`: 删除上下文
- `list_contexts`: 列出上下文

### 2. 记忆节点管理
- `create_memory_node`: 创建记忆节点
- `get_memory_node`: 获取记忆节点
- `update_memory_node`: 更新记忆节点
- `delete_memory_node`: 删除记忆节点
- `link_nodes`: 链接记忆节点
- `search_memory`: 搜索记忆
- `get_related_nodes`: 获取相关节点

### 3. 拓扑算法
- `calculate_node_relationships`: 计算节点关联度
- `build_topology_graph`: 构建拓扑图
- `analyze_topology`: 分析拓扑结构
- `find_central_nodes`: 查找中心节点
- `find_shortest_path`: 查找最短路径
- `find_communities`: 查找社区

## 性能要求

- 所有操作响应时间 < 50ms
- 支持高并发访问
- 内存使用优化

## 目录结构

```
src/core/
├── __init__.py              # 核心模块导出
├── context_manager.py       # 上下文管理器
├── memory_manager.py        # 记忆管理器
├── topology_algorithms.py   # 拓扑算法
├── engine.py               # 核心引擎（集成所有组件）
├── performance_test.py     # 性能测试
├── test_core.py           # 单元测试
├── demo.py               # 演示脚本
└── README.md             # 本文档
```

## 快速开始

### 安装依赖

```bash
pip install numpy scikit-learn networkx
```

### 基本使用

```python
from core.engine import TopologyMemoryEngine, EngineConfig
from core.context_manager import ContextCreate

# 创建引擎
config = EngineConfig(
    max_contexts_per_session=100,
    max_memory_nodes=10000,
    vector_dimension=384
)
engine = TopologyMemoryEngine(config)

# 创建上下文
context_data = ContextCreate(
    session_id="session_1",
    user_id="user_1",
    context_type="conversation",
    content={"text": "Hello, world!"},
    priority=5
)
context = engine.create_context(context_data)

# 创建记忆节点
node_id = engine.create_memory_node(
    content="人工智能是计算机科学的一个分支",
    vector=[0.1] * 384,
    metadata={"category": "knowledge"}
)

# 搜索记忆
from api.schemas import TopologyQuery
query = TopologyQuery(query="人工智能", limit=10)
results = engine.search_memory(query)

# 构建拓扑
nodes, edges = engine.build_topology(node_id, max_nodes=20)

# 获取统计
stats = engine.get_stats()
```

### 运行演示

```bash
cd /workspace/topology-memory-context-manager
python -m src.core.demo
```

### 运行测试

```bash
cd /workspace/topology-memory-context-manager
python -m src.core.test_core
```

### 运行性能测试

```bash
cd /workspace/topology-memory-context-manager
python -c "
from src.core.engine import TopologyMemoryEngine
from src.core.performance_test import PerformanceTester

engine = TopologyMemoryEngine()
tester = PerformanceTester(engine)
results = tester.run_all_tests(num_iterations=50)
print(tester.generate_report())
"
```

## 配置选项

### EngineConfig 配置

```python
config = EngineConfig(
    # 上下文管理器配置
    max_contexts_per_session=100,      # 每个会话最大上下文数
    context_cleanup_interval=300,      # 上下文清理间隔（秒）
    
    # 记忆管理器配置
    max_memory_nodes=10000,            # 最大记忆节点数
    max_edges_per_node=50,             # 每个节点最大边数
    vector_dimension=384,              # 向量维度
    memory_similarity_threshold=0.5,   # 记忆相似度阈值
    
    # 拓扑算法配置
    topology_min_similarity=0.3,       # 拓扑最小相似度
    topology_max_edges_per_node=10,    # 拓扑最大边数
    enable_community_detection=True,   # 启用社区检测
    
    # 性能配置
    enable_caching=True,               # 启用缓存
    cache_ttl=300,                     # 缓存TTL（秒）
    batch_size=100                     # 批处理大小
)
```

## 性能指标

### 测试覆盖率
- 单元测试覆盖率 > 80%
- 集成测试覆盖所有核心功能

### 响应时间要求
- 上下文操作: < 20ms
- 记忆操作: < 30ms
- 拓扑操作: < 50ms
- 搜索操作: < 50ms

### 内存使用
- 上下文存储: O(n)
- 记忆存储: O(n + m) (n=节点数, m=边数)
- 向量索引: O(n * d) (d=向量维度)

## 算法说明

### 1. 相似度计算
- 向量相似度: 余弦相似度
- 文本相似度: Jaccard相似度
- 边相似度: 共同邻居比例

### 2. 拓扑构建
- 基于相似度的节点选择
- 权重优化的边构建
- 社区检测（Louvain算法）

### 3. 搜索算法
- 文本索引搜索
- 向量语义搜索
- 混合结果排序

## 扩展性

### 添加新的记忆类型
```python
# 在memory_manager.py中扩展MemoryNodeData类
class SpecializedMemoryNode(MemoryNodeData):
    def __init__(self, special_field, **kwargs):
        super().__init__(**kwargs)
        self.special_field = special_field
```

### 添加新的拓扑算法
```python
# 在topology_algorithms.py中添加新方法
def custom_topology_algorithm(self, graph, **params):
    # 实现自定义算法
    pass
```

### 集成外部向量服务
```python
# 扩展memory_manager.py中的向量处理
class VectorServiceMemoryManager(MemoryManager):
    def __init__(self, vector_service_url, **kwargs):
        super().__init__(**kwargs)
        self.vector_service_url = vector_service_url
    
    def _get_vector_embedding(self, text):
        # 调用外部向量服务
        response = requests.post(self.vector_service_url, json={"text": text})
        return response.json()["vector"]
```

## 故障排除

### 常见问题

1. **内存使用过高**
   - 减少`max_memory_nodes`配置
   - 启用缓存清理
   - 使用向量降维

2. **响应时间慢**
   - 检查向量维度是否过大
   - 优化搜索算法参数
   - 启用缓存

3. **搜索结果不准确**
   - 调整相似度阈值
   - 优化文本分词
   - 添加更多训练数据

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

engine = TopologyMemoryEngine()
# 现在会输出详细的调试信息
```

## 贡献指南

1. 遵循PEP 8代码规范
2. 为新功能添加单元测试
3. 确保性能要求得到满足
4. 更新相关文档

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 联系方式

如有问题或建议，请通过项目issue系统提交。