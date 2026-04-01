# 向量搜索模块

拓扑记忆项目的向量搜索和语义搜索模块，基于Qdrant + Sentence Transformers + FastAPI技术栈。

## 🎯 功能特性

### 核心功能
1. **向量数据库集成**: Qdrant集群部署和配置
2. **语义搜索**: 基于Sentence Transformers的文本向量化
3. **混合搜索**: 向量搜索 + 关键词搜索融合
4. **相似度计算**: 余弦相似度、欧氏距离、点积
5. **搜索结果排序**: 相关性排序、时间衰减、重要性加权

### 性能指标
- **搜索响应时间**: < 100ms
- **搜索准确率**: > 90%
- **吞吐量**: > 1000 req/sec
- **并发支持**: 100+ 并发用户

## 🚀 快速开始

### 环境要求
- Python 3.8+
- Docker & Docker Compose (用于Qdrant)
- 4GB+ RAM (推荐8GB)

### 安装步骤

#### 1. 安装Python依赖
```bash
pip install qdrant-client sentence-transformers fastapi uvicorn numpy pydantic
```

#### 2. 启动Qdrant向量数据库
```bash
# 进入项目目录
cd /path/to/topology-memory-context-manager

# 启动Qdrant
./docker/qdrant/deploy_qdrant.sh
```

#### 3. 运行演示
```bash
# 运行演示脚本
python src/vector_search/demo.py
```

#### 4. 启动API服务
```bash
# 启动向量搜索API
uvicorn src.vector_search.api.vector_api:router --reload --host 0.0.0.0 --port 8001
```

## 📁 项目结构

```
src/vector_search/
├── __init__.py              # 模块导出
├── demo.py                  # 演示脚本
├── README.md               # 本文档
├── models/                 # 数据模型
│   ├── __init__.py
│   └── vector_models.py    # 向量搜索数据模型
├── services/               # 服务层
│   ├── __init__.py
│   ├── vector_encoder.py   # 向量编码器 (Sentence Transformers)
│   ├── qdrant_service.py   # Qdrant向量数据库服务
│   └── hybrid_search.py    # 混合搜索服务
├── api/                    # API层
│   ├── __init__.py
│   └── vector_api.py       # FastAPI路由
└── utils/                  # 工具函数
    ├── __init__.py
    ├── config.py           # 配置管理
    └── helpers.py          # 工具函数
```

## 🔧 配置说明

### 环境变量
复制示例配置文件：
```bash
cp .env.example .env
```

重要配置项：
```env
# Qdrant配置
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_API_KEY=

# 向量编码器配置
ENCODER_MODEL=paraphrase-multilingual-MiniLM-L12-v2
ENCODER_DEVICE=cpu

# 搜索配置
DEFAULT_SEARCH_LIMIT=10
DEFAULT_SEARCH_THRESHOLD=0.5
DEFAULT_VECTOR_WEIGHT=0.7
DEFAULT_KEYWORD_WEIGHT=0.3
```

### 预训练模型
支持以下Sentence Transformers模型：

| 模型类型 | 向量维度 | 语言支持 | 描述 |
|---------|---------|---------|------|
| multilingual_mini | 384 | 多语言 | 多语言小型模型，适合通用语义搜索 |
| multilingual_base | 768 | 多语言 | 多语言基础模型，提供更高的准确性 |
| chinese_optimized | 512 | 中文/英文 | 中文优化模型，适合中文语义搜索 |
| english_optimized | 384 | 英文 | 英文优化模型，适合英文语义搜索 |

## 📚 API文档

### 健康检查
```bash
GET /vector/health
```

### 语义搜索
```bash
POST /vector/semantic-search
Content-Type: application/json

{
  "query_text": "人工智能发展历史",
  "limit": 10,
  "threshold": 0.5
}
```

### 混合搜索
```bash
POST /vector/hybrid-search
Content-Type: application/json

{
  "query_text": "机器学习算法",
  "keywords": ["机器学习", "算法"],
  "limit": 10,
  "vector_weight": 0.7,
  "keyword_weight": 0.3
}
```

### 文本向量化
```bash
POST /vector/encode
Content-Type: application/json

{
  "text": "需要编码的文本"
}
```

### 完整API文档
启动服务后访问：http://localhost:8001/docs

## 🧪 测试

### 运行单元测试
```bash
# 运行向量搜索测试
pytest tests/vector_search/test_vector_search.py -v

# 运行所有测试
pytest tests/ -v
```

### 性能测试
```bash
# 运行性能测试
python tests/performance/test_vector_performance.py
```

## 🐳 Docker部署

### 开发环境
```bash
# 启动Qdrant
docker-compose -f docker/qdrant/docker-compose.qdrant.yml up -d qdrant

# 启动API服务
docker-compose up -d api
```

### 生产环境（集群模式）
```bash
# 启动Qdrant集群
./docker/qdrant/deploy_qdrant.sh

# 选择模式2（集群模式）
```

## 🔍 使用示例

### Python代码示例

```python
from src.vector_search import (
    create_hybrid_search_service,
    HybridSearchQuery
)

# 创建搜索服务
search_service = create_hybrid_search_service(
    qdrant_host="localhost",
    qdrant_port=6333,
    model_type="multilingual_mini"
)

# 执行混合搜索
query = HybridSearchQuery(
    query_text="人工智能应用场景",
    keywords=["人工智能", "应用"],
    limit=10
)

results = search_service.hybrid_search(query)

# 处理结果
for result in results.results:
    print(f"分数: {result.score:.3f}, 文本: {result.payload.get('text', '')[:50]}...")
```

### 命令行示例
```bash
# 编码文本
curl -X POST "http://localhost:8001/vector/encode?text=人工智能"

# 语义搜索
curl -X POST "http://localhost:8001/vector/semantic-search" \
  -H "Content-Type: application/json" \
  -d '{"query_text": "机器学习", "limit": 5}'
```

## 📊 性能优化

### 缓存策略
- 向量编码结果缓存
- 搜索结果缓存 (TTL: 5分钟)
- 热点数据预加载

### 索引优化
- HNSW索引配置优化
- 向量量化 (PQ/SQ)
- 段合并策略

### 查询优化
- 批量查询处理
- 异步IO操作
- 连接池管理

## 🔧 故障排除

### 常见问题

#### 1. Qdrant连接失败
```bash
# 检查Qdrant服务状态
curl http://localhost:6333/health

# 查看日志
docker logs topology-memory-qdrant
```

#### 2. 模型下载失败
```bash
# 设置代理（如果需要）
export HF_ENDPOINT=https://hf-mirror.com

# 手动下载模型
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
```

#### 3. 内存不足
- 减少批量处理大小
- 使用更小的模型
- 增加系统内存

### 调试模式
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 监控指标

### 关键指标
- 查询响应时间 (P50, P95, P99)
- 查询吞吐量 (QPS)
- 缓存命中率
- 内存使用率
- 错误率

### 监控工具
- Prometheus + Grafana
- Qdrant Dashboard (http://localhost:8080)
- 自定义性能监控

## 🔄 版本历史

### v0.1.0 (2026-03-31)
- 初始版本发布
- 基础向量搜索功能
- Qdrant集成
- Sentence Transformers集成
- FastAPI接口
- 混合搜索算法

## 📄 许可证

本项目采用MIT许可证。详情请查看LICENSE文件。

## 🙏 致谢

- [Qdrant](https://qdrant.tech/) - 向量搜索引擎
- [Sentence Transformers](https://www.sbert.net/) - 文本向量化模型
- [FastAPI](https://fastapi.tiangolo.com/) - Web框架

---

**开发团队**: 拓扑记忆项目组  
**版本**: v0.1.0  
**最后更新**: 2026-03-31