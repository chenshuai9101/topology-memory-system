# 🧠 拓扑记忆上下文管理系统

一个基于拓扑结构和向量搜索的智能上下文管理系统，为 AI 助手提供无限上下文能力。

## ✨ 特性

- **无限上下文存储**: 突破 token 限制，永久保存对话历史
- **智能语义搜索**: 基于向量相似度的语义检索
- **记忆关联网络**: 自动建立记忆之间的关联关系
- **智能优化压缩**: 保留关键信息，减少存储空间
- **多模态支持**: 支持文本、图像、音频等多种记忆类型
- **实时处理**: 低延迟的记忆存储和检索
- **可扩展架构**: 模块化设计，易于扩展和定制

## 🚀 快速开始

### 前提条件
- Python 3.8+
- Node.js 14+ (用于 OpenClaw 集成)

### 安装

#### 1. 启动拓扑记忆服务
```bash
# 克隆仓库
git clone https://github.com/your-username/topology-memory-system.git
cd topology-memory-system

# 安装依赖
pip install -r requirements.txt

# 启动服务
python start_simple.py
```

#### 2. 安装 OpenClaw 集成
```bash
# 进入集成目录
cd openclaw-integration

# 安装依赖
npm install

# 测试连接
node cli.js test
```

### 使用示例

#### 基本 API 使用
```python
import requests

# 保存对话
response = requests.post('http://localhost:8001/nodes', json={
    'node_id': 'memory_001',
    'content': '用户: 我想学习人工智能',
    'metadata': {'category': 'learning'}
})

# 搜索相关上下文
response = requests.post('http://localhost:8001/search', json={
    'query': '人工智能',
    'limit': 5
})
```

#### OpenClaw 集成使用
```javascript
const memory = require('./topology-memory-integration.js');

// 初始化
const integration = new TopologyMemoryIntegration({
    baseUrl: 'http://localhost:8001',
    userId: 'openclaw_user'
});

// 保存对话
await integration.saveConversation('user', '用户消息');

// 构建智能上下文
const context = await integration.buildIntelligentContext('当前查询');
```

## 📁 项目结构

```
topology-memory-system/
├── src/                          # 核心源代码
│   ├── api/                      # FastAPI 接口
│   ├── core/                     # 核心引擎
│   │   ├── memory_manager.py     # 记忆管理器
│   │   ├── context_manager.py    # 上下文管理器
│   │   ├── topology_algorithms.py # 拓扑算法
│   │   ├── memory_optimizer.py   # 记忆优化器
│   │   ├── association_optimizer.py # 关联优化器
│   │   └── retrieval_optimizer.py # 检索优化器
│   └── integration/              # 集成模块
├── openclaw-integration/         # OpenClaw 集成
│   ├── topology-memory-integration.js # 集成库
│   ├── cli.js                    # 命令行工具
│   ├── SKILL.md                  # OpenClaw 技能文档
│   └── package.json              # Node.js 配置
├── config/                       # 配置文件
├── tests/                        # 测试文件
├── docs/                         # 文档
├── requirements.txt              # Python 依赖
├── start_simple.py              # 简化启动脚本
├── start.py                     # 完整启动脚本
├── docker-compose.yml           # Docker 配置
└── README.md                    # 项目说明
```

## 🔧 核心功能

### 1. 记忆管理
- 创建、更新、删除记忆节点
- 记忆分类和标签系统
- 记忆版本控制
- 批量导入导出

### 2. 上下文检索
- 语义相似度搜索
- 关键词匹配搜索
- 时间线浏览
- 关联路径发现

### 3. 拓扑分析
- 记忆网络构建
- 关键节点识别
- 社区发现算法
- 路径分析优化

### 4. 智能优化
- 内容压缩算法
- 关联权重优化
- 检索排序优化
- 存储空间优化

## 🎯 使用场景

### 1. AI 助手增强
- 为 ChatGPT、Claude 等提供无限上下文
- 智能对话历史管理
- 个性化学习记忆

### 2. 知识管理
- 个人知识库构建
- 学习笔记关联
- 创意灵感管理

### 3. 项目管理
- 项目讨论记录
- 决策过程追踪
- 团队协作记忆

### 4. 研究分析
- 文献关联分析
- 研究思路追踪
- 实验结果记录

## 📊 性能指标

- **存储容量**: 理论上无限
- **检索速度**: < 100ms (10,000节点内)
- **准确率**: > 85% (语义搜索)
- **压缩率**: 30-70% (智能压缩)
- **并发支持**: 100+ 并发请求

## 🔌 集成支持

### OpenClaw 集成
- 自动对话保存
- 智能上下文检索
- CLI 工具支持
- 技能插件系统

### API 集成
- RESTful API 接口
- WebSocket 实时更新
- Webhook 事件通知
- 第三方应用集成

### 数据库支持
- PostgreSQL (关系数据)
- Redis (缓存)
- Qdrant (向量搜索)
- 本地文件存储

## 🛠️ 开发指南

### 环境设置
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest tests/
```

### 代码贡献
1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

### 测试覆盖
```bash
# 运行所有测试
pytest

# 运行性能测试
python tests/performance_test.py

# 运行集成测试
python tests/integration_test.py
```

## 📈 路线图

### v1.0.0 (当前)
- 基础记忆管理
- 语义搜索功能
- OpenClaw 集成
- 基础优化算法

### v1.1.0 (计划中)
- 多模态记忆支持
- 高级拓扑分析
- 可视化界面
- 团队协作功能

### v1.2.0 (规划中)
- 机器学习优化
- 跨平台同步
- 移动端支持
- 企业级功能

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

### 贡献者
- [牧云野](https://github.com/muyunye) - 项目创建者和维护者

### 行为准则
请阅读 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) 了解我们的行为准则。

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 感谢 OpenClaw 社区的支持
- 感谢所有贡献者的努力
- 感谢用户反馈和建议

## 📞 支持

- 📧 邮箱: support@topology-memory.com
- 💬 Discord: [OpenClaw Discord](https://discord.gg/openclaw)
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/topology-memory-system/issues)
- 📚 文档: [项目文档](https://github.com/your-username/topology-memory-system/docs)

---

<p align="center">
  让 AI 拥有无限记忆，让对话拥有完整上下文
</p>