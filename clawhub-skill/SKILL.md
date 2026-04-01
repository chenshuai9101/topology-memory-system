# 拓扑记忆上下文管理器技能

为 OpenClaw 提供无限上下文能力的技能，突破 token 限制，实现永久记忆。

## 🎯 功能特性

- **无限上下文存储**: 永久保存所有对话历史
- **智能语义检索**: 基于向量相似度的上下文搜索
- **记忆关联网络**: 自动建立对话之间的关联
- **智能优化压缩**: 保留关键信息，减少存储空间
- **OpenClaw 深度集成**: 无缝集成到 OpenClaw 工作流

## 🚀 快速安装

### 方法一：ClawHub 安装
```bash
clawhub install topology-memory-context
```

### 方法二：手动安装
```bash
# 1. 下载技能
git clone https://github.com/muyunye/topology-memory-system.git
cd topology-memory-system/openclaw-integration

# 2. 安装依赖
npm install

# 3. 启动拓扑记忆服务
cd ..
python start_simple.py

# 4. 测试连接
node openclaw-integration/cli.js test
```

## ⚙️ 配置

### OpenClaw 配置
在 `~/.openclaw/openclaw.json` 中添加：

```json
{
  "skills": {
    "topology-memory-context": {
      "enabled": true,
      "baseUrl": "http://localhost:8001",
      "userId": "openclaw_user",
      "autoSave": true,
      "autoContext": true,
      "contextMaxLength": 2000
    }
  }
}
```

### 环境变量
```bash
export TOPOLOGY_MEMORY_URL="http://localhost:8001"
export TOPOLOGY_MEMORY_USER_ID="your_user_id"
```

## 📖 使用方法

### 命令行工具
```bash
# 测试连接
openclaw-topology-memory test

# 保存对话
openclaw-topology-memory save -r user -c "用户消息"

# 搜索上下文
openclaw-topology-memory search -q "关键词"

# 构建智能上下文
openclaw-topology-memory context -q "当前查询"

# 优化记忆存储
openclaw-topology-memory optimize
```

### 自动集成
启用后，OpenClaw 会自动：
1. 保存每条对话到拓扑记忆系统
2. 在回复前检索相关上下文
3. 构建智能上下文供模型使用

### 手动控制
```javascript
const memory = require('topology-memory-integration');

// 初始化
const integration = new memory.TopologyMemoryIntegration({
  baseUrl: 'http://localhost:8001',
  userId: 'your_user_id'
});

// 保存对话
await integration.saveConversation('user', '用户消息');

// 获取相关上下文
const context = await integration.buildIntelligentContext('当前查询');
```

## 🔧 高级功能

### 记忆优化
```bash
# 定期优化记忆存储
openclaw-topology-memory optimize

# 设置定时任务
crontab -e
# 添加：0 2 * * * cd /path/to/skill && node cli.js optimize
```

### 批量操作
```bash
# 批量导入对话历史
openclaw-topology-memory import -f conversations.json

# 导出对话历史
openclaw-topology-memory export -f backup.json
```

### 系统监控
```bash
# 查看系统状态
openclaw-topology-memory status

# 查看健康状态
curl http://localhost:8001/health
```

## 🎯 使用场景

### 1. 长期项目讨论
- 保存数月甚至数年的项目对话
- 随时检索相关决策和讨论
- 突破 token 限制，保持完整上下文

### 2. 学习记录管理
- 保存所有学习内容和答疑
- 建立知识点之间的关联
- 智能复习和检索

### 3. 创意积累
- 保存所有灵感和想法
- 自动关联相关创意
- 随时检索历史灵感

### 4. 技术问题解决
- 保存所有技术问题和解决方案
- 建立问题-解决方案关联
- 快速找到类似问题的解法

## 📊 性能指标

- **存储容量**: 理论上无限
- **检索速度**: < 100ms (10,000节点内)
- **准确率**: > 85% (语义搜索)
- **压缩率**: 30-70% (智能压缩)
- **并发支持**: 100+ 并发请求

## 🔌 集成支持

### 支持的平台
- ✅ OpenClaw (深度集成)
- ✅ ChatGPT API
- ✅ Claude API
- ✅ 自定义 AI 助手

### 数据库支持
- ✅ PostgreSQL
- ✅ Redis
- ✅ Qdrant (向量搜索)
- ✅ 本地文件存储

## 🛠️ 故障排除

### 常见问题

#### 1. 连接失败
```
错误: 无法连接到拓扑记忆系统
```
**解决方案**:
- 检查拓扑记忆服务是否运行: `curl http://localhost:8001/health`
- 检查网络连接
- 确认 baseUrl 配置正确

#### 2. 保存失败
```
错误: 保存对话失败
```
**解决方案**:
- 检查拓扑记忆API是否正常
- 确认节点ID没有重复
- 检查网络连接

#### 3. 搜索无结果
```
搜索到 0 个结果
```
**解决方案**:
- 确认已有对话数据
- 尝试不同的搜索关键词
- 检查搜索查询格式

#### 4. 上下文过长
```
上下文已截断
```
**解决方案**:
- 调整 `contextMaxLength` 参数
- 减少相关历史数量
- 启用记忆优化

### 日志查看
```bash
# 查看拓扑记忆服务日志
tail -f topology-memory.log

# 查看 OpenClaw 日志
openclaw logs
```

## 📈 性能优化

### 内存优化
- 定期运行优化命令
- 设置合理的上下文长度限制
- 启用自动清理旧对话

### 响应时间优化
- 使用缓存减少API调用
- 并行处理搜索和保存
- 设置合理的超时时间

### 存储优化
- 启用记忆压缩
- 定期归档旧对话
- 使用分层存储策略

## 🔄 更新日志

### v1.0.0 (2026-04-02)
- 🎉 初始版本发布
- ✅ 无限上下文存储
- ✅ 智能语义检索
- ✅ OpenClaw 深度集成
- ✅ 命令行工具支持

### v1.1.0 (计划中)
- 🔄 多模态记忆支持
- 📊 可视化分析界面
- 🤝 团队协作功能
- 🚀 性能优化改进

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

### 贡献方式
1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

### 行为准则
请阅读 [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) 了解我们的行为准则。

## 📄 许可证

MIT License - 查看 [LICENSE](../LICENSE) 文件了解详情。

## 📞 支持

- 📧 邮箱: muyunye@example.com
- 💬 Discord: [OpenClaw Discord](https://discord.gg/openclaw)
- 🐛 Issues: [GitHub Issues](https://github.com/muyunye/topology-memory-system/issues)
- 📚 文档: [项目文档](https://github.com/muyunye/topology-memory-system/docs)

---

<p align="center">
  让 AI 拥有无限记忆，让对话拥有完整上下文
</p>