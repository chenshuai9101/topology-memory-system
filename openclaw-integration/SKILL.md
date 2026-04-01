# 拓扑记忆集成技能

让 OpenClaw 使用拓扑记忆系统作为智能上下文管理器，突破 token 限制，实现无限上下文。

## 功能

- **自动保存对话**: 每次对话自动保存到拓扑记忆系统
- **智能上下文检索**: 根据当前查询检索相关历史
- **无限上下文**: 突破 token 限制，存储无限对话历史
- **记忆关联**: 自动建立对话之间的关联
- **记忆优化**: 定期优化记忆存储

## 安装

### 前提条件
1. 拓扑记忆系统正在运行 (http://localhost:8001)
2. Node.js 环境

### 安装步骤
```bash
# 1. 进入 OpenClaw 技能目录
cd ~/.openclaw/workspace/skills

# 2. 创建技能目录
mkdir -p topology-memory-integration

# 3. 复制集成文件
cp /path/to/topology-memory-integration.js topology-memory-integration/

# 4. 安装依赖
cd topology-memory-integration
npm init -y
npm install axios
```

## 配置

### OpenClaw 配置
在 `~/.openclaw/openclaw.json` 中添加:

```json
{
  "skills": {
    "topology-memory-integration": {
      "enabled": true,
      "baseUrl": "http://localhost:8001",
      "userId": "openclaw_user",
      "autoSave": true,
      "autoContext": true
    }
  }
}
```

### 环境变量
```bash
export TOPOLOGY_MEMORY_URL="http://localhost:8001"
export TOPOLOGY_MEMORY_USER_ID="openclaw_user"
```

## 使用方法

### 命令行使用
```bash
# 测试连接
openclaw topology-memory test

# 保存对话
openclaw topology-memory save --role user --content "用户消息"

# 搜索上下文
openclaw topology-memory search --query "关键词"

# 获取对话历史
openclaw topology-memory history

# 优化记忆
openclaw topology-memory optimize
```

### 在对话中自动使用
技能启用后，OpenClaw 会自动:
1. 保存每条对话到拓扑记忆系统
2. 在回复前检索相关上下文
3. 构建智能上下文供模型使用

### 手动控制
```javascript
// 在 OpenClaw 脚本中使用
const memory = require('./skills/topology-memory-integration/topology-memory-integration.js');
const integration = new memory.TopologyMemoryIntegration();

// 保存当前对话
await integration.saveConversation('user', message);

// 获取相关上下文
const context = await integration.buildIntelligentContext(message);

// 将上下文添加到提示中
const enhancedPrompt = context + '\n\n' + originalPrompt;
```

## API 参考

### TopologyMemoryIntegration 类

#### 构造函数
```javascript
const integration = new TopologyMemoryIntegration({
  baseUrl: 'http://localhost:8001',  // 拓扑记忆API地址
  userId: 'openclaw_user',           // 用户ID
  sessionId: 'session_123'           // 会话ID（可选，自动生成）
});
```

#### 方法

##### saveConversation(role, content, metadata)
保存对话到拓扑记忆系统
- `role`: 'user' 或 'assistant'
- `content`: 对话内容
- `metadata`: 额外元数据（可选）

##### searchContext(query, limit)
搜索相关上下文
- `query`: 搜索查询
- `limit`: 返回结果数量（默认5）

##### buildIntelligentContext(currentQuery, options)
构建智能上下文
- `currentQuery`: 当前查询
- `options`: 配置选项
  - `includeRelevant`: 是否包含相关历史（默认true）
  - `includeRecent`: 是否包含最近对话（默认true）
  - `maxLength`: 最大长度（默认2000）
  - `relevantLimit`: 相关历史数量（默认3）
  - `recentLimit`: 最近对话数量（默认2）

##### getConversationHistory(limit)
获取对话历史
- `limit`: 返回最近多少条（默认10）

##### optimizeMemories()
优化记忆存储

##### getSystemStatus()
获取系统状态

##### testConnection()
测试连接

## 示例

### 示例1: 完整对话流程
```javascript
const TopologyMemoryIntegration = require('./topology-memory-integration.js');

async function conversationExample() {
  const memory = new TopologyMemoryIntegration({
    userId: 'user_001',
    sessionId: 'conversation_001'
  });
  
  // 检查连接
  const connected = await memory.testConnection();
  if (!connected) {
    console.error('无法连接到拓扑记忆系统');
    return;
  }
  
  // 模拟对话
  const userMessage = '我想学习Python编程';
  await memory.saveConversation('user', userMessage);
  
  // 构建上下文
  const context = await memory.buildIntelligentContext(userMessage);
  
  // 生成回复（这里模拟）
  const assistantReply = 'Python是一种高级编程语言，适合初学者学习。';
  await memory.saveConversation('assistant', assistantReply);
  
  console.log('对话已保存到拓扑记忆系统');
  console.log('构建的上下文:', context.substring(0, 200) + '...');
}
```

### 示例2: 集成到 OpenClaw 响应处理
```javascript
// 在 OpenClaw 的响应处理器中
module.exports = {
  name: 'topology-memory-handler',
  
  async handleMessage(message, context) {
    const memory = new TopologyMemoryIntegration({
      userId: context.userId,
      sessionId: context.sessionId
    });
    
    // 保存用户消息
    await memory.saveConversation('user', message.content);
    
    // 构建智能上下文
    const intelligentContext = await memory.buildIntelligentContext(
      message.content,
      { maxLength: 1500 }
    );
    
    // 将上下文添加到提示中
    const enhancedContext = {
      ...context,
      systemPrompt: intelligentContext + '\n\n' + context.systemPrompt
    };
    
    // 返回增强后的上下文
    return enhancedContext;
  },
  
  async handleResponse(response, context) {
    const memory = new TopologyMemoryIntegration({
      userId: context.userId,
      sessionId: context.sessionId
    });
    
    // 保存助手回复
    await memory.saveConversation('assistant', response.content);
    
    return response;
  }
};
```

## 配置选项

### 技能配置
| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| enabled | boolean | true | 是否启用技能 |
| baseUrl | string | http://localhost:8001 | 拓扑记忆API地址 |
| userId | string | openclaw_user | 用户ID |
| autoSave | boolean | true | 是否自动保存对话 |
| autoContext | boolean | true | 是否自动构建上下文 |
| contextMaxLength | number | 2000 | 上下文最大长度 |
| relevantLimit | number | 3 | 相关历史数量 |
| recentLimit | number | 2 | 最近对话数量 |

### 构建上下文选项
| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| includeRelevant | boolean | true | 包含相关历史 |
| includeRecent | boolean | true | 包含最近对话 |
| maxLength | number | 2000 | 最大长度 |
| relevantLimit | number | 3 | 相关历史数量 |
| recentLimit | number | 2 | 最近对话数量 |

## 故障排除

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
- 调整 `maxLength` 参数
- 减少 `relevantLimit` 或 `recentLimit`
- 启用记忆优化

### 日志查看
```bash
# 查看拓扑记忆服务日志
cd /path/to/topology-memory-context-manager
tail -f uvicorn.log

# 查看 OpenClaw 日志
openclaw logs
```

## 性能优化

### 内存优化
- 定期运行 `optimizeMemories()` 优化存储
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

## 更新日志

### v1.0.0 (2026-04-02)
- 初始版本发布
- 支持对话保存和检索
- 支持智能上下文构建
- 基础集成功能

## 支持

- 文档: [拓扑记忆系统文档](https://github.com/your-repo/docs)
- 问题: [GitHub Issues](https://github.com/your-repo/issues)
- 讨论: [OpenClaw Discord](https://discord.gg/openclaw)

## 许可证

MIT License