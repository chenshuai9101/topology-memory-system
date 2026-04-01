/**
 * OpenClaw 拓扑记忆集成脚本
 * 让 OpenClaw 能够使用拓扑记忆系统作为上下文存储
 */

const axios = require('axios');

class TopologyMemoryIntegration {
  constructor(config = {}) {
    this.baseUrl = config.baseUrl || 'http://localhost:8001';
    this.userId = config.userId || 'openclaw_user';
    this.sessionId = config.sessionId || `session_${Date.now()}`;
    this.conversationIndex = 0;
    
    // 初始化 axios 实例
    this.api = axios.create({
      baseURL: this.baseUrl,
      timeout: 5000,
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    console.log(`拓扑记忆集成初始化: ${this.baseUrl}`);
    console.log(`用户: ${this.userId}, 会话: ${this.sessionId}`);
  }
  
  /**
   * 保存对话到拓扑记忆系统
   * @param {string} role - 角色 (user/assistant)
   * @param {string} content - 对话内容
   * @param {object} metadata - 额外元数据
   * @returns {Promise<object>} 保存结果
   */
  async saveConversation(role, content, metadata = {}) {
    try {
      const nodeId = `claw_${this.sessionId}_${this.conversationIndex.toString().padStart(4, '0')}`;
      this.conversationIndex++;
      
      const memoryNode = {
        node_id: nodeId,
        content: `${role}: ${content}`,
        metadata: {
          session_id: this.sessionId,
          user_id: this.userId,
          role: role,
          timestamp: new Date().toISOString(),
          conversation_index: this.conversationIndex,
          category: 'openclaw_conversation',
          source: 'openclaw',
          ...metadata
        }
      };
      
      const response = await this.api.post('/nodes', memoryNode);
      
      // 如果是第二条及以后的记录，建立与前一条的关联
      if (this.conversationIndex > 1) {
        const prevNodeId = `claw_${this.sessionId}_${(this.conversationIndex - 2).toString().padStart(4, '0')}`;
        await this.createAssociation(prevNodeId, nodeId, 'follows', 0.9);
      }
      
      console.log(`✅ 对话已保存: [${role}] ${content.substring(0, 50)}...`);
      return {
        success: true,
        nodeId: nodeId,
        message: '对话已保存到拓扑记忆系统'
      };
      
    } catch (error) {
      console.error('❌ 保存对话失败:', error.message);
      return {
        success: false,
        error: error.message
      };
    }
  }
  
  /**
   * 创建记忆关联
   * @param {string} sourceId - 源节点ID
   * @param {string} targetId - 目标节点ID
   * @param {string} relationship - 关系类型
   * @param {number} weight - 关联权重
   * @returns {Promise<object>} 创建结果
   */
  async createAssociation(sourceId, targetId, relationship = 'related', weight = 1.0) {
    try {
      const edge = {
        source_id: sourceId,
        target_id: targetId,
        relationship: relationship,
        weight: weight,
        metadata: {
          created_at: new Date().toISOString(),
          source: 'openclaw_integration'
        }
      };
      
      await this.api.post('/edges', edge);
      console.log(`✅ 关联已创建: ${sourceId} -> ${targetId} (${relationship})`);
      
      return { success: true };
      
    } catch (error) {
      console.error('❌ 创建关联失败:', error.message);
      return { success: false, error: error.message };
    }
  }
  
  /**
   * 搜索相关上下文
   * @param {string} query - 搜索查询
   * @param {number} limit - 返回结果数量
   * @returns {Promise<Array>} 相关上下文列表
   */
  async searchContext(query, limit = 5) {
    try {
      const response = await this.api.post('/search', {
        query: query,
        limit: limit
      });
      
      const results = response.data.results || [];
      console.log(`🔍 搜索 "${query}": 找到 ${results.length} 个相关上下文`);
      
      // 格式化结果
      return results.map(result => ({
        content: result.content,
        relevance: result.relevance_score || 0,
        nodeId: result.node_id,
        metadata: result.metadata || {}
      }));
      
    } catch (error) {
      console.error('❌ 搜索上下文失败:', error.message);
      return [];
    }
  }
  
  /**
   * 获取对话历史
   * @param {number} limit - 返回最近多少条
   * @returns {Promise<Array>} 对话历史
   */
  async getConversationHistory(limit = 10) {
    try {
      const response = await this.api.get('/nodes');
      const allNodes = response.data.nodes || [];
      
      // 过滤出当前会话的对话，按时间排序
      const sessionNodes = allNodes
        .filter(node => node.metadata?.session_id === this.sessionId)
        .sort((a, b) => {
          const timeA = new Date(a.metadata?.timestamp || 0);
          const timeB = new Date(b.metadata?.timestamp || 0);
          return timeB - timeA; // 最新的在前
        })
        .slice(0, limit);
      
      console.log(`📜 获取到 ${sessionNodes.length} 条会话历史`);
      
      return sessionNodes.map(node => ({
        role: node.metadata?.role || 'unknown',
        content: node.content.replace(/^(user|assistant):\s*/, ''),
        timestamp: node.metadata?.timestamp,
        nodeId: node.node_id
      }));
      
    } catch (error) {
      console.error('❌ 获取对话历史失败:', error.message);
      return [];
    }
  }
  
  /**
   * 构建智能上下文
   * @param {string} currentQuery - 当前查询
   * @param {object} options - 选项
   * @returns {Promise<string>} 构建的上下文
   */
  async buildIntelligentContext(currentQuery, options = {}) {
    const {
      includeRelevant = true,
      includeRecent = true,
      maxLength = 2000,
      relevantLimit = 3,
      recentLimit = 2
    } = options;
    
    const contextParts = [];
    
    // 1. 添加相关历史
    if (includeRelevant) {
      const relevantMemories = await this.searchContext(currentQuery, relevantLimit);
      if (relevantMemories.length > 0) {
        contextParts.push('## 相关历史对话');
        relevantMemories.forEach(memory => {
          contextParts.push(`- ${memory.content}`);
        });
      }
    }
    
    // 2. 添加最近对话
    if (includeRecent) {
      const recentHistory = await this.getConversationHistory(recentLimit);
      if (recentHistory.length > 0) {
        contextParts.push('\n## 最近对话');
        recentHistory.forEach(item => {
          contextParts.push(`- ${item.role}: ${item.content.substring(0, 100)}...`);
        });
      }
    }
    
    // 3. 组合上下文
    let fullContext = contextParts.join('\n');
    
    // 4. 限制长度
    if (fullContext.length > maxLength) {
      fullContext = fullContext.substring(0, maxLength) + '...\n[上下文已截断]';
    }
    
    console.log(`🧠 构建智能上下文: ${fullContext.length} 字符`);
    return fullContext;
  }
  
  /**
   * 优化记忆存储
   * @returns {Promise<object>} 优化结果
   */
  async optimizeMemories() {
    try {
      const response = await this.api.post('/optimize');
      console.log(`🔄 记忆优化: ${response.data.message}`);
      return response.data;
    } catch (error) {
      console.error('❌ 记忆优化失败:', error.message);
      return { success: false, error: error.message };
    }
  }
  
  /**
   * 获取系统状态
   * @returns {Promise<object>} 系统状态
   */
  async getSystemStatus() {
    try {
      const response = await this.api.get('/status');
      return response.data;
    } catch (error) {
      console.error('❌ 获取系统状态失败:', error.message);
      return { status: 'error', error: error.message };
    }
  }
  
  /**
   * 测试连接
   * @returns {Promise<boolean>} 连接是否成功
   */
  async testConnection() {
    try {
      const response = await this.api.get('/health');
      const isHealthy = response.data.status === 'healthy';
      console.log(isHealthy ? '✅ 拓扑记忆系统连接正常' : '❌ 拓扑记忆系统状态异常');
      return isHealthy;
    } catch (error) {
      console.error('❌ 无法连接到拓扑记忆系统:', error.message);
      return false;
    }
  }
}

// 导出模块
if (typeof module !== 'undefined' && module.exports) {
  module.exports = TopologyMemoryIntegration;
}

// 如果直接运行，执行测试
if (require.main === module) {
  (async () => {
    console.log('='.repeat(60));
    console.log('OpenClaw 拓扑记忆集成测试');
    console.log('='.repeat(60));
    
    const integration = new TopologyMemoryIntegration({
      userId: 'test_user',
      sessionId: 'test_session_' + Date.now()
    });
    
    // 测试连接
    const connected = await integration.testConnection();
    if (!connected) {
      console.log('❌ 测试失败: 无法连接到拓扑记忆系统');
      console.log('   请确保拓扑记忆服务正在运行: http://localhost:8001');
      process.exit(1);
    }
    
    // 测试保存对话
    console.log('\n1. 测试保存对话...');
    await integration.saveConversation('user', '你好，我想了解人工智能的基本概念');
    await integration.saveConversation('assistant', '你好！人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。');
    await integration.saveConversation('user', '那机器学习和人工智能有什么区别？');
    
    // 测试搜索上下文
    console.log('\n2. 测试搜索上下文...');
    const searchResults = await integration.searchContext('人工智能', 3);
    console.log(`   搜索到 ${searchResults.length} 个结果`);
    
    // 测试构建智能上下文
    console.log('\n3. 测试构建智能上下文...');
    const context = await integration.buildIntelligentContext('机器学习', {
      includeRelevant: true,
      includeRecent: true,
      maxLength: 1000
    });
    console.log(`   构建的上下文:\n${context.substring(0, 200)}...`);
    
    // 测试获取系统状态
    console.log('\n4. 测试获取系统状态...');
    const status = await integration.getSystemStatus();
    console.log(`   系统状态: ${status.status}`);
    console.log(`   记忆节点: ${status.nodes_count} 个`);
    console.log(`   记忆关联: ${status.edges_count} 条`);
    
    console.log('\n' + '='.repeat(60));
    console.log('✅ 集成测试完成！');
    console.log('='.repeat(60));
    
    console.log('\n📋 使用示例:');
    console.log(`
// 初始化
const TopologyMemoryIntegration = require('./topology-memory-integration.js');
const memory = new TopologyMemoryIntegration({
  baseUrl: 'http://localhost:8001',
  userId: 'your_user_id'
});

// 保存对话
await memory.saveConversation('user', '用户消息');
await memory.saveConversation('assistant', '助手回复');

// 获取上下文
const context = await memory.buildIntelligentContext('当前查询');

// 搜索相关记忆
const results = await memory.searchContext('关键词');
    `);
    
  })().catch(console.error);
}