#!/usr/bin/env node
/**
 * OpenClaw 拓扑记忆集成 CLI 工具
 */

const { program } = require('commander');
const TopologyMemoryIntegration = require('./topology-memory-integration.js');

// 配置
const config = {
  baseUrl: process.env.TOPOLOGY_MEMORY_URL || 'http://localhost:8001',
  userId: process.env.TOPOLOGY_MEMORY_USER_ID || 'openclaw_user'
};

program
  .name('openclaw-topology-memory')
  .description('OpenClaw 拓扑记忆集成工具')
  .version('1.0.0');

// 测试连接命令
program
  .command('test')
  .description('测试拓扑记忆系统连接')
  .action(async () => {
    console.log('🔗 测试拓扑记忆系统连接...');
    const integration = new TopologyMemoryIntegration(config);
    const connected = await integration.testConnection();
    process.exit(connected ? 0 : 1);
  });

// 保存对话命令
program
  .command('save')
  .description('保存对话到拓扑记忆系统')
  .requiredOption('-r, --role <role>', '角色 (user/assistant)')
  .requiredOption('-c, --content <content>', '对话内容')
  .option('-s, --session <session>', '会话ID')
  .option('-m, --metadata <metadata>', '元数据 (JSON字符串)')
  .action(async (options) => {
    console.log('💾 保存对话到拓扑记忆系统...');
    
    const integration = new TopologyMemoryIntegration({
      ...config,
      sessionId: options.session || `session_${Date.now()}`
    });
    
    let metadata = {};
    if (options.metadata) {
      try {
        metadata = JSON.parse(options.metadata);
      } catch (e) {
        console.error('❌ 元数据格式错误，必须是有效的JSON字符串');
        process.exit(1);
      }
    }
    
    const result = await integration.saveConversation(
      options.role,
      options.content,
      metadata
    );
    
    if (result.success) {
      console.log(`✅ ${result.message}`);
      console.log(`   节点ID: ${result.nodeId}`);
    } else {
      console.error(`❌ 保存失败: ${result.error}`);
      process.exit(1);
    }
  });

// 搜索上下文命令
program
  .command('search')
  .description('搜索相关上下文')
  .requiredOption('-q, --query <query>', '搜索查询')
  .option('-l, --limit <limit>', '返回结果数量', '5')
  .action(async (options) => {
    console.log(`🔍 搜索上下文: "${options.query}"`);
    
    const integration = new TopologyMemoryIntegration(config);
    const results = await integration.searchContext(
      options.query,
      parseInt(options.limit)
    );
    
    if (results.length === 0) {
      console.log('📭 未找到相关上下文');
      return;
    }
    
    console.log(`📊 找到 ${results.length} 个相关结果:`);
    results.forEach((result, index) => {
      console.log(`\n${index + 1}. [相关性: ${result.relevance.toFixed(2)}]`);
      console.log(`   内容: ${result.content.substring(0, 100)}...`);
      console.log(`   节点ID: ${result.nodeId}`);
    });
  });

// 获取对话历史命令
program
  .command('history')
  .description('获取对话历史')
  .option('-l, --limit <limit>', '返回最近多少条', '10')
  .option('-s, --session <session>', '会话ID')
  .action(async (options) => {
    console.log('📜 获取对话历史...');
    
    const integration = new TopologyMemoryIntegration({
      ...config,
      sessionId: options.session
    });
    
    const history = await integration.getConversationHistory(
      parseInt(options.limit)
    );
    
    if (history.length === 0) {
      console.log('📭 没有对话历史');
      return;
    }
    
    console.log(`📊 最近 ${history.length} 条对话:`);
    history.forEach((item, index) => {
      console.log(`\n${index + 1}. [${item.role}] ${item.timestamp}`);
      console.log(`   内容: ${item.content.substring(0, 80)}...`);
      console.log(`   节点ID: ${item.nodeId}`);
    });
  });

// 构建上下文命令
program
  .command('context')
  .description('构建智能上下文')
  .requiredOption('-q, --query <query>', '当前查询')
  .option('-l, --length <length>', '最大长度', '2000')
  .option('--no-relevant', '不包含相关历史')
  .option('--no-recent', '不包含最近对话')
  .action(async (options) => {
    console.log(`🧠 为查询构建智能上下文: "${options.query}"`);
    
    const integration = new TopologyMemoryIntegration(config);
    const context = await integration.buildIntelligentContext(options.query, {
      includeRelevant: options.relevant,
      includeRecent: options.recent,
      maxLength: parseInt(options.length)
    });
    
    console.log(`\n📊 构建的上下文 (${context.length} 字符):`);
    console.log('='.repeat(60));
    console.log(context);
    console.log('='.repeat(60));
  });

// 优化记忆命令
program
  .command('optimize')
  .description('优化记忆存储')
  .action(async () => {
    console.log('🔄 优化记忆存储...');
    
    const integration = new TopologyMemoryIntegration(config);
    const result = await integration.optimizeMemories();
    
    if (result.success !== false) {
      console.log(`✅ ${result.message}`);
      console.log(`   优化节点: ${result.optimized_count}/${result.total_nodes}`);
    } else {
      console.error(`❌ 优化失败: ${result.error}`);
      process.exit(1);
    }
  });

// 系统状态命令
program
  .command('status')
  .description('获取系统状态')
  .action(async () => {
    console.log('📊 获取拓扑记忆系统状态...');
    
    const integration = new TopologyMemoryIntegration(config);
    const status = await integration.getSystemStatus();
    
    console.log(`🏥 系统状态: ${status.status}`);
    console.log(`📦 记忆节点: ${status.nodes_count} 个`);
    console.log(`🔗 记忆关联: ${status.edges_count} 条`);
    console.log(`📄 API版本: ${status.api_version}`);
    
    if (status.endpoints_available) {
      console.log(`🔌 可用端点: ${status.endpoints_available.join(', ')}`);
    }
  });

// 批量导入命令
program
  .command('import')
  .description('批量导入对话历史')
  .requiredOption('-f, --file <file>', '导入文件 (JSON格式)')
  .option('-s, --session <session>', '会话ID')
  .action(async (options) => {
    console.log('📥 批量导入对话历史...');
    
    const fs = require('fs');
    const path = require('path');
    
    try {
      const filePath = path.resolve(options.file);
      const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      
      if (!Array.isArray(data)) {
        console.error('❌ 导入文件必须是对话数组');
        process.exit(1);
      }
      
      const integration = new TopologyMemoryIntegration({
        ...config,
        sessionId: options.session || `import_${Date.now()}`
      });
      
      console.log(`📊 导入 ${data.length} 条对话记录...`);
      
      let successCount = 0;
      let errorCount = 0;
      
      for (let i = 0; i < data.length; i++) {
        const item = data[i];
        
        if (!item.role || !item.content) {
          console.warn(`⚠️  跳过第 ${i + 1} 条: 缺少 role 或 content`);
          errorCount++;
          continue;
        }
        
        try {
          await integration.saveConversation(
            item.role,
            item.content,
            item.metadata || {}
          );
          successCount++;
          
          // 显示进度
          if ((i + 1) % 10 === 0 || i === data.length - 1) {
            console.log(`  进度: ${i + 1}/${data.length}`);
          }
          
          // 避免请求过快
          await new Promise(resolve => setTimeout(resolve, 100));
          
        } catch (error) {
          console.error(`❌ 导入第 ${i + 1} 条失败: ${error.message}`);
          errorCount++;
        }
      }
      
      console.log(`\n✅ 导入完成!`);
      console.log(`   成功: ${successCount} 条`);
      console.log(`   失败: ${errorCount} 条`);
      console.log(`   总计: ${data.length} 条`);
      
    } catch (error) {
      console.error(`❌ 导入失败: ${error.message}`);
      process.exit(1);
    }
  });

// 导出命令
program
  .command('export')
  .description('导出对话历史')
  .option('-f, --file <file>', '导出文件', 'conversation_export.json')
  .option('-s, --session <session>', '会话ID')
  .option('-l, --limit <limit>', '导出数量', '100')
  .action(async (options) => {
    console.log('📤 导出对话历史...');
    
    const fs = require('fs');
    const path = require('path');
    
    const integration = new TopologyMemoryIntegration({
      ...config,
      sessionId: options.session
    });
    
    const history = await integration.getConversationHistory(
      parseInt(options.limit)
    );
    
    if (history.length === 0) {
      console.log('📭 没有对话历史可导出');
      return;
    }
    
    const exportData = history.map(item => ({
      role: item.role,
      content: item.content,
      timestamp: item.timestamp,
      nodeId: item.nodeId
    }));
    
    const filePath = path.resolve(options.file);
    fs.writeFileSync(filePath, JSON.stringify(exportData, null, 2), 'utf8');
    
    console.log(`✅ 导出完成!`);
    console.log(`   导出文件: ${filePath}`);
    console.log(`   导出数量: ${exportData.length} 条`);
  });

// 如果没有命令，显示帮助
if (process.argv.length <= 2) {
  program.help();
}

program.parse(process.argv);