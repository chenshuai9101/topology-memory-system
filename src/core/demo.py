"""
拓扑记忆上下文管理器 - 核心引擎演示
展示核心功能的使用方法
"""

import time
import json
from datetime import datetime

from .engine import TopologyMemoryEngine, EngineConfig
from .context_manager import ContextCreate, ContextUpdate
from .performance_test import PerformanceTester
from ..api.schemas import TopologyQuery


def demo_basic_operations():
    """演示基本操作"""
    print("=" * 80)
    print("拓扑记忆上下文管理器 - 核心引擎演示")
    print("=" * 80)
    
    # 创建引擎
    config = EngineConfig(
        max_contexts_per_session=50,
        max_memory_nodes=1000,
        vector_dimension=384
    )
    
    engine = TopologyMemoryEngine(config)
    
    print("1. 创建引擎实例 ✓")
    print(f"   配置: {config.max_contexts_per_session} 上下文/会话, "
          f"{config.max_memory_nodes} 记忆节点")
    
    # 演示上下文管理
    print("\n2. 上下文管理演示")
    print("-" * 40)
    
    # 创建上下文
    context_data = ContextCreate(
        session_id="demo_session_1",
        user_id="demo_user",
        context_type="conversation",
        content={
            "text": "用户询问关于人工智能的未来发展",
            "language": "zh-CN",
            "topic": "AI"
        },
        metadata={
            "source": "demo",
            "importance": "high"
        },
        priority=8,
        ttl=3600
    )
    
    context = engine.create_context(context_data)
    print(f"   创建上下文: ID={context.id[:8]}..., "
          f"类型={context.context_type}, 优先级={context.priority}")
    
    # 获取上下文
    retrieved = engine.get_context("demo_session_1", context.id)
    print(f"   获取上下文: 内容长度={len(str(retrieved.content))} 字符")
    
    # 更新上下文
    update_data = ContextUpdate(
        content={"text": "用户询问关于人工智能和机器学习的未来发展"},
        metadata={"updated": True, "timestamp": datetime.now().isoformat()},
        priority=9
    )
    
    updated = engine.update_context("demo_session_1", context.id, update_data)
    print(f"   更新上下文: 新优先级={updated.priority}")
    
    # 演示记忆管理
    print("\n3. 记忆管理演示")
    print("-" * 40)
    
    # 创建记忆节点
    memory_content = "人工智能是计算机科学的一个分支，旨在创造能够执行通常需要人类智能的任务的机器。"
    vector = [0.1] * 384  # 简化向量
    
    node_id = engine.create_memory_node(
        content=memory_content,
        vector=vector,
        metadata={
            "category": "knowledge",
            "domain": "AI",
            "language": "zh-CN"
        }
    )
    
    print(f"   创建记忆节点: ID={node_id[:8]}..., 内容长度={len(memory_content)} 字符")
    
    # 获取记忆节点
    memory_node = engine.get_memory_node(node_id)
    print(f"   获取记忆节点: 类别={memory_node.metadata.get('category')}")
    
    # 创建另一个节点并链接
    node2_content = "机器学习是人工智能的一个子领域，使计算机能够在没有明确编程的情况下学习。"
    node2_id = engine.create_memory_node(
        content=node2_content,
        vector=[0.15] * 384,
        metadata={"category": "knowledge", "domain": "ML"}
    )
    
    engine.link_memory_nodes(node_id, node2_id, "related_to", 0.85)
    print(f"   链接记忆节点: {node_id[:8]}... ↔ {node2_id[:8]}..., 权重=0.85")
    
    # 演示记忆搜索
    print("\n4. 记忆搜索演示")
    print("-" * 40)
    
    query = TopologyQuery(
        query="人工智能 机器学习",
        limit=5,
        threshold=0.3,
        include_vectors=False
    )
    
    search_start = time.time()
    search_result = engine.search_memory(query)
    search_time = (time.time() - search_start) * 1000
    
    print(f"   搜索查询: '{query.query}'")
    print(f"   找到 {search_result.total_nodes} 个节点, {search_result.total_edges} 条边")
    print(f"   搜索时间: {search_time:.2f}ms")
    
    if search_result.nodes:
        print(f"   第一个结果: {search_result.nodes[0].content[:50]}...")
    
    # 演示拓扑构建
    print("\n5. 拓扑构建演示")
    print("-" * 40)
    
    topology_start = time.time()
    topology_nodes, topology_edges = engine.build_topology(
        center_node_id=node_id,
        max_nodes=10,
        min_similarity=0.2
    )
    topology_time = (time.time() - topology_start) * 1000
    
    print(f"   以节点 {node_id[:8]}... 为中心构建拓扑")
    print(f"   构建 {len(topology_nodes)} 个节点, {len(topology_edges)} 条边")
    print(f"   构建时间: {topology_time:.2f}ms")
    
    # 演示拓扑分析
    print("\n6. 拓扑分析演示")
    print("-" * 40)
    
    analysis_start = time.time()
    analysis = engine.analyze_topology(node_id)
    analysis_time = (time.time() - analysis_start) * 1000
    
    print(f"   拓扑分析完成: {analysis_time:.2f}ms")
    
    if "basic_stats" in analysis:
        stats = analysis["basic_stats"]
        print(f"   节点数: {stats.get('num_nodes', 0)}, "
              f"边数: {stats.get('num_edges', 0)}, "
              f"密度: {stats.get('density', 0):.3f}")
    
    # 演示系统统计
    print("\n7. 系统统计演示")
    print("-" * 40)
    
    stats = engine.get_stats()
    
    ctx_stats = stats["context_manager"]
    mem_stats = stats["memory_manager"]
    perf_stats = stats["performance"]
    
    print(f"   上下文管理器: {ctx_stats.get('total_contexts', 0)} 个上下文")
    print(f"   记忆管理器: {mem_stats.get('total_nodes', 0)} 个节点, "
          f"{mem_stats.get('total_edges', 0)} 条边")
    print(f"   性能统计: {perf_stats.get('total_requests', 0)} 次请求, "
          f"平均响应时间: {perf_stats.get('avg_response_time', 0)*1000:.2f}ms")
    
    # 演示健康检查
    print("\n8. 健康检查演示")
    print("-" * 40)
    
    health = engine.health_check()
    print(f"   系统状态: {health['status']}")
    
    for component, status in health["components"].items():
        print(f"   {component}: {status}")
    
    # 清理演示数据
    print("\n9. 清理演示数据")
    print("-" * 40)
    
    # 删除上下文
    engine.delete_context("demo_session_1", context.id)
    print(f"   删除上下文: {context.id[:8]}...")
    
    print("\n演示完成! ✓")
    print("=" * 80)
    
    return engine


def demo_performance_test():
    """演示性能测试"""
    print("\n" + "=" * 80)
    print("性能测试演示")
    print("=" * 80)
    
    # 创建引擎
    config = EngineConfig(
        max_contexts_per_session=100,
        max_memory_nodes=5000
    )
    
    engine = TopologyMemoryEngine(config)
    
    # 创建性能测试器
    tester = PerformanceTester(engine)
    
    print("运行性能测试...")
    print("(每个测试运行50次迭代)")
    
    # 运行测试
    results = tester.run_all_tests(num_iterations=50)
    
    # 生成报告
    report = tester.generate_report()
    print("\n" + report)
    
    # 检查性能要求
    summary = results.get("summary", {})
    
    if summary.get("performance_requirement_met"):
        print("\n✓ 所有操作满足性能要求 (<50ms)")
    else:
        print("\n✗ 部分操作未满足性能要求")
        
        failed_tests = summary.get("failed_test_details", [])
        for test in failed_tests:
            print(f"  - {test['test']}: {test['avg_time_ms']:.2f}ms > {test['requirement']}ms")
    
    print("=" * 80)
    
    return results


def demo_advanced_features():
    """演示高级功能"""
    print("\n" + "=" * 80)
    print("高级功能演示")
    print("=" * 80)
    
    # 创建引擎
    engine = TopologyMemoryEngine()
    
    print("1. 批量创建记忆节点")
    print("-" * 40)
    
    # 批量创建相关主题的节点
    ai_topics = [
        ("神经网络", "模仿人脑神经元结构的计算模型"),
        ("深度学习", "基于神经网络的机器学习方法"),
        ("自然语言处理", "计算机理解和生成人类语言的技术"),
        ("计算机视觉", "计算机从图像和视频中提取信息的能力"),
        ("强化学习", "通过试错学习最优决策策略的方法")
    ]
    
    node_ids = []
    for topic, description in ai_topics:
        content = f"{topic}: {description}"
        node_id = engine.create_memory_node(
            content=content,
            metadata={"topic": topic, "domain": "AI"}
        )
        node_ids.append(node_id)
        print(f"   创建: {topic}")
    
    # 创建关联边
    print("\n2. 创建关联网络")
    print("-" * 40)
    
    # 链接相关主题
    relationships = [
        (0, 1, "subfield_of", 0.9),  # 深度学习是神经网络的子领域
        (0, 2, "used_in", 0.7),      # 神经网络用于NLP
        (0, 3, "used_in", 0.7),      # 神经网络用于CV
        (1, 2, "applies_to", 0.8),   # 深度学习应用于NLP
        (1, 3, "applies_to", 0.8),   # 深度学习应用于CV
        (1, 4, "related_to", 0.6),   # 深度学习与强化学习相关
    ]
    
    for i, j, rel, weight in relationships:
        engine.link_memory_nodes(
            node_ids[i], node_ids[j],
            relationship=rel,
            weight=weight
        )
        print(f"   链接: {ai_topics[i][0]} → {ai_topics[j][0]} ({rel}, {weight})")
    
    # 构建和分析拓扑
    print("\n3. 拓扑分析")
    print("-" * 40)
    
    center_node = node_ids[0]  # 以"神经网络"为中心
    analysis = engine.analyze_topology(center_node)
    
    if "basic_stats" in analysis:
        stats = analysis["basic_stats"]
        print(f"   拓扑统计: {stats.get('num_nodes', 0)} 节点, "
              f"{stats.get('num_edges', 0)} 边, "
              f"密度: {stats.get('density', 0):.3f}")
    
    if "centrality_measures" in analysis:
        centrality = analysis["centrality_measures"]
        if "degree" in centrality:
            degree_centrality = centrality["degree"]
            most_central = max(degree_centrality.items(), key=lambda x: x[1], default=(None, 0))
            if most_central[0]:
                # 查找节点内容
                node = engine.get_memory_node(most_central[0])
                if node:
                    topic = node.metadata.get("topic", "未知")
                    print(f"   最中心节点: {topic} (中心性: {most_central[1]:.3f})")
    
    # 查找相关上下文
    print("\n4. 上下文关联")
    print("-" * 40)
    
    # 创建与AI主题相关的上下文
    context_data = ContextCreate(
        session_id="ai_discussion",
        user_id="researcher",
        context_type="knowledge",
        content={
            "text": "讨论神经网络在自然语言处理中的应用",
            "topics": ["神经网络", "自然语言处理"],
            "depth": "technical"
        },
        priority=7
    )
    
    context = engine.create_context(context_data)
    print(f"   创建上下文: '{context.content.get('text', '')[:30]}...'")
    
    # 理论上应该能找到相关记忆节点
    query = TopologyQuery(query="神经网络 自然语言处理", limit=3)
    search_result = engine.search_memory(query)
    
    print(f"   找到 {len(search_result.nodes)} 个相关记忆节点")
    for i, node in enumerate(search_result.nodes[:2], 1):
        print(f"     {i}. {node.content[:40]}...")
    
    print("\n高级功能演示完成! ✓")
    print("=" * 80)


def main():
    """主演示函数"""
    try:
        print("拓扑记忆上下文管理器 - 核心引擎完整演示")
        print("=" * 80)
        
        # 演示基本操作
        engine = demo_basic_operations()
        
        # 演示性能测试
        demo_performance_test()
        
        # 演示高级功能
        demo_advanced_features()
        
        print("\n" + "=" * 80)
        print("所有演示完成!")
        print("=" * 80)
        
        # 最终统计
        final_stats = engine.get_stats()
        
        print("\n最终系统统计:")
        print("-" * 40)
        
        ctx_total = final_stats["context_manager"].get("total_contexts", 0)
        mem_nodes = final_stats["memory_manager"].get("total_nodes", 0)
        mem_edges = final_stats["memory_manager"].get("total_edges", 0)
        avg_response = final_stats["performance"].get("avg_response_time", 0) * 1000
        
        print(f"• 总上下文数: {ctx_total}")
        print(f"• 总记忆节点: {mem_nodes}")
        print(f"• 总记忆边数: {mem_edges}")
        print(f"• 平均响应时间: {avg_response:.2f}ms")
        
        # 检查性能要求
        if avg_response < 50:
            print(f"✓ 平均响应时间满足要求 (<50ms)")
        else:
            print(f"✗ 平均响应时间未满足要求: {avg_response:.2f}ms > 50ms")
        
        print("\n核心引擎开发完成!")
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()