"""
性能测试模块
验证核心引擎的性能要求（所有操作<50ms）
"""

import time
import random
import statistics
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta

from .engine import TopologyMemoryEngine, EngineConfig
from .context_manager import ContextCreate, ContextUpdate
from ..api.schemas import TopologyQuery


logger = logging.getLogger(__name__)


class PerformanceTester:
    """性能测试器"""
    
    def __init__(self, engine: TopologyMemoryEngine):
        """
        初始化性能测试器
        
        Args:
            engine: 要测试的引擎实例
        """
        self.engine = engine
        self.results = {}
        
        # 测试数据
        self.test_contexts = []
        self.test_memory_nodes = []
        
        logger.info("PerformanceTester initialized")
    
    def run_all_tests(self, num_iterations: int = 100) -> Dict[str, Any]:
        """
        运行所有性能测试
        
        Args:
            num_iterations: 每个测试的迭代次数
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        logger.info(f"Starting performance tests with {num_iterations} iterations")
        
        # 准备测试数据
        self._prepare_test_data(num_iterations)
        
        # 运行测试
        tests = [
            ("context_create", self.test_context_create),
            ("context_get", self.test_context_get),
            ("context_update", self.test_context_update),
            ("context_delete", self.test_context_delete),
            ("memory_node_create", self.test_memory_node_create),
            ("memory_node_get", self.test_memory_node_get),
            ("memory_search", self.test_memory_search),
            ("topology_build", self.test_topology_build),
            ("topology_analyze", self.test_topology_analyze)
        ]
        
        all_results = {}
        
        for test_name, test_func in tests:
            logger.info(f"Running test: {test_name}")
            try:
                result = test_func(num_iterations)
                all_results[test_name] = result
                
                # 检查是否满足性能要求
                if result["avg_time_ms"] > 50:
                    logger.warning(f"Test {test_name} failed performance requirement: "
                                 f"{result['avg_time_ms']:.2f}ms > 50ms")
                else:
                    logger.info(f"Test {test_name} passed: {result['avg_time_ms']:.2f}ms")
                    
            except Exception as e:
                logger.error(f"Test {test_name} failed: {e}")
                all_results[test_name] = {
                    "error": str(e),
                    "avg_time_ms": 0,
                    "p95_time_ms": 0,
                    "success_rate": 0
                }
        
        # 汇总结果
        summary = self._generate_summary(all_results)
        all_results["summary"] = summary
        
        # 保存结果
        self.results = all_results
        
        return all_results
    
    def test_context_create(self, num_iterations: int) -> Dict[str, Any]:
        """测试上下文创建性能"""
        times = []
        successes = 0
        
        for i in range(num_iterations):
            context_data = ContextCreate(
                session_id=f"test_session_{i % 10}",
                user_id=f"test_user_{i % 5}",
                context_type=random.choice(["conversation", "memory", "task"]),
                content={"text": f"Test content {i}" * 10},
                metadata={"test": True, "iteration": i},
                priority=random.randint(1, 10),
                ttl=random.choice([None, 300, 600])
            )
            
            start_time = time.time()
            try:
                result = self.engine.create_context(context_data)
                if result:
                    successes += 1
                    self.test_contexts.append((result.session_id, result.id))
            except Exception as e:
                logger.debug(f"Context create failed: {e}")
            
            times.append((time.time() - start_time) * 1000)  # 转换为毫秒
        
        return self._calculate_metrics(times, successes, num_iterations)
    
    def test_context_get(self, num_iterations: int) -> Dict[str, Any]:
        """测试上下文获取性能"""
        if not self.test_contexts:
            self._prepare_context_test_data(num_iterations)
        
        times = []
        successes = 0
        
        for i in range(num_iterations):
            if not self.test_contexts:
                break
            
            session_id, context_id = random.choice(self.test_contexts)
            
            start_time = time.time()
            try:
                result = self.engine.get_context(session_id, context_id)
                if result:
                    successes += 1
            except Exception as e:
                logger.debug(f"Context get failed: {e}")
            
            times.append((time.time() - start_time) * 1000)
        
        return self._calculate_metrics(times, successes, num_iterations)
    
    def test_context_update(self, num_iterations: int) -> Dict[str, Any]:
        """测试上下文更新性能"""
        if not self.test_contexts:
            self._prepare_context_test_data(num_iterations)
        
        times = []
        successes = 0
        
        for i in range(num_iterations):
            if not self.test_contexts:
                break
            
            session_id, context_id = random.choice(self.test_contexts)
            update_data = ContextUpdate(
                content={"text": f"Updated content {i}"},
                metadata={"updated": True, "timestamp": datetime.now().isoformat()},
                priority=random.randint(1, 10)
            )
            
            start_time = time.time()
            try:
                result = self.engine.update_context(session_id, context_id, update_data)
                if result:
                    successes += 1
            except Exception as e:
                logger.debug(f"Context update failed: {e}")
            
            times.append((time.time() - start_time) * 1000)
        
        return self._calculate_metrics(times, successes, num_iterations)
    
    def test_context_delete(self, num_iterations: int) -> Dict[str, Any]:
        """测试上下文删除性能"""
        if not self.test_contexts:
            self._prepare_context_test_data(num_iterations)
        
        times = []
        successes = 0
        
        for i in range(min(num_iterations, len(self.test_contexts))):
            if not self.test_contexts:
                break
            
            session_id, context_id = self.test_contexts.pop()
            
            start_time = time.time()
            try:
                result = self.engine.delete_context(session_id, context_id)
                if result:
                    successes += 1
            except Exception as e:
                logger.debug(f"Context delete failed: {e}")
            
            times.append((time.time() - start_time) * 1000)
        
        return self._calculate_metrics(times, successes, num_iterations)
    
    def test_memory_node_create(self, num_iterations: int) -> Dict[str, Any]:
        """测试记忆节点创建性能"""
        times = []
        successes = 0
        
        for i in range(num_iterations):
            content = f"Test memory node content {i} " * 5
            
            # 生成随机向量
            vector = [random.random() for _ in range(384)] if i % 2 == 0 else None
            
            metadata = {
                "test": True,
                "iteration": i,
                "category": random.choice(["fact", "concept", "experience"])
            }
            
            start_time = time.time()
            try:
                node_id = self.engine.create_memory_node(content, vector, metadata)
                if node_id:
                    successes += 1
                    self.test_memory_nodes.append(node_id)
            except Exception as e:
                logger.debug(f"Memory node create failed: {e}")
            
            times.append((time.time() - start_time) * 1000)
        
        return self._calculate_metrics(times, successes, num_iterations)
    
    def test_memory_node_get(self, num_iterations: int) -> Dict[str, Any]:
        """测试记忆节点获取性能"""
        if not self.test_memory_nodes:
            self._prepare_memory_test_data(num_iterations)
        
        times = []
        successes = 0
        
        for i in range(num_iterations):
            if not self.test_memory_nodes:
                break
            
            node_id = random.choice(self.test_memory_nodes)
            
            start_time = time.time()
            try:
                result = self.engine.get_memory_node(node_id)
                if result:
                    successes += 1
            except Exception as e:
                logger.debug(f"Memory node get failed: {e}")
            
            times.append((time.time() - start_time) * 1000)
        
        return self._calculate_metrics(times, successes, num_iterations)
    
    def test_memory_search(self, num_iterations: int) -> Dict[str, Any]:
        """测试记忆搜索性能"""
        if not self.test_memory_nodes:
            self._prepare_memory_test_data(num_iterations)
        
        times = []
        successes = 0
        
        search_queries = [
            "test memory",
            "content search",
            "performance testing",
            "random query",
            "topology memory"
        ]
        
        for i in range(num_iterations):
            query_text = random.choice(search_queries)
            query = TopologyQuery(
                query=query_text,
                limit=random.randint(5, 20),
                threshold=random.uniform(0.3, 0.8)
            )
            
            start_time = time.time()
            try:
                result = self.engine.search_memory(query)
                if result:
                    successes += 1
            except Exception as e:
                logger.debug(f"Memory search failed: {e}")
            
            times.append((time.time() - start_time) * 1000)
        
        return self._calculate_metrics(times, successes, num_iterations)
    
    def test_topology_build(self, num_iterations: int) -> Dict[str, Any]:
        """测试拓扑构建性能"""
        if not self.test_memory_nodes:
            self._prepare_memory_test_data(num_iterations)
        
        times = []
        successes = 0
        
        for i in range(min(num_iterations, len(self.test_memory_nodes))):
            if not self.test_memory_nodes:
                break
            
            center_node_id = random.choice(self.test_memory_nodes)
            
            start_time = time.time()
            try:
                nodes, edges = self.engine.build_topology(
                    center_node_id,
                    max_nodes=random.randint(10, 30),
                    min_similarity=random.uniform(0.2, 0.6)
                )
                if nodes:
                    successes += 1
            except Exception as e:
                logger.debug(f"Topology build failed: {e}")
            
            times.append((time.time() - start_time) * 1000)
        
        return self._calculate_metrics(times, successes, num_iterations)
    
    def test_topology_analyze(self, num_iterations: int) -> Dict[str, Any]:
        """测试拓扑分析性能"""
        if not self.test_memory_nodes:
            self._prepare_memory_test_data(num_iterations)
        
        times = []
        successes = 0
        
        for i in range(min(num_iterations, len(self.test_memory_nodes))):
            if not self.test_memory_nodes:
                break
            
            center_node_id = random.choice(self.test_memory_nodes)
            
            start_time = time.time()
            try:
                analysis = self.engine.analyze_topology(center_node_id)
                if analysis and "error" not in analysis:
                    successes += 1
            except Exception as e:
                logger.debug(f"Topology analyze failed: {e}")
            
            times.append((time.time() - start_time) * 1000)
        
        return self._calculate_metrics(times, successes, num_iterations)
    
    def _prepare_test_data(self, num_items: int) -> None:
        """准备测试数据"""
        # 准备上下文测试数据
        self._prepare_context_test_data(num_items // 2)
        
        # 准备记忆测试数据
        self._prepare_memory_test_data(num_items // 2)
    
    def _prepare_context_test_data(self, num_contexts: int) -> None:
        """准备上下文测试数据"""
        if self.test_contexts:
            return
        
        logger.info(f"Preparing {num_contexts} test contexts")
        
        for i in range(num_contexts):
            context_data = ContextCreate(
                session_id=f"prep_session_{i % 5}",
                user_id="test_user",
                context_type="conversation",
                content={"text": f"Preparation content {i}"},
                metadata={"prepared": True},
                priority=5,
                ttl=3600
            )
            
            try:
                result = self.engine.create_context(context_data)
                if result:
                    self.test_contexts.append((result.session_id, result.id))
            except Exception as e:
                logger.debug(f"Failed to prepare context: {e}")
    
    def _prepare_memory_test_data(self, num_nodes: int) -> None:
        """准备记忆测试数据"""
        if self.test_memory_nodes:
            return
        
        logger.info(f"Preparing {num_nodes} test memory nodes")
        
        for i in range(num_nodes):
            content = f"Preparation memory node {i} with some content for testing."
            
            # 每3个节点生成一个向量
            vector = None
            if i % 3 == 0:
                vector = [random.random() for _ in range(384)]
            
            metadata = {
                "prepared": True,
                "index": i,
                "category": ["fact", "concept", "experience"][i % 3]
            }
            
            try:
                node_id = self.engine.create_memory_node(content, vector, metadata)
                if node_id:
                    self.test_memory_nodes.append(node_id)
            except Exception as e:
                logger.debug(f"Failed to prepare memory node: {e}")
    
    def _calculate_metrics(self, times: List[float], successes: int, total: int) -> Dict[str, Any]:
        """计算性能指标"""
        if not times:
            return {
                "avg_time_ms": 0,
                "min_time_ms": 0,
                "max_time_ms": 0,
                "p95_time_ms": 0,
                "std_dev_ms": 0,
                "success_rate": 0,
                "total_operations": total
            }
        
        # 计算百分位数
        sorted_times = sorted(times)
        p95_index = int(len(sorted_times) * 0.95)
        p95_time = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]
        
        return {
            "avg_time_ms": statistics.mean(times),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "p95_time_ms": p95_time,
            "std_dev_ms": statistics.stdev(times) if len(times) > 1 else 0,
            "success_rate": successes / total if total > 0 else 0,
            "total_operations": total,
            "successful_operations": successes
        }
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成测试摘要"""
        # 检查性能要求
        failed_tests = []
        passed_tests = []
        
        for test_name, result in results.items():
            if test_name == "summary":
                continue
            
            if "avg_time_ms" in result:
                if result["avg_time_ms"] > 50:
                    failed_tests.append({
                        "test": test_name,
                        "avg_time_ms": result["avg_time_ms"],
                        "requirement": 50
                    })
                else:
                    passed_tests.append(test_name)
        
        # 计算总体性能
        all_times = []
        for test_name, result in results.items():
            if test_name == "summary":
                continue
            
            if "avg_time_ms" in result:
                all_times.append(result["avg_time_ms"])
        
        overall_avg = statistics.mean(all_times) if all_times else 0
        
        return {
            "total_tests": len(results) - 1,  # 减去summary本身
            "passed_tests": len(passed_tests),
            "failed_tests": len(failed_tests),
            "overall_avg_time_ms": overall_avg,
            "performance_requirement_met": overall_avg <= 50,
            "failed_test_details": failed_tests,
            "passed_test_list": passed_tests,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_report(self) -> str:
        """生成测试报告"""
        if not self.results:
            return "No test results available. Run tests first."
        
        summary = self.results.get("summary", {})
        
        report_lines = [
            "=" * 80,
            "拓扑记忆上下文管理器 - 性能测试报告",
            "=" * 80,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"测试总数: {summary.get('total_tests', 0)}",
            f"通过测试: {summary.get('passed_tests', 0)}",
            f"失败测试: {summary.get('failed_tests', 0)}",
            f"总体平均响应时间: {summary.get('overall_avg_time_ms', 0):.2f}ms",
            f"性能要求(<50ms): {'✓ 满足' if summary.get('performance_requirement_met') else '✗ 未满足'}",
            "",
            "详细测试结果:",
            "-" * 40
        ]
        
        for test_name, result in self.results.items():
            if test_name == "summary":
                continue
            
            if "avg_time_ms" in result:
                status = "✓" if result["avg_time_ms"] <= 50 else "✗"
                report_lines.append(
                    f"{status} {test_name:<20} "
                    f"平均: {result['avg_time_ms']:6.2f}ms "
                    f"P95: {result.get('p95_time_ms', 0):6.2f}ms "
                    f"成功率: {result.get('success_rate', 0)*100:5.1f}%"
                )
            else:
                report_lines.append(f"✗ {test_name:<20} 错误: {result.get('error', 'Unknown error')}")
        
        # 添加失败测试详情
        if summary.get('failed_tests', 0) > 0:
            report_lines.extend([
                "",
                "失败测试详情:",
                "-" * 40
            ])
            
            for failed_test in summary.get('failed_test_details', []):
                report_lines.append(
                    f"  {failed_test['test']:<20} "
                    f"平均时间: {failed_test['avg_time_ms']:.2f}ms "
                    f"(要求: {failed_test['requirement']}ms)"
                )
        
        report_lines.extend([
            "",
            "=" * 80,
            "测试完成"
        ])
        
        return "\n".join(report_lines)