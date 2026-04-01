"""
数据库性能测试脚本
"""

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import List, Dict, Any
from uuid import UUID

import psutil

from database.config.database_manager import db_manager
from database.services.database_service import db_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceTester:
    """性能测试器"""
    
    def __init__(self, max_workers: int = 100):
        self.max_workers = max_workers
        self.test_data = []
        self.results = []
    
    def generate_test_data(self, count: int = 1000):
        """生成测试数据"""
        logger.info(f"生成 {count} 条测试数据...")
        
        for i in range(count):
            session_id = f"test_session_{i % 100}"
            user_id = f"test_user_{i % 50}"
            
            # 上下文数据
            context_data = {
                "session_id": session_id,
                "user_id": user_id,
                "context_type": "conversation",
                "content": {"message": f"Test message {i}", "index": i},
                "metadata": {"test": True, "index": i},
                "priority": (i % 10) + 1,
                "ttl": 3600
            }
            
            # 记忆节点数据
            node_data = {
                "node_type": "concept",
                "content": f"Test concept {i}",
                "summary": f"Summary for concept {i}",
                "metadata": {"category": "test", "index": i},
                "tags": ["test", f"tag_{i % 10}"]
            }
            
            self.test_data.append({
                "context": context_data,
                "node": node_data,
                "index": i
            })
        
        logger.info("测试数据生成完成")
    
    def test_single_operation(self, operation: str, **kwargs) -> Dict[str, Any]:
        """测试单个操作"""
        start_time = time.time()
        
        try:
            if operation == "create_context":
                result = db_service.create_context(**kwargs)
            elif operation == "get_context":
                result = db_service.get_context(kwargs["context_id"])
            elif operation == "create_node":
                result = db_service.create_memory_node(**kwargs)
            elif operation == "get_node":
                result = db_service.get_memory_node(kwargs["node_id"])
            elif operation == "search_nodes":
                result = db_service.search_nodes(kwargs["query"], kwargs.get("limit", 20))
            elif operation == "get_session_contexts":
                result = db_service.get_session_contexts(kwargs["session_id"])
            elif operation == "get_important_nodes":
                result = db_service.get_important_nodes(
                    kwargs.get("threshold", 0.7),
                    kwargs.get("limit", 50)
                )
            elif operation == "get_related_nodes":
                result = db_service.get_related_nodes(
                    kwargs["node_id"],
                    kwargs.get("relation_type"),
                    kwargs.get("limit", 20)
                )
            elif operation == "create_association":
                result = db_service.create_association(**kwargs)
            elif operation == "get_system_stats":
                result = db_service.get_system_stats()
            elif operation == "health_check":
                result = db_service.health_check()
            else:
                raise ValueError(f"未知操作: {operation}")
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # 转换为毫秒
            
            return {
                "operation": operation,
                "success": True,
                "response_time_ms": response_time,
                "result_size": len(str(result)) if result else 0
            }
            
        except Exception as e:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            return {
                "operation": operation,
                "success": False,
                "response_time_ms": response_time,
                "error": str(e)
            }
    
    def test_concurrent_operations(self, operation: str, count: int, **kwargs) -> List[Dict[str, Any]]:
        """测试并发操作"""
        logger.info(f"开始并发测试: {operation}, 并发数: {count}")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for i in range(count):
                # 为每个请求准备参数
                request_kwargs = kwargs.copy()
                
                if operation == "create_context":
                    # 为每个请求生成不同的数据
                    if self.test_data:
                        data = self.test_data[i % len(self.test_data)]
                        request_kwargs.update(data["context"])
                
                elif operation == "get_context":
                    # 需要有效的context_id，这里使用占位符
                    if "context_id" not in request_kwargs:
                        request_kwargs["context_id"] = uuid.uuid4()
                
                elif operation == "create_node":
                    if self.test_data:
                        data = self.test_data[i % len(self.test_data)]
                        request_kwargs.update(data["node"])
                
                # 提交任务
                future = executor.submit(self.test_single_operation, operation, **request_kwargs)
                futures.append(future)
            
            # 收集结果
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=30)  # 30秒超时
                    results.append(result)
                except Exception as e:
                    results.append({
                        "operation": operation,
                        "success": False,
                        "error": str(e)
                    })
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析测试结果"""
        if not results:
            return {}
        
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]
        
        response_times = [r["response_time_ms"] for r in successful if "response_time_ms" in r]
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
            
            # 计算百分位数
            sorted_times = sorted(response_times)
            p50 = sorted_times[int(len(sorted_times) * 0.5)]
            p95 = sorted_times[int(len(sorted_times) * 0.95)]
            p99 = sorted_times[int(len(sorted_times) * 0.99)]
        else:
            avg_time = max_time = min_time = p50 = p95 = p99 = 0
        
        return {
            "total_requests": len(results),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "success_rate": len(successful) / len(results) * 100 if results else 0,
            "avg_response_time_ms": avg_time,
            "min_response_time_ms": min_time,
            "max_response_time_ms": max_time,
            "p50_response_time_ms": p50,
            "p95_response_time_ms": p95,
            "p99_response_time_ms": p99,
            "throughput_rps": len(successful) / (max(response_times) / 1000) if response_times else 0
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用
            memory = psutil.virtual_memory()
            
            # 磁盘IO
            disk_io = psutil.disk_io_counters()
            
            # 网络IO
            net_io = psutil.net_io_counters()
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_read_mb": disk_io.read_bytes / (1024**2) if disk_io else 0,
                "disk_write_mb": disk_io.write_bytes / (1024**2) if disk_io else 0,
                "network_sent_mb": net_io.bytes_sent / (1024**2),
                "network_recv_mb": net_io.bytes_recv / (1024**2)
            }
        except Exception as e:
            logger.error(f"获取系统指标失败: {e}")
            return {}
    
    def run_comprehensive_test(self, concurrent_users: int = 100, requests_per_user: int = 10):
        """运行综合性能测试"""
        logger.info(f"开始综合性能测试: {concurrent_users}并发用户, 每个用户{requests_per_user}个请求")
        
        test_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "test_config": {
                "concurrent_users": concurrent_users,
                "requests_per_user": requests_per_user,
                "max_workers": self.max_workers
            },
            "system_metrics_before": self.get_system_metrics(),
            "test_results": {},
            "summary": {}
        }
        
        # 生成测试数据
        self.generate_test_data(concurrent_users * 2)
        
        # 测试1: 创建上下文（写入测试）
        logger.info("测试1: 创建上下文（写入测试）")
        create_context_results = []
        for i in range(concurrent_users):
            user_results = self.test_concurrent_operations(
                "create_context",
                requests_per_user
            )
            create_context_results.extend(user_results)
        
        test_report["test_results"]["create_context"] = self.analyze_results(create_context_results)
        
        # 测试2: 获取上下文（读取测试）
        logger.info("测试2: 获取上下文（读取测试）")
        # 先获取一些有效的context_id
        context_ids = []
        with db_manager.get_session() as session:
            from database.models.contexts import Context
            contexts = session.query(Context.id).limit(concurrent_users).all()
            context_ids = [ctx[0] for ctx in contexts]
        
        get_context_results = []
        for i in range(concurrent_users):
            if i < len(context_ids):
                user_results = self.test_concurrent_operations(
                    "get_context",
                    requests_per_user,
                    context_id=context_ids[i]
                )
                get_context_results.extend(user_results)
        
        test_report["test_results"]["get_context"] = self.analyze_results(get_context_results)
        
        # 测试3: 创建记忆节点
        logger.info("测试3: 创建记忆节点")
        create_node_results = []
        for i in range(concurrent_users):
            user_results = self.test_concurrent_operations(
                "create_node",
                requests_per_user
            )
            create_node_results.extend(user_results)
        
        test_report["test_results"]["create_node"] = self.analyze_results(create_node_results)
        
        # 测试4: 搜索节点
        logger.info("测试4: 搜索节点")
        search_results = []
        for i in range(concurrent_users):
            user_results = self.test_concurrent_operations(
                "search_nodes",
                requests_per_user,
                query="test",
                limit=20
            )
            search_results.extend(user_results)
        
        test_report["test_results"]["search_nodes"] = self.analyze_results(search_results)
        
        # 测试5: 获取会话上下文
        logger.info("测试5: 获取会话上下文")
        session_context_results = []
        for i in range(concurrent_users):
            session_id = f"test_session_{i % 100}"
            user_results = self.test_concurrent_operations(
                "get_session_contexts",
                requests_per_user,
                session_id=session_id
            )
            session_context_results.extend(user_results)
        
        test_report["test_results"]["get_session_contexts"] = self.analyze_results(session_context_results)
        
        # 测试6: 系统统计查询
        logger.info("测试6: 系统统计查询")
        stats_results = []
        for i in range(concurrent_users):
            user_results = self.test_concurrent_operations(
                "get_system_stats",
                requests_per_user
            )
            stats_results.extend(user_results)
        
        test_report["test_results"]["get_system_stats"] = self.analyze_results(stats_results)
        
        # 测试7: 健康检查
        logger.info("测试7: 健康检查")
        health_results = []
        for i in range(concurrent_users):
            user_results = self.test_concurrent_operations(
                "health_check",
                requests_per_user
            )
            health_results.extend(user_results)
        
        test_report["test_results"]["health_check"] = self.analyze_results(health_results)
        
        # 计算总体统计
        all_results = []
        for test_name, analysis in test_report["test_results"].items():
            if analysis:
                all_results.extend([analysis] * analysis.get("total_requests", 0))
        
        if all_results:
            total_requests = sum(r.get("total_requests", 0) for r in test_report["test_results"].values())
            successful_requests = sum(r.get("successful_requests", 0) for r in test_report["test_results"].values())
            
            response_times = []
            for test_name, analysis in test_report["test_results"].items():
                if analysis and analysis.get("avg_response_time_ms", 0) > 0:
                    response_times.append(analysis["avg_response_time_ms"])
            
            test_report["summary"] = {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "overall_success_rate": successful_requests / total_requests * 100 if total_requests > 0 else 0,
                "avg_response_time_across_tests": sum(response_times) / len(response_times) if response_times else 0,
                "max_response_time_across_tests": max(response_times) if response_times else 0,
                "meets_20ms_target": all(t <= 20 for t in response_times) if response_times else False
            }
        
        # 获取测试后的系统指标
        test_report["system_metrics_after"] = self.get_system_metrics()
        
        return test_report
    
    def save_test_report(self, report: Dict[str, Any], filename: str = None):
        """保存测试报告"""
        import json
        from pathlib import Path
        
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_test_{timestamp}.json"
        
        # 确保目录存在
        reports_dir = Path(__file__).parent / "performance_reports"
        reports_dir.mkdir(exist_ok=True)
        
        filepath = reports_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"测试报告已保存: {filepath}")
        return filepath
    
    def print_test_summary(self, report: Dict[str, Any]):
        """打印测试摘要"""
        print("\n" + "="*80)
        print("性能测试报告摘要")
        print("="*80)
        
        print(f"\n测试时间: {report.get('timestamp', 'N/A')}")
        print(f"测试配置: {report.get('test_config', {})}")
        
        print("\n" + "-"*80)
        print("各操作性能指标:")
        print("-"*80)
        
        for test_name, analysis in report.get("test_results", {}).items():
            if analysis:
                print(f"\n{test_name}:")
                print(f"  总请求数: {analysis.get('total_requests', 0)}")
                print(f"  成功请求: {analysis.get('successful_requests', 0)}")
                print(f"  成功率: {analysis.get('success_rate', 0):.2f}%")
                print(f"  平均响应时间: {analysis.get('avg_response_time_ms', 0):.2f}ms")
                print(f"  P95响应时间: {analysis.get('p95_response_time_ms', 0):.2f}ms")
                print(f"  P99响应时间: {analysis.get('p99_response_time_ms', 0):.2f}ms")
                print(f"  吞吐量: {analysis.get('throughput_rps', 0):.2f} req/s")
        
        print("\n" + "-"*80)
        print("总体统计:")
        print("-"*80)
        
        summary = report.get("summary", {})
        if summary:
            print(f"总请求数: {summary.get('total_requests', 0)}")
            print(f"成功请求数: {summary.get('successful_requests', 0)}")
            print(f"总体成功率: {summary.get('overall_success_rate', 0):.2f}%")
            print(f"跨测试平均响应时间: {summary.get('avg_response_time_across_tests', 0):.2f}ms")
            print(f"跨测试最大响应时间: {summary.get('max_response_time_across_tests', 0):.2f}ms")
            
            meets_target = summary.get('meets_20ms_target', False)
            target_status = "✅ 达标" if meets_target else "❌ 未达标"
            print(f"20ms响应时间目标: {target_status}")
        
        print("\n" + "-"*80)
        print("系统资源使用:")
        print("-"*80)
        
        metrics_before = report.get("system_metrics_before", {})
        metrics_after = report.get("system_metrics_after", {})
        
        if metrics_before and metrics_after:
            print(f"CPU使用率变化: {metrics_before.get('cpu_percent', 0):.1f}% -> {metrics_after.get('cpu_percent', 0):.1f}%")
            print(f"内存使用变化: {metrics_before.get('memory_percent', 0):.1f}% -> {metrics_after.get('memory_percent', 0):.1f}%")
            print(f"内存使用量: {metrics_after.get('memory_used_gb', 0):.2f}GB / {metrics_after.get('memory_total_gb', 0):.2f}GB")
            print(f"磁盘读取: {metrics_after.get('disk_read_mb', 0):.2f}MB")
            print(f"磁盘写入: {metrics_after.get('disk_write_mb', 0):.2f}MB")
        
        print("\n" + "="*80)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据库性能测试工具")
    parser.add_argument("--concurrent", type=int, default=100, 
                       help="并发用户数 (默认: 100)")
    parser.add_argument("--requests", type=int, default=10,
                       help="每个用户的请求数 (默认: 10)")
    parser.add_argument("--workers", type=int, default=100,
                       help="最大工作线程数 (默认: 100)")
    parser.add_argument("--output", type=str, default=None,
                       help="输出文件名 (默认: 自动生成)")
    
    args = parser.parse_args()
    
    # 初始化数据库连接
    try:
        db_manager.init_all()
        logger.info("数据库连接初始化成功")
    except Exception as e:
        logger.error(f"数据库连接初始化失败: {e}")
        return
    
    # 运行性能测试
    tester = PerformanceTester(max_workers=args.workers)
    
    try:
        report = tester.run_comprehensive_test(
            concurrent_users=args.concurrent,
            requests_per_user=args.requests
        )
        
        # 打印摘要
        tester.print_test_summary(report)
        
        # 保存报告
        if args.output:
            filename = args.output
        else:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_test_{timestamp}.json"
        
        saved_file = tester.save_test_report(report, filename)
        logger.info(f"详细测试报告已保存到: {saved_file}")
        
        # 检查是否达到性能目标
        summary = report.get("summary", {})
        if summary.get("meets_20ms_target", False):
            logger.info("✅ 性能测试通过: 所有操作响应时间 < 20ms")
        else:
            logger.warning("⚠️ 性能测试未完全达标: 部分操作响应时间 > 20ms")
            
    except Exception as e:
        logger.error(f"性能测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()