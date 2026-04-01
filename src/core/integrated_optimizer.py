"""
集成优化器 - 统一管理所有优化模块
负责协调记忆优化、关联优化和检索优化
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, field

from .memory_optimizer import MemoryOptimizer, CompressionConfig, MemoryQualityMetrics
from .association_optimizer import AssociationOptimizer, AssociationConfig, AssociationMetrics
from .retrieval_optimizer import RetrievalOptimizer, RetrievalConfig, RetrievalMetrics

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """集成优化配置"""
    # 记忆优化配置
    memory_compression_enabled: bool = True
    memory_compression_config: CompressionConfig = field(default_factory=CompressionConfig)
    
    # 关联优化配置
    association_optimization_enabled: bool = True
    association_config: AssociationConfig = field(default_factory=AssociationConfig)
    
    # 检索优化配置
    retrieval_optimization_enabled: bool = True
    retrieval_config: RetrievalConfig = field(default_factory=RetrievalConfig)
    
    # 整体配置
    optimization_interval: int = 3600  # 优化间隔(秒)
    batch_size: int = 100  # 批量优化大小
    enable_realtime_optimization: bool = True  # 是否启用实时优化


@dataclass
class OptimizationReport:
    """优化报告"""
    timestamp: datetime
    duration: float
    memory_optimization: Dict[str, Any]
    association_optimization: Dict[str, Any]
    retrieval_optimization: Dict[str, Any]
    overall_quality: float
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'duration': self.duration,
            'memory_optimization': self.memory_optimization,
            'association_optimization': self.association_optimization,
            'retrieval_optimization': self.retrieval_optimization,
            'overall_quality': self.overall_quality,
            'recommendations': self.recommendations
        }


class IntegratedOptimizer:
    """集成优化器核心类"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        初始化集成优化器
        
        Args:
            config: 优化配置
        """
        self.config = config or OptimizationConfig()
        
        # 初始化各优化器
        self.memory_optimizer = MemoryOptimizer(self.config.memory_compression_config)
        self.association_optimizer = AssociationOptimizer(self.config.association_config)
        self.retrieval_optimizer = RetrievalOptimizer(self.config.retrieval_config)
        
        # 优化历史
        self.optimization_history: List[OptimizationReport] = []
        
        # 性能统计
        self.stats = {
            'total_optimizations': 0,
            'total_duration': 0.0,
            'avg_quality': 0.0,
            'last_optimization': None
        }
        
        logger.info("IntegratedOptimizer initialized")
    
    def optimize_memory_system(self,
                              nodes: List[Dict[str, Any]],
                              edges: List[Dict[str, Any]],
                              context: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], OptimizationReport]:
        """
        优化整个记忆系统
        
        Args:
            nodes: 记忆节点列表
            edges: 记忆边列表
            context: 上下文信息
            
        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]], OptimizationReport]: 
            优化后的节点、边和优化报告
        """
        start_time = time.time()
        
        logger.info(f"Starting memory system optimization: {len(nodes)} nodes, {len(edges)} edges")
        
        # 1. 优化记忆内容
        optimized_nodes = self._optimize_memory_content(nodes, context)
        
        # 2. 优化记忆关联
        optimized_edges = self._optimize_memory_associations(optimized_nodes, edges, context)
        
        # 3. 优化检索系统
        retrieval_optimization = self._optimize_retrieval_system(optimized_nodes, optimized_edges, context)
        
        # 4. 生成优化报告
        report = self._generate_optimization_report(
            original_nodes=nodes,
            original_edges=edges,
            optimized_nodes=optimized_nodes,
            optimized_edges=optimized_edges,
            retrieval_optimization=retrieval_optimization,
            start_time=start_time,
            context=context
        )
        
        # 5. 更新统计信息
        self._update_stats(report)
        
        # 6. 保存优化历史
        self.optimization_history.append(report)
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
        
        logger.info(f"Memory system optimization completed: {report.duration:.2f}s, "
                   f"quality: {report.overall_quality:.3f}")
        
        return optimized_nodes, optimized_edges, report
    
    def optimize_single_memory(self,
                              node: Dict[str, Any],
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """
        优化单个记忆
        
        Args:
            node: 记忆节点
            context: 上下文信息
            
        Returns:
            Dict[str, Any]: 优化后的记忆节点
        """
        if not self.config.memory_compression_enabled:
            return node
        
        try:
            # 提取内容
            content = node.get('content', '')
            if not content or len(content) < 50:
                return node
            
            # 优化记忆内容
            optimization_result = self.memory_optimizer.optimize_memory_content(content, context)
            
            # 创建优化后的节点
            optimized_node = node.copy()
            
            # 更新内容
            optimized_node['content'] = optimization_result['content']
            
            # 添加优化元数据
            if 'metadata' not in optimized_node:
                optimized_node['metadata'] = {}
            
            optimized_node['metadata']['optimized'] = True
            optimized_node['metadata']['optimization_result'] = optimization_result
            optimized_node['metadata']['optimized_at'] = datetime.now().isoformat()
            
            # 更新质量指标
            optimized_node['quality_metrics'] = optimization_result['quality_metrics']
            
            logger.debug(f"Optimized single memory: {node.get('node_id', 'unknown')}, "
                        f"compression: {optimization_result['optimization_report']['compression_ratio']:.2f}")
            
            return optimized_node
            
        except Exception as e:
            logger.error(f"Error optimizing single memory: {e}")
            return node
    
    def optimize_retrieval_query(self,
                                query: str,
                                nodes: List[Dict[str, Any]],
                                context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        优化检索查询
        
        Args:
            query: 查询字符串
            nodes: 记忆节点列表
            context: 上下文信息
            
        Returns:
            List[Dict[str, Any]]: 优化后的检索结果
        """
        if not self.config.retrieval_optimization_enabled:
            # 简单排序返回
            return sorted(nodes, key=lambda x: x.get('created_at', ''), reverse=True)[:20]
        
        try:
            return self.retrieval_optimizer.optimize_retrieval(query, nodes, context)
        except Exception as e:
            logger.error(f"Error optimizing retrieval query: {e}")
            # 回退到简单排序
            return sorted(nodes, key=lambda x: x.get('created_at', ''), reverse=True)[:20]
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """获取优化状态"""
        return {
            'status': 'active',
            'config': {
                'memory_compression_enabled': self.config.memory_compression_enabled,
                'association_optimization_enabled': self.config.association_optimization_enabled,
                'retrieval_optimization_enabled': self.config.retrieval_optimization_enabled,
                'optimization_interval': self.config.optimization_interval,
                'enable_realtime_optimization': self.config.enable_realtime_optimization
            },
            'stats': self.stats,
            'last_optimization': self.stats['last_optimization'],
            'optimization_history_count': len(self.optimization_history)
        }
    
    def get_recommendations(self) -> List[str]:
        """获取优化建议"""
        recommendations = []
        
        # 基于统计信息的建议
        if self.stats['total_optimizations'] == 0:
            recommendations.append("尚未进行优化，建议运行完整优化")
        
        if self.stats['avg_quality'] < 0.7:
            recommendations.append(f"平均优化质量较低({self.stats['avg_quality']:.2f})，建议调整优化参数")
        
        # 基于配置的建议
        if not self.config.memory_compression_enabled:
            recommendations.append("记忆压缩未启用，启用后可减少存储空间")
        
        if not self.config.association_optimization_enabled:
            recommendations.append("关联优化未启用，启用后可提升记忆关联质量")
        
        if not self.config.retrieval_optimization_enabled:
            recommendations.append("检索优化未启用，启用后可提升检索准确性")
        
        # 基于历史记录的建议
        if self.optimization_history:
            last_report = self.optimization_history[-1]
            if last_report.overall_quality < 0.8:
                recommendations.append(f"上次优化质量较低({last_report.overall_quality:.2f})，建议检查数据质量")
        
        return recommendations
    
    def _optimize_memory_content(self,
                                nodes: List[Dict[str, Any]],
                                context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """优化记忆内容"""
        if not self.config.memory_compression_enabled:
            return nodes
        
        optimized_nodes = []
        processed_count = 0
        
        for node in nodes:
            optimized_node = self.optimize_single_memory(node, context)
            optimized_nodes.append(optimized_node)
            processed_count += 1
            
            # 批量处理日志
            if processed_count % 100 == 0:
                logger.info(f"Memory content optimization progress: {processed_count}/{len(nodes)}")
        
        logger.info(f"Memory content optimization completed: {len(optimized_nodes)} nodes optimized")
        
        return optimized_nodes
    
    def _optimize_memory_associations(self,
                                     nodes: List[Dict[str, Any]],
                                     edges: List[Dict[str, Any]],
                                     context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """优化记忆关联"""
        if not self.config.association_optimization_enabled:
            return edges
        
        try:
            optimized_edges = self.association_optimizer.optimize_associations(nodes, edges)
            logger.info(f"Memory association optimization completed: {len(optimized_edges)} edges")
            return optimized_edges
        except Exception as e:
            logger.error(f"Error optimizing memory associations: {e}")
            return edges
    
    def _optimize_retrieval_system(self,
                                  nodes: List[Dict[str, Any]],
                                  edges: List[Dict[str, Any]],
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """优化检索系统"""
        if not self.config.retrieval_optimization_enabled:
            return {'status': 'disabled'}
        
        # 这里可以添加检索系统特定的优化
        # 例如：构建索引、训练排序模型等
        
        optimization_result = {
            'status': 'completed',
            'nodes_count': len(nodes),
            'edges_count': len(edges),
            'optimized_at': datetime.now().isoformat()
        }
        
        logger.info(f"Retrieval system optimization completed")
        
        return optimization_result
    
    def _generate_optimization_report(self,
                                     original_nodes: List[Dict[str, Any]],
                                     original_edges: List[Dict[str, Any]],
                                     optimized_nodes: List[Dict[str, Any]],
                                     optimized_edges: List[Dict[str, Any]],
                                     retrieval_optimization: Dict[str, Any],
                                     start_time: float,
                                     context: Dict[str, Any]) -> OptimizationReport:
        """生成优化报告"""
        duration = time.time() - start_time
        
        # 计算记忆优化指标
        memory_optimization = self._calculate_memory_optimization_metrics(
            original_nodes, optimized_nodes
        )
        
        # 计算关联优化指标
        association_optimization = self._calculate_association_optimization_metrics(
            original_edges, optimized_edges
        )
        
        # 计算整体质量
        overall_quality = self._calculate_overall_quality(
            memory_optimization,
            association_optimization,
            retrieval_optimization
        )
        
        # 生成建议
        recommendations = self._generate_optimization_recommendations(
            memory_optimization,
            association_optimization,
            retrieval_optimization,
            overall_quality
        )
        
        report = OptimizationReport(
            timestamp=datetime.now(),
            duration=duration,
            memory_optimization=memory_optimization,
            association_optimization=association_optimization,
            retrieval_optimization=retrieval_optimization,
            overall_quality=overall_quality,
            recommendations=recommendations
        )
        
        return report
    
    def _calculate_memory_optimization_metrics(self,
                                              original_nodes: List[Dict[str, Any]],
                                              optimized_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算记忆优化指标"""
        if not original_nodes or not optimized_nodes:
            return {'status': 'no_data'}
        
        metrics = {
            'status': 'completed',
            'original_count': len(original_nodes),
            'optimized_count': len(optimized_nodes),
            'compression_stats': self._calculate_compression_stats(optimized_nodes),
            'quality_stats': self._calculate_quality_stats(optimized_nodes)
        }
        
        # 计算压缩率
        original_size = sum(len(str(node.get('content', ''))) for node in original_nodes)
        optimized_size = sum(len(str(node.get('content', ''))) for node in optimized_nodes)
        
        if original_size > 0:
            compression_ratio = optimized_size / original_size
            metrics['compression_ratio'] = compression_ratio
            metrics['size_reduction'] = 1 - compression_ratio
        
        return metrics
    
    def _calculate_association_optimization_metrics(self,
                                                   original_edges: List[Dict[str, Any]],
                                                   optimized_edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算关联优化指标"""
        metrics = {
            'status': 'completed',
            'original_count': len(original_edges),
            'optimized_count': len(optimized_edges),
            'edge_quality_stats': self._calculate_edge_quality_stats(optimized_edges)
        }
        
        # 计算关联质量变化
        if original_edges and optimized_edges:
            original_avg_weight = np.mean([e.get('weight', 1.0) for e in original_edges])
            optimized_avg_weight = np.mean([e.get('weight', 1.0) for e in optimized_edges])
            
            metrics['avg_weight_change'] = optimized_avg_weight - original_avg_weight
            metrics['avg_weight_improvement'] = (optimized_avg_weight - original_avg_weight) / max(original_avg_weight, 0.001)
        
        return metrics
    
    def _calculate_overall_quality(self,
                                  memory_optimization: Dict[str, Any],
                                  association_optimization: Dict[str, Any],
                                  retrieval_optimization: Dict[str, Any]) -> float:
        """计算整体质量"""
        weights = {
            'memory': 0.4,
            'association': 0.35,
            'retrieval': 0.25
        }
        
        # 记忆质量
        memory_quality = memory_optimization.get('quality_stats', {}).get('avg_quality', 0.5)
        
        # 关联质量
        association_quality = association_optimization.get('edge_quality_stats', {}).get('avg_quality', 0.5)
        
        # 检索质量（默认值）
        retrieval_quality = 0.6 if retrieval_optimization.get('status') == 'completed' else 0.3
        
        overall_quality = (
            memory_quality * weights['memory'] +
            association_quality * weights['association'] +
            retrieval_quality * weights['retrieval']
        )
        
        return min(1.0, overall_quality)
    
    def _calculate_compression_stats(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算压缩统计"""
        compression_ratios = []
        
        for node in nodes:
            metadata = node.get('metadata', {})
            if 'optimization_result' in metadata:
                opt_result = metadata['optimization_result']
                compression_ratio = opt_result.get('optimization_report', {}).get('compression_ratio')
                if compression_ratio:
                    compression_ratios.append(compression_ratio)
        
        if not compression_ratios:
            return {'count': 0}
        
        return {
            'count': len(compression_ratios),
            'avg_ratio': np.mean(compression_ratios),
            'min_ratio': np.min(compression_ratios),
            'max_ratio': np.max(compression_ratios),
            'std_ratio': np.std(compression_ratios)
        }
    
    def _calculate_quality_stats(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算质量统计"""
        quality_scores = []
        
        for node in nodes:
            quality_metrics = node.get('quality_metrics')
            if quality_