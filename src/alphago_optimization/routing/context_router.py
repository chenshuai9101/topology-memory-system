"""
轻量级上下文路由引擎 - AlphaGo启发式优化
基于规则的路由，快速判断哪些数据源最相关
"""

import re
from typing import Dict, List, Tuple, Any
from collections import defaultdict

class ContextRouter:
    """上下文路由器 - 轻量级策略网络实现"""
    
    def __init__(self, template_cache=None):
        """
        初始化路由器
        
        Args:
            template_cache: 任务模板缓存实例
        """
        self.template_cache = template_cache
        
        # 关键词到任务类型的映射
        self.keyword_to_task = {
            # 代码开发
            '代码': 'coding', '编程': 'coding', '开发': 'coding', 
            '调试': 'coding', 'API': 'coding', '函数': 'coding',
            '类': 'coding', '模块': 'coding', '库': 'coding',
            '错误': 'coding', '异常': 'coding', '测试': 'coding',
            
            # 项目汇报
            '报告': 'reporting', '汇报': 'reporting', '总结': 'reporting',
            '进度': 'reporting', '状态': 'reporting', '完成': 'reporting',
            '里程碑': 'reporting', '成果': 'reporting', '业绩': 'reporting',
            
            # 项目规划
            '计划': 'planning', '规划': 'planning', '安排': 'planning',
            '时间线': 'planning', '日程': 'planning', '排期': 'planning',
            '资源': 'planning', '预算': 'planning', '分配': 'planning',
            
            # 技术研究
            '研究': 'research', '分析': 'research', '调研': 'research',
            '实验': 'research', '验证': 'research', '方案': 'research',
            '对比': 'research', '评估': 'research', '探索': 'research',
            
            # 沟通协调
            '会议': 'communication', '讨论': 'communication', '沟通': 'communication',
            '邮件': 'communication', '聊天': 'communication', '协调': 'communication',
            '通知': 'communication', '反馈': 'communication', '确认': 'communication'
        }
        
        # 数据源类型定义
        self.data_sources = {
            '技术文档': ['文档', '说明', '指南', '手册', '规范'],
            '代码片段': ['代码', '示例', '片段', '实现', '源码'],
            'API参考': ['API', '接口', '端点', '参数', '响应'],
            '错误日志': ['错误', '异常', '故障', '问题', 'bug'],
            '调试记录': ['调试', '排查', '诊断', '修复', '解决'],
            '进度数据': ['进度', '状态', '完成', '百分比', '里程碑'],
            '会议记录': ['会议', '讨论', '纪要', '决议', '行动项'],
            '待办事项': ['待办', '任务', '事项', 'TODO', '计划'],
            '项目计划': ['计划', '规划', '方案', '路线图', '时间线'],
            '资源分配': ['资源', '人员', '预算', '分配', '安排'],
            '研究论文': ['论文', '研究', '文献', '引用', '理论'],
            '技术分析': ['分析', '评估', '对比', '优缺点', '选择'],
            '实验数据': ['数据', '实验', '结果', '统计', '指标']
        }
        
        # 初始化路由缓存
        self.route_cache = {}
    
    def route_query(self, query: str, task_intent: str = None) -> Dict[str, float]:
        """
        路由查询到数据源
        
        Args:
            query: 查询文本
            task_intent: 任务意图（可选）
            
        Returns:
            数据源概率分布字典
        """
        # 检查缓存
        cache_key = f"{query}_{task_intent}"
        if cache_key in self.route_cache:
            return self.route_cache[cache_key]
        
        # 1. 识别任务类型
        detected_task = self._detect_task_type(query, task_intent)
        
        # 2. 获取任务模板（如果有缓存）
        if self.template_cache:
            try:
                template = self.template_cache.get_template(detected_task)
                # 使用模板中的优先级
                source_probs = template.get('priority_sources', {}).copy()
            except:
                source_probs = self._calculate_source_probabilities(query, detected_task)
        else:
            source_probs = self._calculate_source_probabilities(query, detected_task)
        
        # 3. 基于查询内容微调概率
        source_probs = self._adjust_by_query_content(query, source_probs)
        
        # 4. 归一化概率
        source_probs = self._normalize_probabilities(source_probs)
        
        # 缓存结果
        self.route_cache[cache_key] = source_probs
        
        return source_probs
    
    def get_top_sources(self, query: str, task_intent: str = None, top_n: int = 3) -> List[Tuple[str, float]]:
        """
        获取Top N数据源
        
        Args:
            query: 查询文本
            task_intent: 任务意图
            top_n: 返回的数据源数量
            
        Returns:
            Top N数据源列表（数据源，概率）
        """
        source_probs = self.route_query(query, task_intent)
        
        # 排序并取前N个
        sorted_sources = sorted(source_probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_sources[:top_n]
    
    def _detect_task_type(self, query: str, explicit_intent: str = None) -> str:
        """
        检测任务类型
        
        Args:
            query: 查询文本
            explicit_intent: 显式指定的任务意图
            
        Returns:
            检测到的任务类型
        """
        # 如果有显式意图，优先使用
        if explicit_intent and explicit_intent in ['coding', 'reporting', 'planning', 'research', 'communication']:
            return explicit_intent
        
        # 基于关键词检测
        task_scores = defaultdict(float)
        
        for keyword, task_type in self.keyword_to_task.items():
            if keyword.lower() in query.lower():
                task_scores[task_type] += 1.0
        
        # 返回得分最高的任务类型
        if task_scores:
            best_task = max(task_scores.items(), key=lambda x: x[1])[0]
            return best_task
        
        # 默认返回通用任务
        return 'coding'  # 默认代码开发任务
    
    def _calculate_source_probabilities(self, query: str, task_type: str) -> Dict[str, float]:
        """
        计算数据源概率
        
        Args:
            query: 查询文本
            task_type: 任务类型
            
        Returns:
            数据源概率字典
        """
        source_probs = {}
        
        # 基于任务类型的基准概率
        base_probabilities = {
            'coding': {
                '技术文档': 0.8, '代码片段': 0.7, 'API参考': 0.6,
                '错误日志': 0.5, '调试记录': 0.4
            },
            'reporting': {
                '进度数据': 0.9, '会议记录': 0.8, '待办事项': 0.7,
                '项目计划': 0.5, '资源分配': 0.4
            },
            'planning': {
                '项目计划': 0.9, '资源分配': 0.8, '时间线': 0.7,
                '预算数据': 0.6, '团队信息': 0.5
            },
            'research': {
                '研究论文': 0.9, '技术分析': 0.8, '实验数据': 0.7,
                '对比报告': 0.6, '专家意见': 0.5
            },
            'communication': {
                '会议记录': 0.9, '邮件往来': 0.8, '聊天记录': 0.7,
                '通知公告': 0.6, '反馈意见': 0.5
            }
        }
        
        # 获取基准概率
        base_probs = base_probabilities.get(task_type, base_probabilities['coding'])
        source_probs = base_probs.copy()
        
        return source_probs
    
    def _adjust_by_query_content(self, query: str, source_probs: Dict[str, float]) -> Dict[str, float]:
        """
        基于查询内容调整概率
        
        Args:
            query: 查询文本
            source_probs: 原始概率字典
            
        Returns:
            调整后的概率字典
        """
        adjusted_probs = source_probs.copy()
        query_lower = query.lower()
        
        # 检查查询中的特定关键词
        for source, keywords in self.data_sources.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    # 如果查询中提到该数据源的关键词，提高概率
                    if source in adjusted_probs:
                        adjusted_probs[source] = min(adjusted_probs[source] * 1.5, 1.0)
                    else:
                        adjusted_probs[source] = 0.7
        
        return adjusted_probs
    
    def _normalize_probabilities(self, source_probs: Dict[str, float]) -> Dict[str, float]:
        """
        归一化概率
        
        Args:
            source_probs: 原始概率字典
            
        Returns:
            归一化后的概率字典
        """
        if not source_probs:
            return {}
        
        # 确保所有概率在0-1之间
        normalized = {}
        for source, prob in source_probs.items():
            normalized[source] = max(0.0, min(1.0, prob))
        
        # 可选：归一化到总和为1
        # total = sum(normalized.values())
        # if total > 0:
        #     normalized = {k: v/total for k, v in normalized.items()}
        
        return normalized
    
    def clear_cache(self):
        """清空路由缓存"""
        self.route_cache.clear()
        print("🧹 路由缓存已清空")

# 使用示例
if __name__ == "__main__":
    # 初始化路由器
    router = ContextRouter()
    
    # 测试查询
    test_queries = [
        "如何优化FAISS索引性能？",
        "项目本周进度汇报需要哪些数据？",
        "下季度研发资源如何分配？",
        "BERT和GPT模型哪个更适合文本分类？"
    ]
    
    print("🧪 上下文路由测试:")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\n📝 查询: {query}")
        
        # 路由查询
        source_probs = router.route_query(query)
        
        # 获取Top 3数据源
        top_sources = router.get_top_sources(query, top_n=3)
        
        print(f"   🎯 检测任务类型: {router._detect_task_type(query)}")
        print(f"   📊 Top 3数据源:")
        for source, prob in top_sources:
            print(f"      - {source}: {prob:.2f}")
