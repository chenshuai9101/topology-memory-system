"""
任务类型模板缓存 - AlphaGo启发式优化
为高频任务建立热启动上下文模板
"""

from typing import Dict, List, Any
import json
import os

class TaskTemplateCache:
    """任务类型模板缓存管理器"""
    
    def __init__(self, cache_dir: str = None):
        """
        初始化模板缓存
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), 'templates')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 预定义任务模板
        self.predefined_templates = {
            'coding': {
                'name': '代码开发',
                'description': '技术开发、编程、调试相关任务',
                'context_sources': [
                    '技术文档',
                    'API参考',
                    '代码片段',
                    '错误日志',
                    '调试记录'
                ],
                'priority_sources': {
                    '技术文档': 0.9,
                    '代码片段': 0.8,
                    'API参考': 0.7,
                    '错误日志': 0.5,
                    '调试记录': 0.4
                },
                'cache_key': 'coding_template'
            },
            'reporting': {
                'name': '项目汇报',
                'description': '项目状态汇报、进度总结相关任务',
                'context_sources': [
                    '进度数据',
                    '会议记录',
                    '待办事项',
                    '里程碑',
                    '风险记录'
                ],
                'priority_sources': {
                    '进度数据': 0.9,
                    '会议记录': 0.8,
                    '待办事项': 0.7,
                    '里程碑': 0.6,
                    '风险记录': 0.5
                },
                'cache_key': 'reporting_template'
            },
            'planning': {
                'name': '项目规划',
                'description': '项目计划、资源分配、时间线规划',
                'context_sources': [
                    '项目计划',
                    '资源分配',
                    '时间线',
                    '预算数据',
                    '团队信息'
                ],
                'priority_sources': {
                    '项目计划': 0.9,
                    '时间线': 0.8,
                    '资源分配': 0.7,
                    '预算数据': 0.6,
                    '团队信息': 0.5
                },
                'cache_key': 'planning_template'
            },
            'research': {
                'name': '技术研究',
                'description': '技术调研、方案分析、实验验证',
                'context_sources': [
                    '研究论文',
                    '技术分析',
                    '实验数据',
                    '对比报告',
                    '专家意见'
                ],
                'priority_sources': {
                    '研究论文': 0.9,
                    '技术分析': 0.8,
                    '实验数据': 0.7,
                    '对比报告': 0.6,
                    '专家意见': 0.5
                },
                'cache_key': 'research_template'
            }
        }
        
        # 加载缓存模板
        self.loaded_templates = {}
        self._load_cached_templates()
    
    def get_template(self, task_type: str) -> Dict[str, Any]:
        """
        获取任务模板
        
        Args:
            task_type: 任务类型
            
        Returns:
            任务模板字典
        """
        # 首先检查内存缓存
        if task_type in self.loaded_templates:
            return self.loaded_templates[task_type]
        
        # 检查预定义模板
        if task_type in self.predefined_templates:
            template = self.predefined_templates[task_type]
            # 缓存到内存
            self.loaded_templates[task_type] = template
            return template
        
        # 返回默认模板
        return self._get_default_template()
    
    def preload_for_task(self, task_type: str) -> List[str]:
        """
        根据任务类型预加载上下文
        
        Args:
            task_type: 任务类型
            
        Returns:
            预加载的上下文源列表
        """
        template = self.get_template(task_type)
        return template.get('context_sources', [])
    
    def get_source_priority(self, task_type: str, source: str) -> float:
        """
        获取数据源优先级
        
        Args:
            task_type: 任务类型
            source: 数据源名称
            
        Returns:
            优先级分数 (0.0-1.0)
        """
        template = self.get_template(task_type)
        priority_map = template.get('priority_sources', {})
        return priority_map.get(source, 0.3)  # 默认优先级
    
    def warmup_cache(self, task_types: List[str] = None):
        """
        预热缓存
        
        Args:
            task_types: 要预热的任务类型列表，None表示所有类型
        """
        if task_types is None:
            task_types = list(self.predefined_templates.keys())
        
        print(f"🔥 预热任务模板缓存: {len(task_types)} 个任务类型")
        
        for task_type in task_types:
            if task_type in self.predefined_templates:
                self.loaded_templates[task_type] = self.predefined_templates[task_type]
                print(f"  ✅ 预热: {task_type}")
        
        print(f"✅ 缓存预热完成: {len(self.loaded_templates)} 个模板已加载")
    
    def save_template(self, task_type: str, template: Dict[str, Any]):
        """
        保存自定义模板
        
        Args:
            task_type: 任务类型
            template: 模板数据
        """
        cache_file = os.path.join(self.cache_dir, f"{task_type}_template.json")
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
        
        # 更新内存缓存
        self.loaded_templates[task_type] = template
        
        print(f"💾 保存模板: {task_type} -> {cache_file}")
    
    def _load_cached_templates(self):
        """加载缓存的模板文件"""
        if not os.path.exists(self.cache_dir):
            return
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('_template.json'):
                task_type = filename.replace('_template.json', '')
                cache_file = os.path.join(self.cache_dir, filename)
                
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        template = json.load(f)
                    self.loaded_templates[task_type] = template
                    print(f"📂 加载缓存模板: {task_type}")
                except Exception as e:
                    print(f"❌ 加载模板失败 {filename}: {e}")
    
    def _get_default_template(self) -> Dict[str, Any]:
        """获取默认模板"""
        return {
            'name': '通用任务',
            'description': '通用任务模板',
            'context_sources': [
                '通用文档',
                '相关记录',
                '历史数据'
            ],
            'priority_sources': {
                '通用文档': 0.5,
                '相关记录': 0.4,
                '历史数据': 0.3
            },
            'cache_key': 'default_template'
        }

# 使用示例
if __name__ == "__main__":
    # 初始化模板缓存
    cache = TaskTemplateCache()
    
    # 预热缓存
    cache.warmup_cache(['coding', 'reporting', 'planning'])
    
    # 测试获取模板
    print("\n🧪 测试模板获取:")
    test_tasks = ['coding', 'reporting', 'unknown']
    
    for task in test_tasks:
        template = cache.get_template(task)
        print(f"\n📋 任务类型: {task}")
        print(f"   名称: {template['name']}")
        print(f"   描述: {template['description']}")
        print(f"   上下文源: {', '.join(template['context_sources'][:3])}...")
        
        # 测试预加载
        preloaded = cache.preload_for_task(task)
        print(f"   预加载: {len(preloaded)} 个源")
        
        # 测试优先级
        sample_source = template['context_sources'][0] if template['context_sources'] else '通用文档'
        priority = cache.get_source_priority(task, sample_source)
        print(f"   示例源 '{sample_source}' 优先级: {priority:.2f}")
