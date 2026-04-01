"""
分层索引结构 - AlphaGo启发式优化
L0: 摘要索引 (55KB) - 所有节点
L1: 任务类型索引 (10KB) - 按任务分类
L2: 时间窗口索引 (5KB) - 最近7天/30天
L3: 相关性热索引 (2KB) - 高频访问节点
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
import os

class LayeredIndex:
    """分层索引管理器"""
    
    def __init__(self, base_index_path: str):
        """
        初始化分层索引
        
        Args:
            base_index_path: 基础索引文件路径
        """
        self.base_index_path = base_index_path
        self.layers = {
            'L0': None,  # 摘要索引
            'L1': None,  # 任务类型索引
            'L2': None,  # 时间窗口索引
            'L3': None   # 相关性热索引
        }
        
        # 任务类型分类
        self.task_categories = {
            'coding': ['技术', '代码', 'API', '开发', '编程', '调试'],
            'reporting': ['报告', '汇报', '总结', '进度', '状态'],
            'planning': ['计划', '规划', '安排', '时间线', '里程碑'],
            'research': ['研究', '分析', '调查', '探索', '实验'],
            'communication': ['沟通', '会议', '讨论', '邮件', '聊天']
        }
        
        # 加载基础索引
        self._load_base_index()
        
    def _load_base_index(self):
        """加载基础索引"""
        try:
            # 尝试多种编码方式
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(self.base_index_path, 'r', encoding=encoding) as f:
                        content = f.read()
                        self.layers['L0'] = json.loads(content)
                    print(f"✅ 加载基础索引成功 (编码: {encoding}): {len(self.layers['L0'])} 个节点")
                    return
                except UnicodeDecodeError:
                    continue
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON解析失败 (编码: {encoding}): {e}")
                    continue
            
            # 如果所有编码都失败，尝试二进制读取
            with open(self.base_index_path, 'rb') as f:
                binary_content = f.read()
                # 尝试解码为字符串
                try:
                    decoded = binary_content.decode('utf-8', errors='ignore')
                    self.layers['L0'] = json.loads(decoded)
                    print(f"✅ 加载基础索引成功 (二进制解码): {len(self.layers['L0'])} 个节点")
                except:
                    # 最后尝试：创建模拟数据用于测试
                    print("⚠️ 无法加载真实索引，创建模拟数据用于测试")
                    self.layers['L0'] = self._create_mock_index()
                    
        except Exception as e:
            print(f"❌ 加载基础索引失败: {e}")
            self.layers['L0'] = self._create_mock_index()
    
    def _create_mock_index(self):
        """创建模拟索引数据用于测试"""
        mock_nodes = []
        task_types = ['coding', 'reporting', 'planning', 'research', 'communication']
        
        for i in range(50):
            task_type = task_types[i % len(task_types)]
            mock_nodes.append({
                'id': f'node_{i:03d}',
                'summary': f'这是一个{task_type}相关的节点，包含重要信息{i}',
                'created_at': f'2026-03-{31-i%30:02d}T{10+i%10:02d}:00:00Z',
                'content': f'这是节点{i}的详细内容，涉及{task_type}相关的工作。'
            })
        
        return mock_nodes
    
    def build_l1_task_index(self):
        """构建L1任务类型索引"""
        print("🔧 构建L1任务类型索引...")
        
        task_index = defaultdict(list)
        
        for node in self.layers['L0']:
            node_id = node.get('id', '')
            summary = node.get('summary', '')
            
            # 根据摘要内容分类
            for task_type, keywords in self.task_categories.items():
                if any(keyword in summary for keyword in keywords):
                    task_index[task_type].append({
                        'node_id': node_id,
                        'summary': summary,
                        'relevance': self._calculate_relevance(summary, keywords)
                    })
        
        self.layers['L1'] = dict(task_index)
        print(f"✅ L1索引构建完成: {len(task_index)} 个任务类型")
        
        # 保存到文件
        self._save_layer('L1', 'task_type_index.json')
    
    def build_l2_time_index(self, days: int = 7):
        """构建L2时间窗口索引"""
        print(f"🔧 构建L2时间窗口索引 (最近{days}天)...")
        
        time_index = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for node in self.layers['L0']:
            node_id = node.get('id', '')
            created_str = node.get('created_at', '')
            
            try:
                # 解析创建时间
                if created_str:
                    created_date = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                    if created_date > cutoff_date:
                        time_index.append({
                            'node_id': node_id,
                            'summary': node.get('summary', ''),
                            'created_at': created_str,
                            'recency_score': self._calculate_recency_score(created_date)
                        })
            except:
                pass
        
        self.layers['L2'] = time_index
        print(f"✅ L2索引构建完成: {len(time_index)} 个近期节点")
        
        # 保存到文件
        self._save_layer('L2', f'time_index_{days}d.json')
    
    def build_l3_hot_index(self, top_n: int = 20):
        """构建L3相关性热索引"""
        print(f"🔧 构建L3相关性热索引 (Top {top_n})...")
        
        # 模拟访问频率数据（实际应从访问日志获取）
        hot_nodes = []
        
        # 这里简单按节点ID排序取前N个作为热节点
        for i, node in enumerate(self.layers['L0'][:top_n]):
            hot_nodes.append({
                'node_id': node.get('id', ''),
                'summary': node.get('summary', ''),
                'hot_score': 1.0 - (i * 0.05),  # 递减热度
                'category': self._categorize_node(node.get('summary', ''))
            })
        
        self.layers['L3'] = hot_nodes
        print(f"✅ L3索引构建完成: {len(hot_nodes)} 个热节点")
        
        # 保存到文件
        self._save_layer('L3', 'hot_index.json')
    
    def query_by_task(self, task_type: str, limit: int = 10) -> List[Dict]:
        """按任务类型查询"""
        if not self.layers['L1']:
            self.build_l1_task_index()
        
        nodes = self.layers['L1'].get(task_type, [])
        # 按相关性排序
        nodes.sort(key=lambda x: x['relevance'], reverse=True)
        return nodes[:limit]
    
    def query_recent(self, limit: int = 10) -> List[Dict]:
        """查询最近节点"""
        if not self.layers['L2']:
            self.build_l2_time_index()
        
        nodes = self.layers['L2']
        # 按时间排序
        nodes.sort(key=lambda x: x['recency_score'], reverse=True)
        return nodes[:limit]
    
    def query_hot(self, limit: int = 10) -> List[Dict]:
        """查询热节点"""
        if not self.layers['L3']:
            self.build_l3_hot_index()
        
        nodes = self.layers['L3']
        # 按热度排序
        nodes.sort(key=lambda x: x['hot_score'], reverse=True)
        return nodes[:limit]
    
    def _calculate_relevance(self, text: str, keywords: List[str]) -> float:
        """计算文本与关键词的相关性"""
        if not text or not keywords:
            return 0.0
        
        matches = sum(1 for keyword in keywords if keyword in text)
        return min(matches / len(keywords), 1.0)
    
    def _calculate_recency_score(self, date: datetime) -> float:
        """计算时间新鲜度分数"""
        now = datetime.now()
        days_diff = (now - date).days
        
        if days_diff <= 1:
            return 1.0
        elif days_diff <= 7:
            return 0.7
        elif days_diff <= 30:
            return 0.3
        else:
            return 0.1
    
    def _categorize_node(self, summary: str) -> str:
        """分类节点"""
        for task_type, keywords in self.task_categories.items():
            if any(keyword in summary for keyword in keywords):
                return task_type
        return 'other'
    
    def _save_layer(self, layer_name: str, filename: str):
        """保存层级索引到文件"""
        layer_dir = os.path.join(os.path.dirname(self.base_index_path), 'layers')
        os.makedirs(layer_dir, exist_ok=True)
        
        filepath = os.path.join(layer_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.layers[layer_name], f, ensure_ascii=False, indent=2)
        
        print(f"💾 保存{layer_name}索引到: {filepath}")

# 使用示例
if __name__ == "__main__":
    # 初始化分层索引
    base_index = "/Users/muyunye/.openclaw/workspace/memory/topology/index.json"
    layered_index = LayeredIndex(base_index)
    
    # 构建所有层级索引
    layered_index.build_l1_task_index()
    layered_index.build_l2_time_index(days=7)
    layered_index.build_l3_hot_index(top_n=20)
    
    # 测试查询
    print("\n🧪 测试查询:")
    print("1. 按任务类型查询 (coding):")
    coding_nodes = layered_index.query_by_task('coding', limit=3)
    for node in coding_nodes:
        print(f"   - {node['node_id']}: {node['summary'][:50]}...")
    
    print("\n2. 查询最近节点:")
    recent_nodes = layered_index.query_recent(limit=3)
    for node in recent_nodes:
        print(f"   - {node['node_id']}: {node['summary'][:50]}...")
    
    print("\n3. 查询热节点:")
    hot_nodes = layered_index.query_hot(limit=3)
    for node in hot_nodes:
        print(f"   - {node['node_id']}: {node['summary'][:50]}...")
