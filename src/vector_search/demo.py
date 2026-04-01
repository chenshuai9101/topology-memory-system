"""
向量搜索演示脚本
展示向量搜索功能的使用
"""

import logging
import sys
import os
from typing import List, Dict, Any
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.vector_search import (
    VectorEncoder, QdrantService, HybridSearchService,
    VectorPoint, SearchQuery, HybridSearchQuery,
    generate_vector_id, extract_keywords
)


class VectorSearchDemo:
    """向量搜索演示类"""
    
    def __init__(self):
        """初始化演示"""
        logger.info("初始化向量搜索演示...")
        
        # 初始化服务
        self.vector_encoder = VectorEncoder(
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
            device="cpu"
        )
        
        self.qdrant_service = QdrantService(
            host="localhost",
            port=6333
        )
        
        self.hybrid_search_service = HybridSearchService(
            qdrant_service=self.qdrant_service,
            vector_encoder=self.vector_encoder
        )
        
        # 测试数据
        self.test_documents = [
            "人工智能是计算机科学的一个分支，旨在创造能够执行通常需要人类智能的任务的机器。",
            "机器学习是人工智能的一个子领域，使计算机能够从数据中学习而无需明确编程。",
            "深度学习是机器学习的一个分支，使用神经网络模拟人脑的工作方式。",
            "自然语言处理是人工智能的一个领域，专注于计算机与人类语言之间的交互。",
            "计算机视觉是人工智能的一个领域，使计算机能够从数字图像或视频中获取信息。",
            "强化学习是机器学习的一种类型，智能体通过与环境交互来学习最优行为策略。",
            "神经网络是受人脑启发的计算模型，用于识别数据中的模式和关系。",
            "大数据是指传统数据处理应用软件无法处理的庞大而复杂的数据集。",
            "云计算是通过互联网提供计算服务，包括服务器、存储、数据库、网络等。",
            "物联网是指通过互联网连接的物理设备网络，能够收集和交换数据。"
        ]
    
    def run_demo(self):
        """运行完整演示"""
        logger.info("开始向量搜索演示...")
        
        try:
            # 1. 连接Qdrant
            self._connect_qdrant()
            
            # 2. 创建集合
            self._create_collection()
            
            # 3. 编码和存储文档
            self._encode_and_store_documents()
            
            # 4. 向量搜索演示
            self._demo_vector_search()
            
            # 5. 语义搜索演示
            self._demo_semantic_search()
            
            # 6. 混合搜索演示
            self._demo_hybrid_search()
            
            # 7. 性能测试
            self._demo_performance()
            
            logger.info("向量搜索演示完成！")
            
        except Exception as e:
            logger.error(f"演示过程中出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _connect_qdrant(self):
        """连接Qdrant"""
        logger.info("1. 连接Qdrant向量数据库...")
        
        try:
            self.qdrant_service.connect()
            logger.info("✓ Qdrant连接成功")
        except Exception as e:
            logger.error(f"✗ Qdrant连接失败: {e}")
            raise
    
    def _create_collection(self):
        """创建集合"""
        logger.info("2. 创建向量集合...")
        
        try:
            success = self.qdrant_service.create_collection(
                collection_name="demo_collection",
                vector_size=384,  # 模型向量维度
                distance="Cosine"
            )
            
            if success:
                logger.info("✓ 向量集合创建成功")
            else:
                logger.info("ℹ 向量集合可能已存在")
        except Exception as e:
            logger.error(f"✗ 创建集合失败: {e}")
            raise
    
    def _encode_and_store_documents(self):
        """编码和存储文档"""
        logger.info("3. 编码和存储测试文档...")
        
        # 编码文档
        logger.info(f"编码 {len(self.test_documents)} 个文档...")
        vectors = self.vector_encoder.encode_texts(self.test_documents)
        
        # 创建向量点
        points = []
        for i, (text, vector) in enumerate(zip(self.test_documents, vectors)):
            point = VectorPoint(
                id=f"doc_{i+1:03d}",
                vector=vector,
                payload={
                    "text": text,
                    "doc_id": i+1,
                    "category": "AI/ML",
                    "timestamp": datetime.now().isoformat(),
                    "keywords": extract_keywords(text)
                }
            )
            points.append(point)
        
        # 存储到Qdrant
        logger.info("存储向量点到数据库...")
        success = self.qdrant_service.upsert_points(
            collection_name="demo_collection",
            points=points,
            wait=True
        )
        
        if success:
            logger.info(f"✓ 成功存储 {len(points)} 个文档")
        else:
            logger.error("✗ 存储文档失败")
            raise Exception("存储文档失败")
    
    def _demo_vector_search(self):
        """演示向量搜索"""
        logger.info("4. 向量搜索演示...")
        
        # 编码查询文本
        query_text = "人工智能和机器学习的关系"
        query_vector = self.vector_encoder.encode_text(query_text)
        
        # 创建搜索查询
        search_query = SearchQuery(
            query_vector=query_vector,
            query_text=query_text,
            limit=5,
            threshold=0.3,
            with_payload=True,
            with_vector=False
        )
        
        # 执行搜索
        logger.info(f"查询: '{query_text}'")
        response = self.qdrant_service.search(search_query, "demo_collection")
        
        # 显示结果
        self._display_search_results(response, "向量搜索")
    
    def _demo_semantic_search(self):
        """演示语义搜索"""
        logger.info("5. 语义搜索演示...")
        
        query_text = "如何让计算机理解人类语言"
        
        logger.info(f"查询: '{query_text}'")
        response = self.hybrid_search_service.semantic_search(
            query_text=query_text,
            limit=5,
            threshold=0.3,
            collection_name="demo_collection"
        )
        
        # 显示结果
        self._display_search_results(response, "语义搜索")
    
    def _demo_hybrid_search(self):
        """演示混合搜索"""
        logger.info("6. 混合搜索演示...")
        
        query_text = "神经网络在图像识别中的应用"
        keywords = extract_keywords(query_text)
        
        # 创建混合搜索查询
        hybrid_query = HybridSearchQuery(
            query_text=query_text,
            keywords=keywords,
            limit=5,
            vector_weight=0.7,
            keyword_weight=0.3,
            time_decay_factor=0.1,
            importance_weight=1.0
        )
        
        logger.info(f"查询: '{query_text}'")
        logger.info(f"提取的关键词: {keywords}")
        
        response = self.hybrid_search_service.hybrid_search(hybrid_query)
        
        # 显示结果
        self._display_search_results(response, "混合搜索")
    
    def _demo_performance(self):
        """演示性能"""
        logger.info("7. 性能演示...")
        
        # 测试查询
        test_queries = [
            "人工智能",
            "机器学习",
            "深度学习",
            "自然语言处理"
        ]
        
        total_time = 0
        total_results = 0
        
        for query in test_queries:
            response = self.hybrid_search_service.semantic_search(
                query_text=query,
                limit=3,
                threshold=0.3
            )
            
            total_time += response.query_time_ms
            total_results += len(response.results)
            
            logger.info(f"查询 '{query}': {response.query_time_ms:.2f}ms, 找到 {len(response.results)} 个结果")
        
        avg_time = total_time / len(test_queries)
        
        logger.info(f"平均查询时间: {avg_time:.2f}ms")
        logger.info(f"总找到结果: {total_results}")
        
        # 检查性能指标
        if avg_time < 100:
            logger.info("✓ 性能达标: 平均响应时间 < 100ms")
        else:
            logger.warning("⚠ 性能警告: 平均响应时间 > 100ms")
    
    def _display_search_results(self, response, search_type: str):
        """显示搜索结果"""
        logger.info(f"=== {search_type} 结果 ===")
        logger.info(f"查询耗时: {response.query_time_ms:.2f}ms")
        logger.info(f"找到结果: {len(response.results)} 个")
        
        for i, result in enumerate(response.results[:3]):  # 只显示前3个
            text = result.payload.get("text", "")[:80] + "..." if len(
                result.payload.get("text", "")
            ) > 80 else result.payload.get("text", "")
            
            logger.info(f"{i+1}. [分数: {result.score:.3f}] {text}")
        
        if len(response.results) > 3:
            logger.info(f"... 还有 {len(response.results) - 3} 个结果")
        
        logger.info("")
    
    def cleanup(self):
        """清理演示数据"""
        logger.info("清理演示数据...")
        
        try:
            # 这里可以添加清理逻辑
            # 例如删除测试集合
            pass
        except Exception as e:
            logger.warning(f"清理过程中出错: {e}")


def main():
    """主函数"""
    demo = VectorSearchDemo()
    
    try:
        demo.run_demo()
    finally:
        demo.cleanup()


if __name__ == "__main__":
    main()