"""
向量编码器服务
基于Sentence Transformers的文本向量化
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

from ..models.vector_models import DistanceMetric

logger = logging.getLogger(__name__)


class VectorEncoder:
    """向量编码器"""
    
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
        cache_folder: Optional[str] = None
    ):
        """
        初始化向量编码器
        
        Args:
            model_name: Sentence Transformers模型名称
            device: 计算设备 (cpu/cuda)
            cache_folder: 模型缓存目录
        """
        self.model_name = model_name
        self.device = device
        self.cache_folder = cache_folder
        self.model = None
        self.vector_size = 384  # 默认向量维度
        
        logger.info(f"初始化向量编码器: model={model_name}, device={device}")
        
    def load_model(self):
        """加载模型"""
        if self.model is None:
            try:
                logger.info(f"加载Sentence Transformers模型: {self.model_name}")
                self.model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    cache_folder=self.cache_folder
                )
                # 获取向量维度
                test_text = "test"
                test_vector = self.model.encode(test_text)
                self.vector_size = len(test_vector)
                logger.info(f"模型加载成功，向量维度: {self.vector_size}")
            except Exception as e:
                logger.error(f"加载模型失败: {e}")
                raise
    
    def encode_text(self, text: str) -> List[float]:
        """
        编码单个文本为向量
        
        Args:
            text: 输入文本
            
        Returns:
            向量表示
        """
        self.load_model()
        
        try:
            vector = self.model.encode(text)
            return vector.tolist()
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            raise
    
    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量编码文本为向量
        
        Args:
            texts: 输入文本列表
            
        Returns:
            向量列表
        """
        self.load_model()
        
        try:
            vectors = self.model.encode(texts)
            return vectors.tolist()
        except Exception as e:
            logger.error(f"批量文本编码失败: {e}")
            raise
    
    def encode_with_metadata(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        编码文本并包含元数据
        
        Args:
            text: 输入文本
            metadata: 元数据
            
        Returns:
            包含向量和元数据的字典
        """
        vector = self.encode_text(text)
        
        result = {
            "vector": vector,
            "text": text,
            "vector_size": self.vector_size
        }
        
        if metadata:
            result.update(metadata)
            
        return result
    
    def batch_encode_with_metadata(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        批量编码文本并包含元数据
        
        Args:
            texts: 输入文本列表
            metadata_list: 元数据列表
            
        Returns:
            包含向量和元数据的字典列表
        """
        vectors = self.encode_texts(texts)
        
        results = []
        for i, (text, vector) in enumerate(zip(texts, vectors)):
            result = {
                "vector": vector,
                "text": text,
                "vector_size": self.vector_size
            }
            
            if metadata_list and i < len(metadata_list):
                result.update(metadata_list[i])
                
            results.append(result)
            
        return results
    
    def get_vector_size(self) -> int:
        """获取向量维度"""
        if self.model is None:
            self.load_model()
        return self.vector_size
    
    def calculate_similarity(
        self,
        vector1: List[float],
        vector2: List[float],
        metric: DistanceMetric = DistanceMetric.COSINE
    ) -> float:
        """
        计算两个向量的相似度
        
        Args:
            vector1: 第一个向量
            vector2: 第二个向量
            metric: 距离度量标准
            
        Returns:
            相似度分数 (0-1之间，1表示最相似)
        """
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        if metric == DistanceMetric.COSINE:
            # 余弦相似度
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            similarity = np.dot(v1, v2) / (norm1 * norm2)
            # 归一化到0-1
            return (similarity + 1) / 2
            
        elif metric == DistanceMetric.EUCLIDEAN:
            # 欧氏距离，转换为相似度
            distance = np.linalg.norm(v1 - v2)
            # 使用指数衰减函数将距离转换为相似度
            similarity = np.exp(-distance / 10)
            return similarity
            
        elif metric == DistanceMetric.DOT:
            # 点积，归一化处理
            dot_product = np.dot(v1, v2)
            # 简单归一化
            max_dot = np.linalg.norm(v1) * np.linalg.norm(v2)
            if max_dot == 0:
                return 0.0
            return dot_product / max_dot
            
        else:
            raise ValueError(f"不支持的度量标准: {metric}")
    
    def find_most_similar(
        self,
        query_vector: List[float],
        candidate_vectors: List[List[float]],
        metric: DistanceMetric = DistanceMetric.COSINE,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        在候选向量中查找最相似的向量
        
        Args:
            query_vector: 查询向量
            candidate_vectors: 候选向量列表
            metric: 距离度量标准
            top_k: 返回最相似的数量
            
        Returns:
            相似度结果列表，包含索引和分数
        """
        similarities = []
        
        for i, candidate in enumerate(candidate_vectors):
            similarity = self.calculate_similarity(query_vector, candidate, metric)
            similarities.append((i, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top_k结果
        results = []
        for i, (idx, score) in enumerate(similarities[:top_k]):
            results.append({
                "index": idx,
                "score": score,
                "rank": i + 1
            })
            
        return results
    
    def validate_vector(self, vector: List[float]) -> bool:
        """
        验证向量是否有效
        
        Args:
            vector: 待验证的向量
            
        Returns:
            是否有效
        """
        if not vector:
            return False
            
        if len(vector) != self.vector_size:
            logger.warning(f"向量维度不匹配: 期望{self.vector_size}, 实际{len(vector)}")
            return False
            
        # 检查向量是否包含NaN或Inf
        vector_array = np.array(vector)
        if np.any(np.isnan(vector_array)) or np.any(np.isinf(vector_array)):
            logger.warning("向量包含NaN或Inf值")
            return False
            
        return True
    
    def normalize_vector(self, vector: List[float]) -> List[float]:
        """
        归一化向量（L2归一化）
        
        Args:
            vector: 输入向量
            
        Returns:
            归一化后的向量
        """
        vector_array = np.array(vector)
        norm = np.linalg.norm(vector_array)
        
        if norm == 0:
            return vector
            
        normalized = vector_array / norm
        return normalized.tolist()


# 预训练模型配置
PRETRAINED_MODELS = {
    "multilingual_mini": "paraphrase-multilingual-MiniLM-L12-v2",  # 384维，多语言
    "multilingual_base": "paraphrase-multilingual-mpnet-base-v2",  # 768维，多语言
    "chinese_mini": "distiluse-base-multilingual-cased-v1",  # 512维，中文优化
    "english_mini": "all-MiniLM-L6-v2",  # 384维，英文优化
    "universal_sentence": "all-mpnet-base-v2",  # 768维，通用句子编码
}


def create_encoder(
    model_type: str = "multilingual_mini",
    device: str = "cpu",
    cache_folder: Optional[str] = None
) -> VectorEncoder:
    """
    创建向量编码器工厂函数
    
    Args:
        model_type: 模型类型
        device: 计算设备
        cache_folder: 模型缓存目录
        
    Returns:
        向量编码器实例
    """
    model_name = PRETRAINED_MODELS.get(model_type)
    if not model_name:
        raise ValueError(f"未知的模型类型: {model_type}")
        
    return VectorEncoder(model_name, device, cache_folder)