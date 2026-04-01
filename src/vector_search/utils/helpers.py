"""
向量搜索工具函数
"""

import logging
import time
import hashlib
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


def generate_vector_id(text: str, prefix: str = "vec") -> str:
    """
    生成向量ID
    
    Args:
        text: 文本内容
        prefix: ID前缀
        
    Returns:
        向量ID
    """
    # 使用SHA256生成唯一ID
    hash_obj = hashlib.sha256(text.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()[:16]  # 取前16位
    
    return f"{prefix}_{hash_hex}"


def calculate_text_hash(text: str) -> str:
    """
    计算文本哈希值
    
    Args:
        text: 文本内容
        
    Returns:
        哈希值
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def normalize_vector(vector: List[float]) -> List[float]:
    """
    L2归一化向量
    
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


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    计算余弦相似度
    
    Args:
        vec1: 第一个向量
        vec2: 第二个向量
        
    Returns:
        余弦相似度 (-1 到 1)
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    计算欧氏距离
    
    Args:
        vec1: 第一个向量
        vec2: 第二个向量
        
    Returns:
        欧氏距离
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    return np.linalg.norm(v1 - v2)


def dot_product(vec1: List[float], vec2: List[float]) -> float:
    """
    计算点积
    
    Args:
        vec1: 第一个向量
        vec2: 第二个向量
        
    Returns:
        点积
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    return np.dot(v1, v2)


def similarity_to_score(similarity: float, metric: str = "cosine") -> float:
    """
    将相似度转换为0-1的分数
    
    Args:
        similarity: 相似度值
        metric: 度量标准
        
    Returns:
        0-1的分数
    """
    if metric == "cosine":
        # 余弦相似度范围是-1到1，转换为0到1
        return (similarity + 1) / 2
    elif metric == "euclidean":
        # 欧氏距离越小越相似，使用指数衰减
        return np.exp(-similarity / 10)
    else:
        # 默认处理
        return max(0.0, min(1.0, similarity))


def apply_time_decay(score: float, timestamp: datetime, decay_factor: float = 0.1) -> float:
    """
    应用时间衰减
    
    Args:
        score: 原始分数
        timestamp: 时间戳
        decay_factor: 衰减因子
        
    Returns:
        衰减后的分数
    """
    if decay_factor <= 0:
        return score
    
    current_time = datetime.now()
    time_diff = (current_time - timestamp).total_seconds() / (24 * 3600)  # 转换为天
    
    # 指数衰减
    decayed_score = score * np.exp(-decay_factor * time_diff)
    
    return decayed_score


def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """
    从文本中提取关键词（简单实现）
    
    Args:
        text: 输入文本
        max_keywords: 最大关键词数量
        
    Returns:
        关键词列表
    """
    if not text:
        return []
    
    # 简单的关键词提取：按空格分割，过滤停用词
    words = text.lower().split()
    
    # 简单停用词列表
    stop_words = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
        '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
        '自己', '这', 'the', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of',
        'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
        'can', 'could', 'may', 'might', 'must'
    }
    
    # 过滤停用词和短词
    keywords = []
    for word in words:
        word = word.strip('.,!?;:"\'()[]{}')
        if (len(word) > 1 and 
            word not in stop_words and 
            not word.isdigit()):
            keywords.append(word)
    
    # 去重并限制数量
    unique_keywords = list(dict.fromkeys(keywords))
    
    return unique_keywords[:max_keywords]


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    计算文本相似度（基于Jaccard相似度）
    
    Args:
        text1: 第一个文本
        text2: 第二个文本
        
    Returns:
        相似度分数 (0-1)
    """
    if not text1 or not text2:
        return 0.0
    
    # 分词
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # 计算Jaccard相似度
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    if union == 0:
        return 0.0
    
    return intersection / union


def batch_process(
    items: List[Any],
    batch_size: int,
    process_func,
    *args,
    **kwargs
) -> List[Any]:
    """
    批量处理项目
    
    Args:
        items: 项目列表
        batch_size: 批次大小
        process_func: 处理函数
        *args: 位置参数
        **kwargs: 关键字参数
        
    Returns:
        处理结果列表
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_func(batch, *args, **kwargs)
        results.extend(batch_results)
        
        logger.debug(f"处理批次 {i//batch_size + 1}/{(len(items)-1)//batch_size + 1}")
    
    return results


def measure_time(func):
    """
    测量函数执行时间的装饰器
    
    Args:
        func: 要测量的函数
        
    Returns:
        装饰后的函数
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = (end_time - start_time) * 1000  # 转换为毫秒
        
        logger.debug(f"函数 {func.__name__} 执行时间: {execution_time:.2f}ms")
        
        # 如果函数返回字典，添加执行时间
        if isinstance(result, dict):
            result["execution_time_ms"] = execution_time
        
        return result
    
    return wrapper


def validate_vector_dimension(vector: List[float], expected_dim: int) -> bool:
    """
    验证向量维度
    
    Args:
        vector: 向量
        expected_dim: 期望维度
        
    Returns:
        是否有效
    """
    if not vector:
        return False
    
    if len(vector) != expected_dim:
        logger.warning(f"向量维度不匹配: 期望{expected_dim}, 实际{len(vector)}")
        return False
    
    # 检查NaN或Inf
    vector_array = np.array(vector)
    if np.any(np.isnan(vector_array)) or np.any(np.isinf(vector_array)):
        logger.warning("向量包含NaN或Inf值")
        return False
    
    return True


def create_payload(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    timestamp: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    创建向量点的payload
    
    Args:
        text: 文本内容
        metadata: 元数据
        timestamp: 时间戳
        
    Returns:
        payload字典
    """
    payload = {
        "text": text,
        "text_hash": calculate_text_hash(text),
        "keywords": extract_keywords(text),
        "created_at": timestamp.isoformat() if timestamp else datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    if metadata:
        payload.update(metadata)
    
    return payload


def format_search_results(
    results: List[Dict[str, Any]],
    include_details: bool = False
) -> List[Dict[str, Any]]:
    """
    格式化搜索结果
    
    Args:
        results: 原始结果
        include_details: 是否包含详细信息
        
    Returns:
        格式化后的结果
    """
    formatted = []
    
    for i, result in enumerate(results):
        formatted_result = {
            "rank": i + 1,
            "id": result.get("id", ""),
            "score": round(result.get("score", 0), 4),
            "text": result.get("payload", {}).get("text", "")[:100] + "..." if len(
                result.get("payload", {}).get("text", "")
            ) > 100 else result.get("payload", {}).get("text", "")
        }
        
        if include_details:
            formatted_result.update({
                "payload": result.get("payload", {}),
                "metadata": result.get("metadata", {}),
                "vector_score": result.get("vector_score", 0),
                "keyword_score": result.get("keyword_score", 0)
            })
        
        formatted.append(formatted_result)
    
    return formatted


def calculate_recall_precision(
    relevant_items: List[str],
    retrieved_items: List[str],
    k: Optional[int] = None
) -> Tuple[float, float]:
    """
    计算召回率和准确率
    
    Args:
        relevant_items: 相关项目列表
        retrieved_items: 检索到的项目列表
        k: 前k个结果（如果为None，使用所有结果）
        
    Returns:
        (召回率, 准确率)
    """
    if k is not None:
        retrieved_items = retrieved_items[:k]
    
    # 转换为集合以便快速查找
    relevant_set = set(relevant_items)
    retrieved_set = set(retrieved_items)
    
    # 计算交集
    true_positives = len(relevant_set.intersection(retrieved_set))
    
    # 计算召回率
    if len(relevant_set) == 0:
        recall = 0.0
    else:
        recall = true_positives / len(relevant_set)
    
    # 计算准确率
    if len(retrieved_set) == 0:
        precision = 0.0
    else:
        precision = true_positives / len(retrieved_set)
    
    return recall, precision