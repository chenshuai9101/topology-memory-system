"""
向量搜索数据模型
定义向量存储和搜索相关的数据结构
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class DistanceMetric(str, Enum):
    """向量距离度量标准"""
    COSINE = "Cosine"
    EUCLIDEAN = "Euclid"
    DOT = "Dot"


class VectorPoint(BaseModel):
    """向量点数据模型"""
    id: str = Field(..., description="向量点ID")
    vector: List[float] = Field(..., description="向量数据")
    payload: Dict[str, Any] = Field(default_factory=dict, description="附加数据")
    score: Optional[float] = Field(None, description="相似度分数")


class SearchQuery(BaseModel):
    """向量搜索查询"""
    query_vector: Optional[List[float]] = Field(None, description="查询向量")
    query_text: Optional[str] = Field(None, description="查询文本")
    limit: int = Field(10, ge=1, le=100, description="返回结果数量")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="相似度阈值")
    distance_metric: DistanceMetric = Field(DistanceMetric.COSINE, description="距离度量标准")
    with_payload: bool = Field(True, description="是否返回payload")
    with_vector: bool = Field(False, description="是否返回向量")
    filter_conditions: Optional[Dict[str, Any]] = Field(None, description="过滤条件")


class HybridSearchQuery(BaseModel):
    """混合搜索查询（向量+关键词）"""
    query_text: str = Field(..., description="查询文本")
    keywords: Optional[List[str]] = Field(None, description="关键词列表")
    limit: int = Field(10, ge=1, le=100, description="返回结果数量")
    vector_weight: float = Field(0.7, ge=0.0, le=1.0, description="向量搜索权重")
    keyword_weight: float = Field(0.3, ge=0.0, le=1.0, description="关键词搜索权重")
    time_decay_factor: float = Field(0.1, ge=0.0, le=1.0, description="时间衰减因子")
    importance_weight: float = Field(1.0, ge=0.0, description="重要性权重")
    filter_conditions: Optional[Dict[str, Any]] = Field(None, description="过滤条件")


class SearchResult(BaseModel):
    """搜索结果"""
    id: str = Field(..., description="结果ID")
    score: float = Field(..., ge=0.0, le=1.0, description="相似度分数")
    payload: Dict[str, Any] = Field(default_factory=dict, description="附加数据")
    vector: Optional[List[float]] = Field(None, description="向量数据")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class SearchResponse(BaseModel):
    """搜索响应"""
    results: List[SearchResult] = Field(default_factory=list, description="搜索结果列表")
    total: int = Field(0, description="总结果数量")
    query_time_ms: float = Field(0.0, description="查询耗时(毫秒)")
    query_type: str = Field("vector", description="查询类型")


class CollectionInfo(BaseModel):
    """向量集合信息"""
    name: str = Field(..., description="集合名称")
    vector_size: int = Field(..., description="向量维度")
    distance_metric: DistanceMetric = Field(..., description="距离度量标准")
    points_count: int = Field(0, description="向量点数量")
    segments_count: int = Field(0, description="段数量")
    config: Dict[str, Any] = Field(default_factory=dict, description="配置信息")
    status: str = Field("green", description="集合状态")


class VectorIndexConfig(BaseModel):
    """向量索引配置"""
    hnsw_m: int = Field(16, ge=4, le=64, description="HNSW M参数")
    hnsw_ef_construct: int = Field(100, ge=10, le=500, description="HNSW EF构造参数")
    hnsw_ef_search: int = Field(100, ge=10, le=500, description="HNSW EF搜索参数")
    optimizers_default_segment_number: int = Field(2, ge=1, le=10, description="默认段数量")
    optimizers_max_segment_size: int = Field(50000, ge=1000, description="最大段大小")
    wal_capacity_mb: int = Field(32, ge=1, description="WAL容量(MB)")


class BatchUpsertRequest(BaseModel):
    """批量插入/更新请求"""
    points: List[VectorPoint] = Field(..., description="向量点列表")
    collection_name: str = Field("topology_memory", description="集合名称")
    wait: bool = Field(True, description="是否等待操作完成")


class DeleteRequest(BaseModel):
    """删除请求"""
    ids: List[str] = Field(..., description="要删除的ID列表")
    collection_name: str = Field("topology_memory", description="集合名称")
    wait: bool = Field(True, description="是否等待操作完成")


class SimilarityMetrics(BaseModel):
    """相似度度量结果"""
    cosine_similarity: float = Field(..., ge=-1.0, le=1.0, description="余弦相似度")
    euclidean_distance: float = Field(..., ge=0.0, description="欧氏距离")
    dot_product: float = Field(..., description="点积")
    normalized_score: float = Field(..., ge=0.0, le=1.0, description="归一化分数")


class SearchPerformance(BaseModel):
    """搜索性能指标"""
    response_time_ms: float = Field(..., description="响应时间(毫秒)")
    recall_rate: float = Field(..., ge=0.0, le=1.0, description="召回率")
    precision_rate: float = Field(..., ge=0.0, le=1.0, description="准确率")
    throughput_qps: float = Field(..., description="吞吐量(QPS)")
    memory_usage_mb: float = Field(..., description="内存使用(MB)")