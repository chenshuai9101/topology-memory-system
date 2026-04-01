"""
Qdrant向量数据库服务
提供向量存储和搜索功能
"""

import logging
import time
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from ..models.vector_models import (
    VectorPoint, SearchQuery, SearchResult, SearchResponse,
    CollectionInfo, VectorIndexConfig, DistanceMetric,
    BatchUpsertRequest, DeleteRequest
)

logger = logging.getLogger(__name__)


class QdrantService:
    """Qdrant向量数据库服务"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        api_key: Optional[str] = None,
        timeout: int = 30,
        prefer_grpc: bool = False
    ):
        """
        初始化Qdrant服务
        
        Args:
            host: Qdrant主机地址
            port: REST API端口
            grpc_port: gRPC端口
            api_key: API密钥
            timeout: 超时时间(秒)
            prefer_grpc: 是否优先使用gRPC
        """
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.api_key = api_key
        self.timeout = timeout
        self.prefer_grpc = prefer_grpc
        self.client = None
        
        logger.info(f"初始化Qdrant服务: host={host}:{port}, grpc_port={grpc_port}")
        
    def connect(self):
        """连接到Qdrant服务"""
        if self.client is None:
            try:
                # 构建连接URL
                url = f"http://{self.host}:{self.port}"
                
                # 创建客户端
                self.client = QdrantClient(
                    url=url,
                    port=self.grpc_port if self.prefer_grpc else None,
                    api_key=self.api_key,
                    timeout=self.timeout,
                    prefer_grpc=self.prefer_grpc
                )
                
                # 测试连接
                self.client.get_collections()
                logger.info(f"Qdrant连接成功: {url}")
                
            except Exception as e:
                logger.error(f"Qdrant连接失败: {e}")
                raise
    
    def ensure_connected(self):
        """确保已连接"""
        if self.client is None:
            self.connect()
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: DistanceMetric = DistanceMetric.COSINE,
        config: Optional[VectorIndexConfig] = None
    ) -> bool:
        """
        创建向量集合
        
        Args:
            collection_name: 集合名称
            vector_size: 向量维度
            distance: 距离度量标准
            config: 索引配置
            
        Returns:
            是否创建成功
        """
        self.ensure_connected()
        
        try:
            # 检查集合是否已存在
            collections = self.client.get_collections()
            for collection in collections.collections:
                if collection.name == collection_name:
                    logger.info(f"集合已存在: {collection_name}")
                    return True
            
            # 使用默认配置或提供的配置
            if config is None:
                config = VectorIndexConfig()
            
            # 创建集合
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=self._convert_distance(distance)
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=config.hnsw_m,
                    ef_construct=config.hnsw_ef_construct
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=config.optimizers_default_segment_number,
                    max_segment_size=config.optimizers_max_segment_size
                ),
                wal_config=models.WalConfigDiff(
                    wal_capacity_mb=config.wal_capacity_mb
                )
            )
            
            logger.info(f"集合创建成功: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Optional[CollectionInfo]:
        """
        获取集合信息
        
        Args:
            collection_name: 集合名称
            
        Returns:
            集合信息
        """
        self.ensure_connected()
        
        try:
            collection_info = self.client.get_collection(collection_name)
            
            return CollectionInfo(
                name=collection_name,
                vector_size=collection_info.config.params.vectors.size,
                distance_metric=self._parse_distance(
                    collection_info.config.params.vectors.distance
                ),
                points_count=collection_info.points_count,
                segments_count=collection_info.segments_count,
                config=collection_info.config.dict(),
                status=collection_info.status
            )
            
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return None
    
    def upsert_points(
        self,
        collection_name: str,
        points: List[VectorPoint],
        wait: bool = True
    ) -> bool:
        """
        插入或更新向量点
        
        Args:
            collection_name: 集合名称
            points: 向量点列表
            wait: 是否等待操作完成
            
        Returns:
            是否成功
        """
        self.ensure_connected()
        
        try:
            # 转换向量点为Qdrant格式
            qdrant_points = []
            for point in points:
                qdrant_point = models.PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=point.payload
                )
                qdrant_points.append(qdrant_point)
            
            # 执行插入/更新
            self.client.upsert(
                collection_name=collection_name,
                points=qdrant_points,
                wait=wait
            )
            
            logger.info(f"成功插入/更新 {len(points)} 个向量点到集合 {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"插入/更新向量点失败: {e}")
            return False
    
    def batch_upsert(self, request: BatchUpsertRequest) -> bool:
        """
        批量插入/更新向量点
        
        Args:
            request: 批量插入请求
            
        Returns:
            是否成功
        """
        return self.upsert_points(
            collection_name=request.collection_name,
            points=request.points,
            wait=request.wait
        )
    
    def delete_points(self, request: DeleteRequest) -> bool:
        """
        删除向量点
        
        Args:
            request: 删除请求
            
        Returns:
            是否成功
        """
        self.ensure_connected()
        
        try:
            self.client.delete(
                collection_name=request.collection_name,
                points_selector=models.PointIdsList(
                    points=request.ids
                ),
                wait=request.wait
            )
            
            logger.info(f"成功删除 {len(request.ids)} 个向量点")
            return True
            
        except Exception as e:
            logger.error(f"删除向量点失败: {e}")
            return False
    
    def search(
        self,
        query: SearchQuery,
        collection_name: str = "topology_memory"
    ) -> SearchResponse:
        """
        向量搜索
        
        Args:
            query: 搜索查询
            collection_name: 集合名称
            
        Returns:
            搜索结果
        """
        self.ensure_connected()
        
        start_time = time.time()
        
        try:
            # 构建搜索参数
            search_params = models.SearchParams(
                hnsw_ef=query.limit * 2  # 优化搜索精度
            )
            
            # 构建过滤条件
            filter_condition = None
            if query.filter_conditions:
                filter_condition = self._build_filter(query.filter_conditions)
            
            # 执行搜索
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query.query_vector,
                query_filter=filter_condition,
                limit=query.limit,
                score_threshold=query.threshold,
                with_payload=query.with_payload,
                with_vectors=query.with_vector,
                search_params=search_params
            )
            
            # 转换结果
            results = []
            for hit in search_results:
                result = SearchResult(
                    id=str(hit.id),
                    score=hit.score,
                    payload=hit.payload or {},
                    vector=hit.vector if query.with_vector else None,
                    metadata={
                        "version": hit.version,
                        "collection": collection_name
                    }
                )
                results.append(result)
            
            query_time_ms = (time.time() - start_time) * 1000
            
            return SearchResponse(
                results=results,
                total=len(results),
                query_time_ms=query_time_ms,
                query_type="vector"
            )
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return SearchResponse(
                results=[],
                total=0,
                query_time_ms=(time.time() - start_time) * 1000,
                query_type="vector"
            )
    
    def scroll_points(
        self,
        collection_name: str,
        limit: int = 100,
        offset: Optional[str] = None,
        with_payload: bool = True,
        with_vector: bool = False
    ) -> Dict[str, Any]:
        """
        滚动获取向量点
        
        Args:
            collection_name: 集合名称
            limit: 每次获取数量
            offset: 偏移量
            with_payload: 是否包含payload
            with_vector: 是否包含向量
            
        Returns:
            向量点列表和下一个偏移量
        """
        self.ensure_connected()
        
        try:
            scroll_result = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=None,
                limit=limit,
                offset=offset,
                with_payload=with_payload,
                with_vectors=with_vector
            )
            
            points = []
            for record in scroll_result[0]:
                point = VectorPoint(
                    id=str(record.id),
                    vector=record.vector if with_vector else [],
                    payload=record.payload or {},
                    score=None
                )
                points.append(point)
            
            return {
                "points": points,
                "next_offset": scroll_result[1],
                "total": len(points)
            }
            
        except Exception as e:
            logger.error(f"滚动获取向量点失败: {e}")
            return {"points": [], "next_offset": None, "total": 0}
    
    def get_point(
        self,
        collection_name: str,
        point_id: str,
        with_payload: bool = True,
        with_vector: bool = False
    ) -> Optional[VectorPoint]:
        """
        获取单个向量点
        
        Args:
            collection_name: 集合名称
            point_id: 向量点ID
            with_payload: 是否包含payload
            with_vector: 是否包含向量
            
        Returns:
            向量点
        """
        self.ensure_connected()
        
        try:
            records = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=with_payload,
                with_vectors=with_vector
            )
            
            if records and len(records) > 0:
                record = records[0]
                return VectorPoint(
                    id=str(record.id),
                    vector=record.vector if with_vector else [],
                    payload=record.payload or {},
                    score=None
                )
            
            return None
            
        except Exception as e:
            logger.error(f"获取向量点失败: {e}")
            return None
    
    def update_payload(
        self,
        collection_name: str,
        point_id: str,
        payload: Dict[str, Any],
        wait: bool = True
    ) -> bool:
        """
        更新向量点的payload
        
        Args:
            collection_name: 集合名称
            point_id: 向量点ID
            payload: 新的payload
            wait: 是否等待操作完成
            
        Returns:
            是否成功
        """
        self.ensure_connected()
        
        try:
            self.client.set_payload(
                collection_name=collection_name,
                payload=payload,
                points=[point_id],
                wait=wait
            )
            
            logger.info(f"成功更新向量点 {point_id} 的payload")
            return True
            
        except Exception as e:
            logger.error(f"更新payload失败: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态信息
        """
        self.ensure_connected()
        
        try:
            # 获取服务状态
            collections = self.client.get_collections()
            
            # 检查每个集合的状态
            collection_status = {}
            for collection in collections.collections:
                info = self.get_collection_info(collection.name)
                if info:
                    collection_status[collection.name] = {
                        "status": info.status,
                        "points_count": info.points_count,
                        "vector_size": info.vector_size
                    }
            
            return {
                "status": "healthy",
                "collections": collection_status,
                "total_collections": len(collections.collections),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _convert_distance(self, distance: DistanceMetric) -> models.Distance:
        """转换距离度量标准"""
        if distance == DistanceMetric.COSINE:
            return models.Distance.COSINE
        elif distance == DistanceMetric.EUCLIDEAN:
            return models.Distance.EUCLID
        elif distance == DistanceMetric.DOT:
            return models.Distance.DOT
        else:
            return models.Distance.COSINE
    
    def _parse_distance(self, distance: models.Distance) -> DistanceMetric:
        """解析距离度量标准"""
        if distance == models.Distance.COSINE:
            return DistanceMetric.COSINE
        elif distance == models.Distance.EUCLID:
            return DistanceMetric.EUCLIDEAN
        elif distance == models.Distance.DOT:
            return DistanceMetric.DOT
        else:
            return DistanceMetric.COSINE
    
    def _build_filter(self, conditions: Dict[str, Any]) -> Optional[models.Filter]:
        """构建过滤条件"""
        if not conditions:
            return None
        
        filter_conditions = []
        
        for key, value in conditions.items():
            if isinstance(value, (list, tuple)):
                # 列表条件
                filter_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value)
                    )
                )
            elif isinstance(value, dict):
                # 范围条件
                if "gte" in value or "lte" in value:
                    range_dict = {}
                    if "gte" in value:
                        range_dict["gte"] = value["gte"]
                    if "lte" in value:
                        range_dict["lte"] = value["lte"]
                    
                    filter_conditions.append(
                        models.FieldCondition(
                            key=key,
                            range=models.Range(**range_dict)
                        )
                    )
            else:
                # 精确匹配
                filter_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
        
        if filter_conditions:
            return models.Filter(must=filter_conditions)
        
        return None
    
    def close(self):
        """关闭连接"""
        if self.client:
            try:
                # Qdrant客户端没有显式的close方法
                self.client = None
                logger.info("Qdrant连接已关闭")
            except Exception as e:
                logger.error(f"关闭连接失败: {e}")


# 单例模式
_qdrant_service_instance = None

def get_qdrant_service(
    host: str = "localhost",
    port: int = 6333,
    grpc_port: int = 6334,
    api_key: Optional[str] = None,
    timeout: int = 30,
    prefer_grpc: bool = False
) -> QdrantService:
    """
    获取Qdrant服务实例（单例模式）
    
    Args:
        host: Qdrant主机地址
        port: REST API端口
        grpc_port: gRPC端口
        api_key: API密钥
        timeout: 超时时间(秒)
        prefer_grpc: 是否优先使用gRPC
        
    Returns:
        Qdrant服务实例
    """
    global _qdrant_service_instance
    
    if _qdrant_service_instance is None:
        _qdrant_service_instance = QdrantService(
            host=host,
            port=port,
            grpc_port=grpc_port,
            api_key=api_key,
            timeout=timeout,
            prefer_grpc=prefer_grpc
        )
    
    return _qdrant_service_instance