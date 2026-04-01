"""
向量搜索API接口
提供RESTful API用于向量搜索和语义搜索
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, status
from fastapi.responses import JSONResponse

from ..models.vector_models import (
    SearchQuery, SearchResponse, HybridSearchQuery,
    CollectionInfo, VectorIndexConfig, BatchUpsertRequest,
    DeleteRequest, DistanceMetric
)
from ..services.qdrant_service import QdrantService, get_qdrant_service
from ..services.vector_encoder import VectorEncoder, create_encoder
from ..services.hybrid_search import HybridSearchService, create_hybrid_search_service

logger = logging.getLogger(__name__)

# 创建API路由器
router = APIRouter(prefix="/vector", tags=["向量搜索"])

# 依赖项
def get_vector_encoder():
    """获取向量编码器依赖项"""
    return create_encoder()

def get_hybrid_search_service(
    qdrant_service: QdrantService = Depends(get_qdrant_service),
    vector_encoder: VectorEncoder = Depends(get_vector_encoder)
):
    """获取混合搜索服务依赖项"""
    return HybridSearchService(
        qdrant_service=qdrant_service,
        vector_encoder=vector_encoder
    )


@router.get("/health", summary="健康检查")
async def health_check(
    qdrant_service: QdrantService = Depends(get_qdrant_service)
):
    """
    检查向量搜索服务健康状态
    """
    try:
        health_status = qdrant_service.health_check()
        return {
            "status": "healthy",
            "vector_search": health_status,
            "timestamp": health_status.get("timestamp")
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"向量搜索服务不可用: {str(e)}"
        )


@router.get("/collections", summary="获取所有集合")
async def get_collections(
    qdrant_service: QdrantService = Depends(get_qdrant_service)
):
    """
    获取所有向量集合的信息
    """
    try:
        qdrant_service.ensure_connected()
        collections = qdrant_service.client.get_collections()
        
        collection_infos = []
        for collection in collections.collections:
            info = qdrant_service.get_collection_info(collection.name)
            if info:
                collection_infos.append(info.dict())
        
        return {
            "collections": collection_infos,
            "total": len(collection_infos)
        }
    except Exception as e:
        logger.error(f"获取集合失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取集合失败: {str(e)}"
        )


@router.post("/collections/{collection_name}", summary="创建集合")
async def create_collection(
    collection_name: str,
    vector_size: int = Query(384, description="向量维度"),
    distance: DistanceMetric = Query(DistanceMetric.COSINE, description="距离度量标准"),
    config: Optional[VectorIndexConfig] = None,
    qdrant_service: QdrantService = Depends(get_qdrant_service)
):
    """
    创建新的向量集合
    """
    try:
        success = qdrant_service.create_collection(
            collection_name=collection_name,
            vector_size=vector_size,
            distance=distance,
            config=config
        )
        
        if success:
            return {
                "message": f"集合 {collection_name} 创建成功",
                "collection_name": collection_name,
                "vector_size": vector_size,
                "distance": distance
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"创建集合 {collection_name} 失败"
            )
    except Exception as e:
        logger.error(f"创建集合失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建集合失败: {str(e)}"
        )


@router.get("/collections/{collection_name}", summary="获取集合信息")
async def get_collection(
    collection_name: str,
    qdrant_service: QdrantService = Depends(get_qdrant_service)
):
    """
    获取指定集合的详细信息
    """
    try:
        info = qdrant_service.get_collection_info(collection_name)
        
        if info:
            return info.dict()
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"集合 {collection_name} 不存在"
            )
    except Exception as e:
        logger.error(f"获取集合信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取集合信息失败: {str(e)}"
        )


@router.post("/search", response_model=SearchResponse, summary="向量搜索")
async def vector_search(
    query: SearchQuery,
    collection_name: str = Query("topology_memory", description="集合名称"),
    qdrant_service: QdrantService = Depends(get_qdrant_service)
):
    """
    执行向量搜索
    """
    try:
        response = qdrant_service.search(query, collection_name)
        return response
    except Exception as e:
        logger.error(f"向量搜索失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"向量搜索失败: {str(e)}"
        )


@router.post("/semantic-search", response_model=SearchResponse, summary="语义搜索")
async def semantic_search(
    query_text: str = Query(..., description="查询文本"),
    limit: int = Query(10, ge=1, le=100, description="返回结果数量"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="相似度阈值"),
    collection_name: str = Query("topology_memory", description="集合名称"),
    search_service: HybridSearchService = Depends(get_hybrid_search_service)
):
    """
    执行语义搜索（基于文本的向量搜索）
    """
    try:
        response = search_service.semantic_search(
            query_text=query_text,
            limit=limit,
            threshold=threshold,
            collection_name=collection_name
        )
        return response
    except Exception as e:
        logger.error(f"语义搜索失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"语义搜索失败: {str(e)}"
        )


@router.post("/hybrid-search", response_model=SearchResponse, summary="混合搜索")
async def hybrid_search(
    query: HybridSearchQuery,
    collection_name: str = Query("topology_memory", description="集合名称"),
    search_service: HybridSearchService = Depends(get_hybrid_search_service)
):
    """
    执行混合搜索（向量+关键词）
    """
    try:
        # 暂时设置集合名称
        search_service.collection_name = collection_name
        response = search_service.hybrid_search(query)
        return response
    except Exception as e:
        logger.error(f"混合搜索失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"混合搜索失败: {str(e)}"
        )


@router.post("/encode", summary="文本向量化")
async def encode_text(
    text: str = Query(..., description="要编码的文本"),
    vector_encoder: VectorEncoder = Depends(get_vector_encoder)
):
    """
    将文本编码为向量
    """
    try:
        vector = vector_encoder.encode_text(text)
        vector_size = vector_encoder.get_vector_size()
        
        return {
            "text": text,
            "vector": vector,
            "vector_size": vector_size,
            "vector_norm": sum(x**2 for x in vector)**0.5
        }
    except Exception as e:
        logger.error(f"文本编码失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文本编码失败: {str(e)}"
        )


@router.post("/batch-encode", summary="批量文本向量化")
async def batch_encode_text(
    texts: List[str],
    vector_encoder: VectorEncoder = Depends(get_vector_encoder)
):
    """
    批量将文本编码为向量
    """
    try:
        if len(texts) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="批量编码最多支持100个文本"
            )
        
        vectors = vector_encoder.encode_texts(texts)
        
        return {
            "texts": texts,
            "vectors": vectors,
            "count": len(texts),
            "vector_size": len(vectors[0]) if vectors else 0
        }
    except Exception as e:
        logger.error(f"批量文本编码失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量文本编码失败: {str(e)}"
        )


@router.post("/points", summary="插入/更新向量点")
async def upsert_points(
    request: BatchUpsertRequest,
    qdrant_service: QdrantService = Depends(get_qdrant_service)
):
    """
    批量插入或更新向量点
    """
    try:
        success = qdrant_service.batch_upsert(request)
        
        if success:
            return {
                "message": f"成功处理 {len(request.points)} 个向量点",
                "count": len(request.points),
                "collection": request.collection_name,
                "wait": request.wait
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="插入/更新向量点失败"
            )
    except Exception as e:
        logger.error(f"插入/更新向量点失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"插入/更新向量点失败: {str(e)}"
        )


@router.delete("/points", summary="删除向量点")
async def delete_points(
    request: DeleteRequest,
    qdrant_service: QdrantService = Depends(get_qdrant_service)
):
    """
    删除向量点
    """
    try:
        success = qdrant_service.delete_points(request)
        
        if success:
            return {
                "message": f"成功删除 {len(request.ids)} 个向量点",
                "count": len(request.ids),
                "collection": request.collection_name
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="删除向量点失败"
            )
    except Exception as e:
        logger.error(f"删除向量点失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除向量点失败: {str(e)}"
        )


@router.get("/points/{point_id}", summary="获取向量点")
async def get_point(
    point_id: str,
    collection_name: str = Query("topology_memory", description="集合名称"),
    with_payload: bool = Query(True, description="是否包含payload"),
    with_vector: bool = Query(False, description="是否包含向量"),
    qdrant_service: QdrantService = Depends(get_qdrant_service)
):
    """
    获取指定向量点的详细信息
    """
    try:
        point = qdrant_service.get_point(
            collection_name=collection_name,
            point_id=point_id,
            with_payload=with_payload,
            with_vector=with_vector
        )
        
        if point:
            return point.dict()
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"向量点 {point_id} 不存在"
            )
    except Exception as e:
        logger.error(f"获取向量点失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取向量点失败: {str(e)}"
        )


@router.get("/similarity", summary="计算相似度")
async def calculate_similarity(
    vector1: List[float],
    vector2: List[float],
    metric: DistanceMetric = Query(DistanceMetric.COSINE, description="距离度量标准"),
    vector_encoder: VectorEncoder = Depends(get_vector_encoder)
):
    """
    计算两个向量的相似度
    """
    try:
        if len(vector1) != len(vector2):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="向量维度不匹配"
            )
        
        similarity = vector_encoder.calculate_similarity(vector1, vector2, metric)
        
        return {
            "vector1_dim": len(vector1),
            "vector2_dim": len(vector2),
            "metric": metric,
            "similarity": similarity,
            "distance": 1.0 - similarity if metric == DistanceMetric.COSINE else None
        }
    except Exception as e:
        logger.error(f"计算相似度失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"计算相似度失败: {str(e)}"
        )


@router.get("/performance", summary="性能指标")
async def get_performance(
    qdrant_service: QdrantService = Depends(get_qdrant_service)
):
    """
    获取向量搜索性能指标
    """
    try:
        # 这里可以添加更复杂的性能监控逻辑
        health_status = qdrant_service.health_check()
        
        return {
            "status": "healthy",
            "metrics": {
                "collection_count": health_status.get("total_collections", 0),
                "response_time_target": "< 100ms",
                "accuracy_target": "> 90%",
                "throughput_target": "> 1000 req/sec"
            },
            "timestamp": health_status.get("timestamp")
        }
    except Exception as e:
        logger.error(f"获取性能指标失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取性能指标失败: {str(e)}"
        )


# 错误处理
@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all():
    """捕获未定义的路由"""
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="API端点不存在"
    )