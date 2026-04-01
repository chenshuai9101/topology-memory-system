"""
拓扑记忆项目 - 向量搜索模块
提供向量搜索和语义搜索功能
"""

__version__ = "0.1.0"
__author__ = "拓扑记忆开发团队"

from .models.vector_models import (
    VectorPoint, SearchQuery, HybridSearchQuery, SearchResult,
    SearchResponse, CollectionInfo, VectorIndexConfig, DistanceMetric,
    BatchUpsertRequest, DeleteRequest
)

from .services.vector_encoder import VectorEncoder, create_encoder
from .services.qdrant_service import QdrantService, get_qdrant_service
from .services.hybrid_search import HybridSearchService, create_hybrid_search_service

from .api.vector_api import router as vector_api_router

from .utils.config import get_config, validate_config, list_available_models
from .utils.helpers import (
    generate_vector_id, normalize_vector, cosine_similarity,
    extract_keywords, measure_time, format_search_results
)

__all__ = [
    # 模型
    "VectorPoint", "SearchQuery", "HybridSearchQuery", "SearchResult",
    "SearchResponse", "CollectionInfo", "VectorIndexConfig", "DistanceMetric",
    "BatchUpsertRequest", "DeleteRequest",
    
    # 服务
    "VectorEncoder", "create_encoder",
    "QdrantService", "get_qdrant_service",
    "HybridSearchService", "create_hybrid_search_service",
    
    # API
    "vector_api_router",
    
    # 工具
    "get_config", "validate_config", "list_available_models",
    "generate_vector_id", "normalize_vector", "cosine_similarity",
    "extract_keywords", "measure_time", "format_search_results"
]