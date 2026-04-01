"""
拓扑记忆上下文管理器 - FastAPI应用
提供REST API接口
"""

import time
import uuid
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Query, Path, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
import loguru

from .schemas import (
    ContextCreate, ContextUpdate, ContextResponse,
    TopologyQuery, TopologyResponse, PerformanceMetrics,
    HealthCheckResponse, ErrorResponse, PaginationParams, PaginatedResponse
)
from ..core.context_manager import ContextManager
from ..integration.topology_adapter import TopologyAdapter
from ..performance.cache_manager import CacheManager


# 配置日志
logger = loguru.logger
logger.add("logs/api.log", rotation="500 MB", retention="10 days", level="INFO")


# 全局实例
context_manager = None
topology_adapter = None
cache_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global context_manager, topology_adapter, cache_manager
    
    # 启动时初始化
    logger.info("Starting Topology Memory Context Manager API...")
    
    # 初始化组件
    context_manager = ContextManager(max_contexts_per_session=100)
    topology_adapter = TopologyAdapter()
    cache_manager = CacheManager(max_size=1000, ttl=300)
    
    logger.info("All components initialized successfully")
    
    yield
    
    # 关闭时清理
    logger.info("Shutting down Topology Memory Context Manager API...")
    
    if cache_manager:
        cache_manager.clear()
    
    logger.info("API shutdown complete")


# 创建FastAPI应用
app = FastAPI(
    title="拓扑记忆上下文管理器 API",
    description="基于拓扑学的多维记忆管理系统",
    version="0.1.0",
    docs_url=None,  # 自定义文档
    redoc_url="/redoc",
    lifespan=lifespan
)


# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 依赖项
def get_context_manager() -> ContextManager:
    """获取上下文管理器实例"""
    if context_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Context manager not initialized"
        )
    return context_manager


def get_topology_adapter() -> TopologyAdapter:
    """获取拓扑适配器实例"""
    if topology_adapter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Topology adapter not initialized"
        )
    return topology_adapter


def get_cache_manager() -> CacheManager:
    """获取缓存管理器实例"""
    if cache_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache manager not initialized"
        )
    return cache_manager


# 自定义文档页面
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """自定义Swagger UI"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
    )


# 中间件：请求日志和性能监控
@app.middleware("http")
async def add_process_time_header(request, call_next):
    """添加处理时间头并记录日志"""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # 添加请求ID到请求状态
    request.state.request_id = request_id
    
    # 处理请求
    response = await call_next(request)
    
    # 计算处理时间
    process_time = time.time() - start_time
    
    # 添加头信息
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request_id
    
    # 记录性能指标
    if cache_manager:
        metrics = PerformanceMetrics(
            request_id=request_id,
            endpoint=request.url.path,
            method=request.method,
            response_time=process_time,
            status_code=response.status_code,
            timestamp=datetime.now()
        )
        cache_manager.set(f"metrics:{request_id}", metrics.dict(), ttl=3600)
    
    # 记录访问日志
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s - "
        f"Request-ID: {request_id}"
    )
    
    return response


# 异常处理器
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理器"""
    logger.error(f"HTTP Exception: {exc.detail} - Status: {exc.status_code}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            request_id=getattr(request.state, "request_id", None)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理器"""
    logger.exception(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            request_id=getattr(request.state, "request_id", None)
        ).dict()
    )


# 健康检查端点
@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="健康检查",
    description="检查服务及其依赖的健康状态"
)
async def health_check(
    cm: ContextManager = Depends(get_context_manager),
    ta: TopologyAdapter = Depends(get_topology_adapter),
    cache: CacheManager = Depends(get_cache_manager)
):
    """健康检查端点"""
    dependencies = {
        "context_manager": "healthy",
        "topology_adapter": "healthy" if ta.is_connected() else "unhealthy",
        "cache_manager": "healthy"
    }
    
    # 检查所有依赖
    all_healthy = all(status == "healthy" for status in dependencies.values())
    
    return HealthCheckResponse(
        status="healthy" if all_healthy else "degraded",
        version="0.1.0",
        timestamp=datetime.now(),
        dependencies=dependencies
    )


# 上下文管理端点
@app.post(
    "/contexts",
    response_model=ContextResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建上下文",
    description="创建新的上下文条目"
)
async def create_context(
    context_data: ContextCreate,
    cm: ContextManager = Depends(get_context_manager)
):
    """创建新的上下文"""
    try:
        context = cm.create_context(context_data)
        logger.info(f"Created context {context.id} for session {context.session_id}")
        return context
    except Exception as e:
        logger.error(f"Failed to create context: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create context: {str(e)}"
        )


@app.get(
    "/contexts/{session_id}/{context_id}",
    response_model=ContextResponse,
    summary="获取上下文",
    description="根据会话ID和上下文ID获取上下文"
)
async def get_context(
    session_id: str = Path(..., description="会话ID"),
    context_id: str = Path(..., description="上下文ID"),
    cm: ContextManager = Depends(get_context_manager)
):
    """获取上下文"""
    context = cm.get_context(session_id, context_id)
    if not context:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Context {context_id} not found in session {session_id}"
        )
    
    logger.debug(f"Retrieved context {context_id} from session {session_id}")
    return context


@app.put(
    "/contexts/{session_id}/{context_id}",
    response_model=ContextResponse,
    summary="更新上下文",
    description="更新现有上下文"
)
async def update_context(
    session_id: str = Path(..., description="会话ID"),
    context_id: str = Path(..., description="上下文ID"),
    update_data: ContextUpdate = ...,
    cm: ContextManager = Depends(get_context_manager)
):
    """更新上下文"""
    context = cm.update_context(session_id, context_id, update_data)
    if not context:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Context {context_id} not found in session {session_id}"
        )
    
    logger.info(f"Updated context {context_id} in session {session_id}")
    return context


@app.delete(
    "/contexts/{session_id}/{context_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除上下文",
    description="删除指定上下文"
)
async def delete_context(
    session_id: str = Path(..., description="会话ID"),
    context_id: str = Path(..., description="上下文ID"),
    cm: ContextManager = Depends(get_context_manager)
):
    """删除上下文"""
    success = cm.delete_context(session_id, context_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Context {context_id} not found in session {session_id}"
        )
    
    logger.info(f"Deleted context {context_id} from session {session_id}")


@app.get(
    "/contexts/{session_id}",
    response_model=PaginatedResponse,
    summary="列出会话上下文",
    description="列出会话中的所有上下文（分页）"
)
async def list_contexts(
    session_id: str = Path(..., description="会话ID"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    cm: ContextManager = Depends(get_context_manager)
):
    """列出会话上下文"""
    contexts, total = cm.list_contexts(session_id, page, page_size)
    
    total_pages = (total + page_size - 1) // page_size
    
    return PaginatedResponse(
        items=contexts,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages
    )


@app.get(
    "/sessions/{session_id}/contexts",
    response_model=List[ContextResponse],
    summary="获取会话所有上下文",
    description="获取会话的所有上下文（不分页）"
)
async def get_session_contexts(
    session_id: str = Path(..., description="会话ID"),
    cm: ContextManager = Depends(get_context_manager)
):
    """获取会话所有上下文"""
    contexts = cm.get_session_contexts(session_id)
    return contexts


@app.get(
    "/users/{user_id}/contexts",
    response_model=List[ContextResponse],
    summary="获取用户所有上下文",
    description="获取用户的所有上下文"
)
async def get_user_contexts(
    user_id: str = Path(..., description="用户ID"),
    cm: ContextManager = Depends(get_context_manager)
):
    """获取用户所有上下文"""
    contexts = cm.get_user_contexts(user_id)
    return contexts


@app.post(
    "/contexts/search",
    response_model=List[ContextResponse],
    summary="搜索上下文",
    description="根据查询条件搜索上下文"
)
async def search_contexts(
    query: dict,
    session_id: Optional[str] = Query(None, description="可选的会话ID限制"),
    cm: ContextManager = Depends(get_context_manager)
):
    """搜索上下文"""
    contexts = cm.search_contexts(query, session_id)
    return contexts


# 拓扑记忆集成端点
@app.post(
    "/topology/query",
    response_model=TopologyResponse,
    summary="拓扑查询",
    description="查询拓扑记忆系统"
)
async def query_topology(
    query: TopologyQuery,
    ta: TopologyAdapter = Depends(get_topology_adapter),
    cache: CacheManager = Depends(get_cache_manager)
):
    """查询拓扑记忆系统"""
    # 检查缓存
    cache_key = f"topology_query:{hash(str(query.dict()))}"
    cached_result = cache.get(cache_key)
    
    if cached_result:
        logger.debug(f"Cache hit for topology query: {query.query}")
        return TopologyResponse(**cached_result)
    
    # 执行查询
    try:
        result = ta.query_memory(query)
        
        # 缓存结果
        cache.set(cache_key, result.dict(), ttl=60)  # 缓存60秒
        
        logger.info(f"Topology query executed: {query.query} - Found {len(result.nodes)} nodes")
        return result
    except Exception as e:
        logger.error(f"Topology query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Topology query failed: {str(e)}"
        )


@app.get(
    "/topology/nodes/{node_id}",
    response_model=dict,
    summary="获取记忆节点",
    description="获取拓扑记忆系统中的特定节点"
)
async def get_memory_node(
    node_id: str = Path(..., description="节点ID"),
    ta: TopologyAdapter = Depends(get_topology_adapter),
    cache: CacheManager = Depends(get_cache_manager)
):
    """获取记忆节点"""
    # 检查缓存
    cache_key = f"memory_node:{node_id}"
    cached_node = cache.get(cache_key)
    
    if cached_node:
        logger.debug(f"Cache hit for memory node: {node_id}")
        return cached_node
    
    # 获取节点
    try:
        node = ta.get_node(node_id)
        
        if not node:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory node {node_id} not found"
            )
        
        # 缓存节点
        cache.set(cache_key, node, ttl=300)  # 缓存5分钟
        
        return node
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get memory node {node_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get memory node: {str(e)}"
        )


# 系统管理端点
@app.get(
    "/stats",
    response_model=dict,
    summary="系统统计",
    description="获取系统统计信息"
)
async def get_stats(
    cm: ContextManager = Depends(get_context_manager),
    cache: CacheManager = Depends(get_cache_manager)
):
    """获取系统统计信息"""
    stats = cm.get_stats()
    cache_stats = cache.get_stats()
    
    return {
        "context_manager": stats,
        "cache_manager": cache_stats,
        "timestamp": datetime.now().isoformat()
    }


@app.post(
    "/cache/clear",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="清理缓存",
    description="清理所有缓存"
)
async def clear_cache(cache: CacheManager = Depends(get_cache_manager)):
    """清理缓存"""
    cache.clear()
    logger.info("Cache cleared")


@app.post(
    "/contexts/clear",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="清理所有上下文",
    description="清理所有上下文（仅用于测试）"
)
async def clear_all_contexts(cm: ContextManager = Depends(get_context_manager)):
    """清理所有上下文"""
    cm.clear_all()
    logger.info("All contexts cleared")


# 根端点
@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "欢迎使用拓扑记忆上下文管理器 API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "openapi": "/openapi.json"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "context_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )