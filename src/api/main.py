"""
拓扑记忆上下文管理器 - 主API应用
整合所有模块，提供完整的REST API服务
"""

import time
import uuid
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Query, Path, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
import loguru

from .schemas import (
    ContextCreate, ContextUpdate, ContextResponse,
    TopologyQuery, TopologyResponse, PerformanceMetrics,
    HealthCheckResponse, ErrorResponse, PaginationParams, PaginatedResponse,
    UserCreate, UserResponse, Token, LoginRequest, RefreshTokenRequest,
    APIVersion
)
from .dependencies import get_settings, get_auth_service, AuthService, PermissionChecker
from .versioning import (
    create_versioned_router, version_manager, create_version_endpoints,
    version_middleware, create_versioned_docs
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


# 创建版本化路由器
v1_router = create_versioned_router(prefix="/v1")
v2_router = create_versioned_router(prefix="/v2")

# 创建FastAPI应用
app = FastAPI(
    title="拓扑记忆上下文管理器 API",
    description="基于拓扑学的多维记忆管理系统",
    version="0.1.0",
    docs_url=None,  # 自定义文档
    redoc_url=None,
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "认证",
            "description": "用户认证和授权相关操作"
        },
        {
            "name": "上下文管理",
            "description": "上下文创建、读取、更新、删除操作"
        },
        {
            "name": "拓扑记忆",
            "description": "拓扑记忆系统集成操作"
        },
        {
            "name": "系统管理",
            "description": "系统统计、缓存管理等操作"
        },
        {
            "name": "API信息",
            "description": "API版本和状态信息"
        }
    ]
)


# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加版本中间件
app.middleware("http")(version_middleware)


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


# 中间件：请求日志和性能监控
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
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
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理器"""
    logger.error(f"HTTP Exception: {exc.detail} - Status: {exc.status_code}")
    
    # 添加错误ID
    error_id = str(uuid.uuid4())
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=f"Error ID: {error_id}",
            request_id=getattr(request.state, "request_id", None)
        ).dict(),
        headers={
            "X-Error-ID": error_id,
            "X-Error-Type": "http_exception"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理器"""
    error_id = str(uuid.uuid4())
    logger.exception(f"Unhandled exception (ID: {error_id}): {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=f"Error ID: {error_id}. Please contact support.",
            request_id=getattr(request.state, "request_id", None)
        ).dict(),
        headers={
            "X-Error-ID": error_id,
            "X-Error-Type": "internal_error"
        }
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception):
    """404异常处理器"""
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error="Resource not found",
            detail=f"The requested resource {request.url.path} was not found",
            request_id=getattr(request.state, "request_id", None)
        ).dict()
    )


# 根端点
@app.get("/")
async def root():
    """根端点 - 重定向到默认版本文档"""
    return RedirectResponse(url="/v1/docs")


@app.get("/health")
async def health_check():
    """健康检查端点（无版本）"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# 认证端点（v1版本）
@v1_router.post(
    "/auth/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["认证"],
    summary="用户注册",
    description="注册新用户账户"
)
async def register_user(
    user_data: UserCreate,
    auth_service: AuthService = Depends(get_auth_service)
):
    """用户注册"""
    # 在实际应用中，这里应该将用户保存到数据库
    # 这里简化为直接返回用户信息
    return UserResponse(
        id=str(uuid.uuid4()),
        username=user_data.username,
        email=user_data.email,
        is_active=True,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )


@v1_router.post(
    "/auth/login",
    response_model=Token,
    tags=["认证"],
    summary="用户登录",
    description="用户登录并获取访问令牌"
)
async def login(
    login_data: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """用户登录"""
    # 在实际应用中，这里应该验证用户名和密码
    # 这里简化为直接生成令牌
    
    user_data = {
        "sub": str(uuid.uuid4()),  # 用户ID
        "username": login_data.username,
        "roles": ["user"],
        "permissions": ["read", "write"]
    }
    
    access_token = auth_service.create_access_token(user_data)
    refresh_token = auth_service.create_refresh_token(user_data)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=1800  # 30分钟
    )


@v1_router.post(
    "/auth/refresh",
    response_model=Token,
    tags=["认证"],
    summary="刷新令牌",
    description="使用刷新令牌获取新的访问令牌"
)
async def refresh_token(
    refresh_data: RefreshTokenRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """刷新令牌"""
    # 验证刷新令牌
    payload = auth_service.verify_token(refresh_data.refresh_token)
    
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # 创建新的访问令牌
    user_data = {
        "sub": payload.get("sub"),
        "username": payload.get("username"),
        "roles": payload.get("roles", []),
        "permissions": payload.get("permissions", [])
    }
    
    access_token = auth_service.create_access_token(user_data)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_data.refresh_token,  # 刷新令牌不变
        token_type="bearer",
        expires_in=1800
    )


# 受保护的上下文管理端点（v1版本）
@v1_router.post(
    "/contexts",
    response_model=ContextResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["上下文管理"],
    summary="创建上下文",
    description="创建新的上下文条目（需要认证）",
    dependencies=[Depends(PermissionChecker(["write"]))]
)
async def create_context(
    context_data: ContextCreate,
    current_user = Depends(get_auth_service().get_current_user),
    cm: ContextManager = Depends(get_context_manager)
):
    """创建新的上下文"""
    try:
        # 确保用户ID匹配
        context_data.user_id = current_user.user_id
        
        context = cm.create_context(context_data)
        logger.info(f"Created context {context.id} for session {context.session_id}")
        return context
    except Exception as e:
        logger.error(f"Failed to create context: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create context: {str(e)}"
        )


@v1_router.get(
    "/contexts/{session_id}/{context_id}",
    response_model=ContextResponse,
    tags=["上下文管理"],
    summary="获取上下文",
    description="根据会话ID和上下文ID获取上下文（需要认证）",
    dependencies=[Depends(PermissionChecker(["read"]))]
)
async def get_context(
    session_id: str = Path(..., description="会话ID"),
    context_id: str = Path(..., description="上下文ID"),
    current_user = Depends(get_auth_service().get_current_user),
    cm: ContextManager = Depends(get_context_manager)
):
    """获取上下文"""
    context = cm.get_context(session_id, context_id)
    if not context:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Context {context_id} not found in session {session_id}"
        )
    
    # 检查权限：用户只能访问自己的上下文
    if context.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    logger.debug(f"Retrieved context {context_id} from session {session_id}")
    return context


@v1_router.put(
    "/contexts/{session_id}/{context_id}",
    response_model=ContextResponse,
    tags=["上下文管理"],
    summary="更新上下文",
    description="更新现有上下文（需要认证）",
    dependencies=[Depends(PermissionChecker(["write"]))]
)
async def update_context(
    session_id: str = Path(..., description="会话ID"),
    context_id: str = Path(..., description="上下文ID"),
    update_data: ContextUpdate = ...,
    current_user = Depends(get_auth_service().get_current_user),
    cm: ContextManager = Depends(get_context_manager)
):
    """更新上下文"""
    # 先获取上下文以检查权限
    context = cm.get_context(session_id, context_id)
    if not context:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Context {context_id} not found in session {session_id}"
        )
    
    # 检查权限
    if context.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    updated_context = cm.update_context(session_id, context_id, update_data)
    
    logger.info(f"Updated context {context_id} in session {session_id}")
    return updated_context


@v1_router.delete(
    "/contexts/{session_id}/{context_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["上下文管理"],
    summary="删除上下文",
    description="删除指定上下文（需要认证）",
    dependencies=[Depends(PermissionChecker(["write"]))]
)
async def delete_context(
    session_id: str = Path(..., description="会话ID"),
    context_id: str = Path(..., description="上下文ID"),
    current_user = Depends(get_auth_service().get_current_user),
    cm: ContextManager = Depends(get_context_manager)
):
    """删除上下文"""
    # 先获取上下文以检查权限
    context = cm.get_context(session_id, context_id)
    if not context:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Context {context_id} not found in session {session_id}"
        )
    
    # 检查权限
    if context.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    success = cm.delete_context(session_id, context_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Context {context_id} not found in session {session_id}"
        )
    
    logger.info(f"Deleted context {context_id} from session {session_id}")


# 拓扑记忆集成端点（v1版本）
@v1_router.post(
    "/topology/query",
    response_model=TopologyResponse,
    tags=["拓扑记忆"],
    summary="拓扑查询",
    description="查询拓扑记忆系统（需要认证）",
    dependencies=[Depends(PermissionChecker(["read"]))]
)
async def query_topology(
    query: TopologyQuery,
    current_user = Depends(get_auth_service().get_current_user),
    ta: TopologyAdapter = Depends(get_topology_adapter),
    cache: CacheManager = Depends(get_cache_manager)
):
    """查询拓扑记忆系统"""
    # 检查缓存
    cache_key = f"topology_query:{current_user.user_id}:{hash(str(query.dict()))}"
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


# 系统管理端点（v1版本）
@v1_router.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["系统管理"],
    summary="健康检查",
    description="检查服务及其依赖的健康状态"
)
async def health_check_v1(
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


@v1_router.get(
    "/stats",
    response_model=dict,
    tags=["系统管理"],
    summary="系统统计",
    description="获取系统统计信息（需要管理员权限）",
    dependencies=[Depends(PermissionChecker(["admin"]))]
)
async def get_stats_v1(
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


@v1_router.post(
    "/cache/clear",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["系统管理"],
    summary="清理缓存",
    description="清理所有缓存（需要管理员权限）",
    dependencies=[Depends(PermissionChecker(["admin"]))]
)
async def clear_cache_v1(cache: CacheManager = Depends(get_cache_manager)):
    """清理缓存"""
    cache.clear()
    logger.info("Cache cleared")


# 添加版本信息端点到v1路由器
create_version_endpoints(v1_router)

# v2版本示例端点（演示版本升级）
@v2_router.post(
    "/contexts",
    response_model=ContextResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["上下文管理"],
    summary="创建上下文（v2）",
    description="创建新的上下文条目 - v2版本增强功能",
    dependencies=[Depends(PermissionChecker(["write"]))]
)
async def create_context_v2(
    context_data: ContextCreate,
    current_user = Depends(get_auth_service().get_current_user),
    cm: ContextManager = Depends(get_context_manager)
):
    """创建新的上下文（v2版本）"""
    try:
        # v2版本：添加自动标签生成
        context_data.user_id = current_user.user_id
        
        # 如果metadata中没有标签，自动生成
        if "tags" not in context_data.metadata:
            # 简化的标签生成逻辑
            content_text = str(context_data.content)
            tags = []
            if "conversation" in content_text.lower():
                tags.append("conversation")
            if "memory" in content_text.lower():
                tags.append("memory")
            if "task" in content_text.lower():
                tags.append("task")
            
            if tags:
                context_data.metadata["auto_tags"] = tags
        
        context = cm.create_context(context_data)
        logger.info(f"Created context {context.id} (v2) for session {context.session_id}")
        return context
    except Exception as e:
        logger.error(f"Failed to create context (v2): {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create context: {str(e)}"
        )


# 将路由器添加到应用
app.include_router(v1_router)
app.include_router(v2_router)

# 创建版本化文档
create_versioned_docs(app, "v1")
create_versioned_docs(app, "v2")

# 自定义文档页面
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """自定义Swagger UI - 显示版本选择"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - API Versions",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
        swagger_ui_parameters={"defaultModelsExpandDepth": -1}
    )


@app.get("/redoc", include_in_schema=False)
async def custom_redoc_html():
    """自定义ReDoc页面"""
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )