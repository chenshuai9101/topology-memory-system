"""
拓扑记忆上下文管理器 - API版本控制
提供API版本管理功能
"""

from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.routing import APIRoute
from fastapi.openapi.docs import get_swagger_ui_html

from .schemas import APIVersion
from .dependencies import get_settings


class VersionedAPIRoute(APIRoute):
    """版本化API路由"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 从路径中提取版本信息
        path = self.path
        if path.startswith("/v"):
            parts = path.split("/")
            if len(parts) > 1 and parts[1].startswith("v"):
                self.api_version = parts[1]
            else:
                self.api_version = "v1"
        else:
            self.api_version = "v1"


class APIVersionManager:
    """API版本管理器"""
    
    def __init__(self):
        self.versions: Dict[str, APIVersion] = {
            "v1": APIVersion(
                version="v1",
                status="stable",
                deprecated=False,
                sunset_date=None
            ),
            "v2": APIVersion(
                version="v2",
                status="beta",
                deprecated=False,
                sunset_date=None
            )
        }
        self.current_version = "v1"
        self.default_version = "v1"
    
    def get_version_info(self, version: str) -> Optional[APIVersion]:
        """获取版本信息"""
        return self.versions.get(version)
    
    def get_all_versions(self) -> List[APIVersion]:
        """获取所有版本信息"""
        return list(self.versions.values())
    
    def is_version_deprecated(self, version: str) -> bool:
        """检查版本是否已弃用"""
        version_info = self.get_version_info(version)
        if version_info:
            return version_info.deprecated
        return False
    
    def get_supported_versions(self) -> List[str]:
        """获取支持的版本列表"""
        return [v for v in self.versions.keys() if not self.is_version_deprecated(v)]
    
    def add_version_header(self, response, version: str):
        """添加版本头信息"""
        version_info = self.get_version_info(version)
        if version_info:
            response.headers["X-API-Version"] = version
            response.headers["X-API-Version-Status"] = version_info.status
            
            if version_info.deprecated:
                response.headers["X-API-Deprecated"] = "true"
                if version_info.sunset_date:
                    response.headers["X-API-Sunset-Date"] = version_info.sunset_date.isoformat()
        
        return response


# 全局版本管理器实例
version_manager = APIVersionManager()


def create_versioned_router(prefix: str = "", **kwargs):
    """创建版本化路由器"""
    return APIRouter(
        prefix=prefix,
        route_class=VersionedAPIRoute,
        **kwargs
    )


def version_header_dependency(request: Request):
    """版本头依赖项"""
    # 从请求头中获取版本信息
    version_header = request.headers.get("X-API-Version")
    
    if version_header:
        version = version_header
    else:
        # 从路径中提取版本
        path = request.url.path
        if path.startswith("/v"):
            parts = path.split("/")
            if len(parts) > 1 and parts[1].startswith("v"):
                version = parts[1]
            else:
                version = version_manager.default_version
        else:
            version = version_manager.default_version
    
    # 检查版本是否支持
    if version not in version_manager.get_supported_versions():
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported API version: {version}. Supported versions: {', '.join(version_manager.get_supported_versions())}"
        )
    
    # 检查版本是否已弃用
    if version_manager.is_version_deprecated(version):
        # 添加弃用警告头
        version_info = version_manager.get_version_info(version)
        warning_msg = f"This API version ({version}) is deprecated"
        if version_info and version_info.sunset_date:
            warning_msg += f" and will be sunset on {version_info.sunset_date.date()}"
        
        # 在实际应用中，这里可以记录弃用警告
        pass
    
    return version


# 版本信息端点
def create_version_endpoints(router: APIRouter):
    """创建版本信息端点"""
    
    @router.get(
        "/versions",
        response_model=List[APIVersion],
        summary="获取API版本信息",
        description="获取所有可用的API版本及其状态"
    )
    async def get_api_versions():
        """获取API版本信息"""
        return version_manager.get_all_versions()
    
    @router.get(
        "/version",
        response_model=APIVersion,
        summary="获取当前API版本",
        description="获取当前使用的API版本信息"
    )
    async def get_current_version(version: str = Depends(version_header_dependency)):
        """获取当前API版本"""
        version_info = version_manager.get_version_info(version)
        if not version_info:
            raise HTTPException(status_code=404, detail="Version not found")
        return version_info
    
    return router


# 版本化中间件
async def version_middleware(request: Request, call_next):
    """版本中间件"""
    response = await call_next(request)
    
    # 从请求中提取版本信息
    path = request.url.path
    if path.startswith("/v"):
        parts = path.split("/")
        if len(parts) > 1 and parts[1].startswith("v"):
            version = parts[1]
        else:
            version = version_manager.default_version
    else:
        version = version_manager.default_version
    
    # 添加版本头
    response = version_manager.add_version_header(response, version)
    
    return response


# 版本化文档
def create_versioned_docs(app, version: str = "v1"):
    """创建版本化文档"""
    
    @app.get(f"/{version}/docs", include_in_schema=False)
    async def versioned_swagger_ui_html():
        """版本化Swagger UI"""
        return get_swagger_ui_html(
            openapi_url=f"/{version}/openapi.json",
            title=f"拓扑记忆上下文管理器 API - {version.upper()}",
            swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
        )
    
    @app.get(f"/{version}/openapi.json", include_in_schema=False)
    async def versioned_openapi():
        """版本化OpenAPI JSON"""
        # 这里需要根据版本过滤路由
        # 在实际实现中，需要根据版本号过滤路由
        return app.openapi()