"""
拓扑记忆上下文管理器 - 数据模型
定义Pydantic模型用于请求/响应验证
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator


class ContextBase(BaseModel):
    """上下文基础模型"""
    session_id: str = Field(..., description="会话ID")
    user_id: str = Field(..., description="用户ID")
    context_type: str = Field(..., description="上下文类型")
    content: Dict[str, Any] = Field(..., description="上下文内容")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="元数据")


class ContextCreate(ContextBase):
    """创建上下文请求模型"""
    priority: int = Field(default=1, ge=1, le=10, description="优先级(1-10)")
    ttl: Optional[int] = Field(default=None, description="生存时间(秒)")


class ContextUpdate(BaseModel):
    """更新上下文请求模型"""
    content: Optional[Dict[str, Any]] = Field(None, description="上下文内容")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    priority: Optional[int] = Field(None, ge=1, le=10, description="优先级(1-10)")
    ttl: Optional[int] = Field(None, description="生存时间(秒)")


class ContextResponse(ContextBase):
    """上下文响应模型"""
    id: str = Field(..., description="上下文ID")
    priority: int = Field(..., description="优先级")
    ttl: Optional[int] = Field(None, description="生存时间(秒)")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    expires_at: Optional[datetime] = Field(None, description="过期时间")

    class Config:
        from_attributes = True


class MemoryNode(BaseModel):
    """记忆节点模型"""
    node_id: str = Field(..., description="节点ID")
    content: str = Field(..., description="节点内容")
    vector: Optional[List[float]] = Field(None, description="向量表示")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="元数据")


class MemoryEdge(BaseModel):
    """记忆边模型"""
    source_id: str = Field(..., description="源节点ID")
    target_id: str = Field(..., description="目标节点ID")
    relationship: str = Field(..., description="关系类型")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="权重")


class TopologyQuery(BaseModel):
    """拓扑查询请求模型"""
    query: str = Field(..., description="查询文本")
    limit: int = Field(default=10, ge=1, le=100, description="返回数量限制")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="相似度阈值")
    include_vectors: bool = Field(default=False, description="是否包含向量")


class TopologyResponse(BaseModel):
    """拓扑查询响应模型"""
    nodes: List[MemoryNode] = Field(..., description="记忆节点列表")
    edges: List[MemoryEdge] = Field(..., description="记忆边列表")
    query_time: float = Field(..., description="查询耗时(秒)")
    total_nodes: int = Field(..., description="总节点数")
    total_edges: int = Field(..., description="总边数")


class PerformanceMetrics(BaseModel):
    """性能指标模型"""
    request_id: str = Field(..., description="请求ID")
    endpoint: str = Field(..., description="API端点")
    method: str = Field(..., description="HTTP方法")
    response_time: float = Field(..., description="响应时间(秒)")
    status_code: int = Field(..., description="状态码")
    timestamp: datetime = Field(..., description="时间戳")


class HealthCheckResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="服务版本")
    timestamp: datetime = Field(..., description="检查时间")
    dependencies: Dict[str, str] = Field(..., description="依赖服务状态")


class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str = Field(..., description="错误信息")
    detail: Optional[str] = Field(None, description="错误详情")
    request_id: Optional[str] = Field(None, description="请求ID")


class PaginationParams(BaseModel):
    """分页参数模型"""
    page: int = Field(default=1, ge=1, description="页码")
    page_size: int = Field(default=20, ge=1, le=100, description="每页大小")


class PaginatedResponse(BaseModel):
    """分页响应模型"""
    items: List[Any] = Field(..., description="数据列表")
    total: int = Field(..., description="总数量")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页大小")
    total_pages: int = Field(..., description="总页数")


# 用户认证相关模型
class UserBase(BaseModel):
    """用户基础模型"""
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    email: str = Field(..., description="邮箱")


class UserCreate(UserBase):
    """创建用户请求模型"""
    password: str = Field(..., min_length=8, description="密码")


class UserUpdate(BaseModel):
    """更新用户请求模型"""
    username: Optional[str] = Field(None, min_length=3, max_length=50, description="用户名")
    email: Optional[str] = Field(None, description="邮箱")
    password: Optional[str] = Field(None, min_length=8, description="密码")


class UserResponse(UserBase):
    """用户响应模型"""
    id: str = Field(..., description="用户ID")
    is_active: bool = Field(default=True, description="是否激活")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    
    class Config:
        from_attributes = True


class UserTokenData(BaseModel):
    """用户令牌数据模型"""
    user_id: str = Field(..., description="用户ID")
    username: Optional[str] = Field(None, description="用户名")
    roles: List[str] = Field(default=[], description="角色列表")
    permissions: List[str] = Field(default=[], description="权限列表")


class Token(BaseModel):
    """令牌响应模型"""
    access_token: str = Field(..., description="访问令牌")
    refresh_token: str = Field(..., description="刷新令牌")
    token_type: str = Field(default="bearer", description="令牌类型")
    expires_in: int = Field(..., description="过期时间(秒)")


class TokenData(BaseModel):
    """令牌数据模型"""
    user_id: Optional[str] = Field(None, description="用户ID")
    username: Optional[str] = Field(None, description="用户名")


class LoginRequest(BaseModel):
    """登录请求模型"""
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class RefreshTokenRequest(BaseModel):
    """刷新令牌请求模型"""
    refresh_token: str = Field(..., description="刷新令牌")


# API版本控制
class APIVersion(BaseModel):
    """API版本信息"""
    version: str = Field(..., description="API版本")
    status: str = Field(..., description="版本状态")
    deprecated: bool = Field(default=False, description="是否已弃用")
    sunset_date: Optional[datetime] = Field(None, description="停止支持日期")


# 验证器
@validator('context_type')
def validate_context_type(cls, v):
    """验证上下文类型"""
    valid_types = ['conversation', 'memory', 'task', 'knowledge', 'other']
    if v not in valid_types:
        raise ValueError(f'context_type must be one of {valid_types}')
    return v


@validator('relationship')
def validate_relationship(cls, v):
    """验证关系类型"""
    valid_relationships = ['related_to', 'similar_to', 'part_of', 'causes', 'precedes', 'other']
    if v not in valid_relationships:
        raise ValueError(f'relationship must be one of {valid_relationships}')
    return v


@validator('username')
def validate_username(cls, v):
    """验证用户名"""
    if not v.isalnum():
        raise ValueError('Username must be alphanumeric')
    return v