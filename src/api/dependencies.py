"""
拓扑记忆上下文管理器 - 依赖注入
提供FastAPI依赖项，包括认证、授权、数据库连接等
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from functools import lru_cache

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from pydantic import BaseSettings
import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .schemas import UserTokenData


class Settings(BaseSettings):
    """应用设置"""
    # API设置
    api_title: str = "拓扑记忆上下文管理器 API"
    api_description: str = "基于拓扑学的多维记忆管理系统"
    api_version: str = "0.1.0"
    
    # 安全设置
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # 数据库设置
    database_url: str = "sqlite:///./topology_memory.db"
    redis_url: str = "redis://localhost:6379/0"
    
    # CORS设置
    cors_origins: list = ["*"]
    
    # 性能设置
    max_contexts_per_session: int = 100
    cache_max_size: int = 1000
    cache_ttl: int = 300
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """获取应用设置（缓存）"""
    return Settings()


# JWT认证
security = HTTPBearer()


class AuthService:
    """认证服务"""
    
    def __init__(self, settings: Settings = Depends(get_settings)):
        self.settings = settings
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """创建访问令牌"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.settings.access_token_expire_minutes
            )
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(
            to_encode, 
            self.settings.secret_key, 
            algorithm=self.settings.algorithm
        )
        return encoded_jwt
    
    def create_refresh_token(self, data: dict) -> str:
        """创建刷新令牌"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.settings.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(
            to_encode, 
            self.settings.secret_key, 
            algorithm=self.settings.algorithm
        )
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """验证JWT令牌"""
        try:
            payload = jwt.decode(
                token, 
                self.settings.secret_key, 
                algorithms=[self.settings.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserTokenData:
        """获取当前用户"""
        token = credentials.credentials
        payload = self.verify_token(token)
        
        # 检查令牌类型
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return UserTokenData(
            user_id=user_id,
            username=payload.get("username"),
            roles=payload.get("roles", []),
            permissions=payload.get("permissions", [])
        )


def get_auth_service(settings: Settings = Depends(get_settings)) -> AuthService:
    """获取认证服务"""
    return AuthService(settings)


# 数据库连接
def get_db_engine(settings: Settings = Depends(get_settings)):
    """获取数据库引擎"""
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
    )
    return engine


def get_db_session(engine = Depends(get_db_engine)):
    """获取数据库会话"""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Redis连接
def get_redis_client(settings: Settings = Depends(get_settings)):
    """获取Redis客户端"""
    return redis.from_url(settings.redis_url, decode_responses=True)


# 权限检查
class PermissionChecker:
    """权限检查器"""
    
    def __init__(self, required_permissions: list):
        self.required_permissions = required_permissions
    
    def __call__(self, current_user: UserTokenData = Depends(get_auth_service().get_current_user)):
        """检查用户权限"""
        user_permissions = set(current_user.permissions)
        required_permissions = set(self.required_permissions)
        
        if not required_permissions.issubset(user_permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        return current_user


# 速率限制
class RateLimiter:
    """速率限制器"""
    
    def __init__(self, redis_client, limit: int = 100, window: int = 60):
        self.redis = redis_client
        self.limit = limit
        self.window = window
    
    async def check_rate_limit(self, key: str) -> bool:
        """检查速率限制"""
        current = self.redis.get(key)
        if current is None:
            self.redis.setex(key, self.window, 1)
            return True
        
        current_count = int(current)
        if current_count >= self.limit:
            return False
        
        self.redis.incr(key)
        return True


def get_rate_limiter(
    redis_client = Depends(get_redis_client),
    settings: Settings = Depends(get_settings)
) -> RateLimiter:
    """获取速率限制器"""
    return RateLimiter(redis_client, limit=100, window=60)