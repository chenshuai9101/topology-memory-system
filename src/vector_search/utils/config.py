"""
向量搜索配置管理
"""

import os
import logging
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field

logger = logging.getLogger(__name__)


class VectorSearchConfig(BaseSettings):
    """向量搜索配置"""
    
    # Qdrant配置
    qdrant_host: str = Field("localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(6333, env="QDRANT_PORT")
    qdrant_grpc_port: int = Field(6334, env="QDRANT_GRPC_PORT")
    qdrant_api_key: Optional[str] = Field(None, env="QDRANT_API_KEY")
    qdrant_timeout: int = Field(30, env="QDRANT_TIMEOUT")
    qdrant_prefer_grpc: bool = Field(False, env="QDRANT_PREFER_GRPC")
    
    # 向量编码器配置
    encoder_model: str = Field("paraphrase-multilingual-MiniLM-L12-v2", env="ENCODER_MODEL")
    encoder_device: str = Field("cpu", env="ENCODER_DEVICE")
    encoder_cache_folder: Optional[str] = Field(None, env="ENCODER_CACHE_FOLDER")
    
    # 集合配置
    default_collection: str = Field("topology_memory", env="DEFAULT_COLLECTION")
    default_vector_size: int = Field(384, env="DEFAULT_VECTOR_SIZE")
    
    # 搜索配置
    default_search_limit: int = Field(10, env="DEFAULT_SEARCH_LIMIT")
    default_search_threshold: float = Field(0.5, env="DEFAULT_SEARCH_THRESHOLD")
    default_vector_weight: float = Field(0.7, env="DEFAULT_VECTOR_WEIGHT")
    default_keyword_weight: float = Field(0.3, env="DEFAULT_KEYWORD_WEIGHT")
    default_time_decay_factor: float = Field(0.1, env="DEFAULT_TIME_DECAY_FACTOR")
    default_importance_weight: float = Field(1.0, env="DEFAULT_IMPORTANCE_WEIGHT")
    
    # 性能配置
    max_batch_size: int = Field(100, env="MAX_BATCH_SIZE")
    search_timeout_ms: int = Field(100, env="SEARCH_TIMEOUT_MS")
    cache_enabled: bool = Field(True, env="CACHE_ENABLED")
    cache_ttl: int = Field(300, env="CACHE_TTL")  # 5分钟
    
    # 日志配置
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(None, env="LOG_FILE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def get_qdrant_url(self) -> str:
        """获取Qdrant URL"""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"
    
    def get_grpc_url(self) -> str:
        """获取gRPC URL"""
        return f"{self.qdrant_host}:{self.qdrant_grpc_port}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "qdrant": {
                "host": self.qdrant_host,
                "port": self.qdrant_port,
                "grpc_port": self.qdrant_grpc_port,
                "timeout": self.qdrant_timeout,
                "prefer_grpc": self.qdrant_prefer_grpc
            },
            "encoder": {
                "model": self.encoder_model,
                "device": self.encoder_device,
                "cache_folder": self.encoder_cache_folder
            },
            "collection": {
                "default": self.default_collection,
                "vector_size": self.default_vector_size
            },
            "search": {
                "default_limit": self.default_search_limit,
                "default_threshold": self.default_search_threshold,
                "vector_weight": self.default_vector_weight,
                "keyword_weight": self.default_keyword_weight,
                "time_decay_factor": self.default_time_decay_factor,
                "importance_weight": self.default_importance_weight
            },
            "performance": {
                "max_batch_size": self.max_batch_size,
                "search_timeout_ms": self.search_timeout_ms,
                "cache_enabled": self.cache_enabled,
                "cache_ttl": self.cache_ttl
            }
        }


# 全局配置实例
_config_instance: Optional[VectorSearchConfig] = None


def get_config() -> VectorSearchConfig:
    """
    获取配置实例（单例模式）
    
    Returns:
        配置实例
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = VectorSearchConfig()
        logger.info("向量搜索配置已加载")
    
    return _config_instance


def reload_config() -> VectorSearchConfig:
    """
    重新加载配置
    
    Returns:
        新的配置实例
    """
    global _config_instance
    
    _config_instance = VectorSearchConfig()
    logger.info("向量搜索配置已重新加载")
    
    return _config_instance


def validate_config() -> Dict[str, Any]:
    """
    验证配置
    
    Returns:
        验证结果
    """
    config = get_config()
    issues = []
    
    # 验证Qdrant配置
    if not config.qdrant_host:
        issues.append("QDRANT_HOST不能为空")
    
    if config.qdrant_port <= 0 or config.qdrant_port > 65535:
        issues.append(f"QDRANT_PORT无效: {config.qdrant_port}")
    
    # 验证搜索配置
    if config.default_search_limit <= 0 or config.default_search_limit > 1000:
        issues.append(f"DEFAULT_SEARCH_LIMIT无效: {config.default_search_limit}")
    
    if config.default_search_threshold < 0 or config.default_search_threshold > 1:
        issues.append(f"DEFAULT_SEARCH_THRESHOLD无效: {config.default_search_threshold}")
    
    if config.default_vector_weight < 0 or config.default_vector_weight > 1:
        issues.append(f"DEFAULT_VECTOR_WEIGHT无效: {config.default_vector_weight}")
    
    if config.default_keyword_weight < 0 or config.default_keyword_weight > 1:
        issues.append(f"DEFAULT_KEYWORD_WEIGHT无效: {config.default_keyword_weight}")
    
    if abs(config.default_vector_weight + config.default_keyword_weight - 1.0) > 0.01:
        issues.append(f"向量权重和关键词权重之和应为1.0，当前为: {config.default_vector_weight + config.default_keyword_weight}")
    
    # 验证性能配置
    if config.max_batch_size <= 0:
        issues.append(f"MAX_BATCH_SIZE无效: {config.max_batch_size}")
    
    if config.search_timeout_ms <= 0:
        issues.append(f"SEARCH_TIMEOUT_MS无效: {config.search_timeout_ms}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "config_summary": config.to_dict()
    }


# 预定义模型配置
MODEL_CONFIGS = {
    "multilingual_mini": {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "vector_size": 384,
        "languages": ["zh", "en", "es", "fr", "de", "it", "nl", "pl", "pt", "ru"],
        "description": "多语言小型模型，适合通用语义搜索"
    },
    "multilingual_base": {
        "name": "paraphrase-multilingual-mpnet-base-v2",
        "vector_size": 768,
        "languages": ["zh", "en", "es", "fr", "de", "it", "nl", "pl", "pt", "ru"],
        "description": "多语言基础模型，提供更高的准确性"
    },
    "chinese_optimized": {
        "name": "distiluse-base-multilingual-cased-v1",
        "vector_size": 512,
        "languages": ["zh", "en"],
        "description": "中文优化模型，适合中文语义搜索"
    },
    "english_optimized": {
        "name": "all-MiniLM-L6-v2",
        "vector_size": 384,
        "languages": ["en"],
        "description": "英文优化模型，适合英文语义搜索"
    }
}


def get_model_config(model_type: str) -> Dict[str, Any]:
    """
    获取模型配置
    
    Args:
        model_type: 模型类型
        
    Returns:
        模型配置
    """
    return MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["multilingual_mini"])


def list_available_models() -> Dict[str, Dict[str, Any]]:
    """
    列出所有可用模型
    
    Returns:
        模型列表
    """
    return MODEL_CONFIGS