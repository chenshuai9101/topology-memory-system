"""
拓扑记忆系统向后兼容层
为OpenClaw对话系统提供100%向后兼容支持
"""

__version__ = "1.0.0"
__author__ = "Topology Memory Compatibility Team"

from .session_adapter import SessionAdapter, SessionLifecycleManager
from .context_converter import ContextConverter, ReverseContextConverter
from .query_proxy import QueryProxy
from .event_bridge import EventBridge
from .cache_adapter import CacheAdapter
from .error_handler import ErrorHandler
from .compatibility_layer import CompatibilityLayer

__all__ = [
    'SessionAdapter',
    'SessionLifecycleManager',
    'ContextConverter',
    'ReverseContextConverter',
    'QueryProxy',
    'EventBridge',
    'CacheAdapter',
    'ErrorHandler',
    'CompatibilityLayer'
]