"""
拓扑记忆上下文管理器 - 核心引擎模块
"""

from .context_manager import (
    ContextManager,
    ContextCreate,
    ContextUpdate,
    ContextEntry,
    ContextResponse
)

from .memory_manager import (
    MemoryManager,
    MemoryNodeData,
    MemoryEdgeData
)

from .topology_algorithms import (
    TopologyAlgorithms,
    TopologyConfig
)

from .engine import (
    TopologyMemoryEngine,
    EngineConfig
)

from .performance_test import PerformanceTester

__all__ = [
    # 上下文管理
    "ContextManager",
    "ContextCreate",
    "ContextUpdate",
    "ContextEntry",
    "ContextResponse",
    
    # 记忆管理
    "MemoryManager",
    "MemoryNodeData",
    "MemoryEdgeData",
    
    # 拓扑算法
    "TopologyAlgorithms",
    "TopologyConfig",
    
    # 核心引擎
    "TopologyMemoryEngine",
    "EngineConfig",
    
    # 性能测试
    "PerformanceTester"
]

__version__ = "1.0.0"
__author__ = "拓扑记忆项目组"
__description__ = "拓扑记忆上下文管理器核心引擎"