#!/usr/bin/env python3
"""
拓扑记忆系统简化启动脚本
绕过复杂的依赖，直接启动核心功能
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# 创建简化应用
app = FastAPI(
    title="拓扑记忆上下文管理器 (简化版)",
    description="记忆优化和管理系统",
    version="1.0.0-simple"
)

# 数据模型
class MemoryNode(BaseModel):
    node_id: str
    content: str
    metadata: Optional[dict] = None

class MemoryEdge(BaseModel):
    source_id: str
    target_id: str
    relationship: str = "related"
    weight: float = 1.0

class QueryRequest(BaseModel):
    query: str
    limit: int = 10

# 内存存储（简化版）
memory_nodes = {}
memory_edges = []

@app.get("/")
async def root():
    return {
        "service": "拓扑记忆上下文管理器",
        "version": "1.0.0-simple",
        "status": "运行中",
        "endpoints": [
            "/health",
            "/nodes",
            "/edges",
            "/search",
            "/optimize"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": "2026-04-02T02:52:00Z",
        "service": "topology-memory",
        "version": "1.0.0"
    }

@app.post("/nodes")
async def create_node(node: MemoryNode):
    """创建记忆节点"""
    memory_nodes[node.node_id] = node.dict()
    return {
        "status": "created",
        "node_id": node.node_id,
        "message": f"节点 {node.node_id} 已创建"
    }

@app.get("/nodes")
async def list_nodes():
    """列出所有记忆节点"""
    return {
        "count": len(memory_nodes),
        "nodes": list(memory_nodes.values())
    }

@app.post("/edges")
async def create_edge(edge: MemoryEdge):
    """创建记忆关联"""
    memory_edges.append(edge.dict())
    return {
        "status": "created",
        "edge": edge.dict(),
        "message": "关联已创建"
    }

@app.get("/edges")
async def list_edges():
    """列出所有记忆关联"""
    return {
        "count": len(memory_edges),
        "edges": memory_edges
    }

@app.post("/search")
async def search_memories(request: QueryRequest):
    """搜索记忆"""
    query = request.query.lower()
    results = []
    
    for node_id, node in memory_nodes.items():
        content = node.get('content', '').lower()
        if query in content:
            # 简单相关性评分
            relevance = content.count(query) / max(len(content.split()), 1)
            results.append({
                **node,
                "relevance_score": min(1.0, relevance * 10),
                "match_type": "content"
            })
    
    # 按相关性排序
    results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    return {
        "query": request.query,
        "count": len(results),
        "results": results[:request.limit]
    }

@app.post("/optimize")
async def optimize_memories():
    """优化记忆（简化版）"""
    try:
        # 简化优化逻辑
        optimized_count = 0
        
        for node_id, node in memory_nodes.items():
            content = node.get('content', '')
            if len(content) > 50:
                # 简单优化：截断过长的内容
                if len(content) > 100:
                    optimized_content = content[:97] + "..."
                    memory_nodes[node_id]['content'] = optimized_content
                    memory_nodes[node_id]['optimized'] = True
                    memory_nodes[node_id]['optimization_ratio'] = len(optimized_content) / len(content)
                    optimized_count += 1
        
        return {
            "status": "success",
            "optimized_count": optimized_count,
            "total_nodes": len(memory_nodes),
            "message": f"优化了 {optimized_count} 个记忆节点"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"优化失败: {str(e)}",
            "optimized_count": 0
        }

@app.get("/status")
async def system_status():
    """系统状态"""
    return {
        "status": "running",
        "nodes_count": len(memory_nodes),
        "edges_count": len(memory_edges),
        "memory_usage": "minimal",
        "api_version": "1.0.0-simple",
        "endpoints_available": [
            "/health",
            "/nodes",
            "/edges",
            "/search",
            "/optimize",
            "/status"
        ]
    }

if __name__ == "__main__":
    print("=" * 60)
    print("拓扑记忆上下文管理器 - 简化版")
    print("=" * 60)
    print("服务启动中...")
    print(f"API地址: http://localhost:8000")
    print(f"文档地址: http://localhost:8000/docs")
    print(f"健康检查: http://localhost:8000/health")
    print("=" * 60)
    
    # 启动服务
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )