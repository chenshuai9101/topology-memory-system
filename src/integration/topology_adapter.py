"""
拓扑记忆适配器
与现有拓扑记忆系统集成
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

import httpx
from ..api.schemas import TopologyQuery, TopologyResponse, MemoryNode, MemoryEdge


logger = logging.getLogger(__name__)


class TopologyAdapter:
    """拓扑记忆适配器"""
    
    def __init__(self, base_url: str = None, timeout: int = 30):
        """
        初始化拓扑适配器
        
        Args:
            base_url: 拓扑记忆系统基础URL
            timeout: 请求超时时间(秒)
        """
        self.base_url = base_url or "http://localhost:8080"  # 默认URL
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        
        # 本地内存文件路径（备用）
        self.memory_file_path = Path("memory/topology/memory-vectors.json")
        
        # 缓存
        self._cache = {}
        self._cache_ttl = 300  # 5分钟
        
        logger.info(f"TopologyAdapter initialized with base_url={self.base_url}")
    
    async def is_connected(self) -> bool:
        """检查是否连接到拓扑记忆系统"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Failed to connect to topology system: {e}")
            return False
    
    async def query_memory(self, query: TopologyQuery) -> TopologyResponse:
        """
        查询拓扑记忆系统
        
        Args:
            query: 拓扑查询
            
        Returns:
            TopologyResponse: 查询结果
        """
        start_time = time.time()
        
        try:
            # 尝试使用HTTP API
            if await self.is_connected():
                return await self._query_via_api(query)
            else:
                # 回退到本地文件
                return await self._query_via_file(query)
        except Exception as e:
            logger.error(f"Topology query failed: {e}")
            # 返回空结果
            return TopologyResponse(
                nodes=[],
                edges=[],
                query_time=time.time() - start_time,
                total_nodes=0,
                total_edges=0
            )
    
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        获取记忆节点
        
        Args:
            node_id: 节点ID
            
        Returns:
            Optional[Dict[str, Any]]: 节点数据或None
        """
        # 检查缓存
        cache_key = f"node:{node_id}"
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached_data
        
        try:
            # 尝试使用HTTP API
            if await self.is_connected():
                response = await self.client.get(f"{self.base_url}/nodes/{node_id}")
                if response.status_code == 200:
                    node_data = response.json()
                    self._cache[cache_key] = (node_data, time.time())
                    return node_data
            else:
                # 从本地文件获取
                node_data = await self._get_node_from_file(node_id)
                if node_data:
                    self._cache[cache_key] = (node_data, time.time())
                    return node_data
        except Exception as e:
            logger.error(f"Failed to get node {node_id}: {e}")
        
        return None
    
    async def create_node(self, node_data: Dict[str, Any]) -> Optional[str]:
        """
        创建记忆节点
        
        Args:
            node_data: 节点数据
            
        Returns:
            Optional[str]: 创建的节点ID或None
        """
        try:
            if await self.is_connected():
                response = await self.client.post(
                    f"{self.base_url}/nodes",
                    json=node_data
                )
                if response.status_code == 201:
                    created_node = response.json()
                    node_id = created_node.get("id")
                    
                    # 清理缓存
                    self._clear_cache()
                    
                    logger.info(f"Created topology node: {node_id}")
                    return node_id
        except Exception as e:
            logger.error(f"Failed to create node: {e}")
        
        return None
    
    async def update_node(self, node_id: str, node_data: Dict[str, Any]) -> bool:
        """
        更新记忆节点
        
        Args:
            node_id: 节点ID
            node_data: 节点数据
            
        Returns:
            bool: 是否成功更新
        """
        try:
            if await self.is_connected():
                response = await self.client.put(
                    f"{self.base_url}/nodes/{node_id}",
                    json=node_data
                )
                if response.status_code == 200:
                    # 清理缓存
                    cache_key = f"node:{node_id}"
                    if cache_key in self._cache:
                        del self._cache[cache_key]
                    
                    logger.info(f"Updated topology node: {node_id}")
                    return True
        except Exception as e:
            logger.error(f"Failed to update node {node_id}: {e}")
        
        return False
    
    async def delete_node(self, node_id: str) -> bool:
        """
        删除记忆节点
        
        Args:
            node_id: 节点ID
            
        Returns:
            bool: 是否成功删除
        """
        try:
            if await self.is_connected():
                response = await self.client.delete(f"{self.base_url}/nodes/{node_id}")
                if response.status_code == 204:
                    # 清理缓存
                    cache_key = f"node:{node_id}"
                    if cache_key in self._cache:
                        del self._cache[cache_key]
                    
                    logger.info(f"Deleted topology node: {node_id}")
                    return True
        except Exception as e:
            logger.error(f"Failed to delete node {node_id}: {e}")
        
        return False
    
    async def create_edge(self, edge_data: Dict[str, Any]) -> bool:
        """
        创建记忆边
        
        Args:
            edge_data: 边数据
            
        Returns:
            bool: 是否成功创建
        """
        try:
            if await self.is_connected():
                response = await self.client.post(
                    f"{self.base_url}/edges",
                    json=edge_data
                )
                if response.status_code == 201:
                    logger.info(f"Created topology edge: {edge_data}")
                    return True
        except Exception as e:
            logger.error(f"Failed to create edge: {e}")
        
        return False
    
    async def search_similar(self, vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """
        搜索相似记忆
        
        Args:
            vector: 查询向量
            limit: 返回数量限制
            
        Returns:
            List[Dict[str, Any]]: 相似节点列表
        """
        try:
            if await self.is_connected():
                response = await self.client.post(
                    f"{self.base_url}/search/similar",
                    json={"vector": vector, "limit": limit}
                )
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.error(f"Failed to search similar: {e}")
        
        return []
    
    async def get_topology_stats(self) -> Dict[str, Any]:
        """
        获取拓扑系统统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            if await self.is_connected():
                response = await self.client.get(f"{self.base_url}/stats")
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.error(f"Failed to get topology stats: {e}")
        
        # 返回默认统计
        return {
            "total_nodes": 0,
            "total_edges": 0,
            "status": "disconnected"
        }
    
    async def _query_via_api(self, query: TopologyQuery) -> TopologyResponse:
        """通过API查询拓扑记忆"""
        try:
            response = await self.client.post(
                f"{self.base_url}/query",
                json=query.dict()
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # 转换为响应模型
                nodes = [
                    MemoryNode(
                        node_id=node["id"],
                        content=node.get("content", ""),
                        vector=node.get("vector"),
                        metadata=node.get("metadata", {})
                    )
                    for node in result.get("nodes", [])
                ]
                
                edges = [
                    MemoryEdge(
                        source_id=edge["source"],
                        target_id=edge["target"],
                        relationship=edge.get("relationship", "related_to"),
                        weight=edge.get("weight", 1.0)
                    )
                    for edge in result.get("edges", [])
                ]
                
                return TopologyResponse(
                    nodes=nodes,
                    edges=edges,
                    query_time=result.get("query_time", 0.0),
                    total_nodes=result.get("total_nodes", 0),
                    total_edges=result.get("total_edges", 0)
                )
            else:
                raise Exception(f"API returned status {response.status_code}")
                
        except Exception as e:
            logger.error(f"API query failed: {e}")
            raise
    
    async def _query_via_file(self, query: TopologyQuery) -> TopologyResponse:
        """通过本地文件查询拓扑记忆"""
        start_time = time.time()
        
        try:
            # 检查文件是否存在
            if not self.memory_file_path.exists():
                logger.warning(f"Memory file not found: {self.memory_file_path}")
                return TopologyResponse(
                    nodes=[],
                    edges=[],
                    query_time=time.time() - start_time,
                    total_nodes=0,
                    total_edges=0
                )
            
            # 读取内存文件
            with open(self.memory_file_path, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            # 简单文本匹配（实际应该使用向量相似度搜索）
            query_lower = query.query.lower()
            matched_nodes = []
            
            for node_id, node_data in memory_data.items():
                content = node_data.get("content", "")
                if query_lower in content.lower():
                    matched_nodes.append(
                        MemoryNode(
                            node_id=node_id,
                            content=content,
                            vector=node_data.get("vector"),
                            metadata=node_data.get("metadata", {})
                        )
                    )
                
                if len(matched_nodes) >= query.limit:
                    break
            
            # 创建简单的边（实际应该从文件中读取边数据）
            edges = []
            if len(matched_nodes) > 1:
                for i in range(len(matched_nodes) - 1):
                    edges.append(
                        MemoryEdge(
                            source_id=matched_nodes[i].node_id,
                            target_id=matched_nodes[i + 1].node_id,
                            relationship="related_to",
                            weight=0.8
                        )
                    )
            
            query_time = time.time() - start_time
            
            logger.info(f"File query found {len(matched_nodes)} nodes for query: {query.query}")
            
            return TopologyResponse(
                nodes=matched_nodes,
                edges=edges,
                query_time=query_time,
                total_nodes=len(matched_nodes),
                total_edges=len(edges)
            )
            
        except Exception as e:
            logger.error(f"File query failed: {e}")
            return TopologyResponse(
                nodes=[],
                edges=[],
                query_time=time.time() - start_time,
                total_nodes=0,
                total_edges=0
            )
    
    async def _get_node_from_file(self, node_id: str) -> Optional[Dict[str, Any]]:
        """从本地文件获取节点"""
        try:
            if not self.memory_file_path.exists():
                return None
            
            with open(self.memory_file_path, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            node_data = memory_data.get(node_id)
            if node_data:
                return {
                    "id": node_id,
                    "content": node_data.get("content", ""),
                    "vector": node_data.get("vector"),
                    "metadata": node_data.get("metadata", {})
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get node from file: {e}")
            return None
    
    def _clear_cache(self):
        """清理缓存"""
        self._cache.clear()
        logger.debug("Topology adapter cache cleared")
    
    async def close(self):
        """关闭适配器"""
        await self.client.aclose()
        logger.info("Topology adapter closed")
    
    def __del__(self):
        """析构函数"""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close())
        except:
            pass