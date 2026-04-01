"""
查询代理 - 处理OpenClaw查询到拓扑记忆查询的转换和执行
"""

import re
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List, Union
import logging

logger = logging.getLogger(__name__)


class QueryProxy:
    """查询代理，将OpenClaw查询转换为拓扑记忆查询并执行"""
    
    def __init__(self, session_adapter, topology_api_client, context_converter=None):
        """
        初始化查询代理
        
        Args:
            session_adapter: 会话适配器实例
            topology_api_client: 拓扑记忆API客户端
            context_converter: 上下文转换器（可选）
        """
        self.session_adapter = session_adapter
        self.api_client = topology_api_client
        self.context_converter = context_converter
        
        # 查询类型处理器映射
        self.query_handlers = {
            'get_context': self._handle_get_context,
            'search_messages': self._handle_search_messages,
            'get_session_history': self._handle_get_session_history,
            'get_related_contexts': self._handle_get_related_contexts,
            'get_recent_sessions': self._handle_get_recent_sessions,
            'get_context_by_id': self._handle_get_context_by_id,
            'get_context_stats': self._handle_get_context_stats
        }
    
    def execute_openclaw_query(self, openclaw_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行OpenClaw查询，转换为拓扑记忆查询
        
        Args:
            openclaw_query: OpenClaw查询对象
            
        Returns:
            查询结果
        """
        try:
            # 解析查询类型
            query_type = openclaw_query.get('type', 'get_context')
            
            # 获取对应的处理器
            handler = self.query_handlers.get(query_type)
            if not handler:
                raise ValueError(f"Unsupported query type: {query_type}")
            
            # 执行查询
            result = handler(openclaw_query)
            
            logger.info(f"Executed OpenClaw query: {query_type}, result size: {result.get('count', 0)}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute OpenClaw query: {e}")
            return {
                'success': False,
                'error': str(e),
                'query_type': openclaw_query.get('type'),
                'timestamp': datetime.now().isoformat()
            }
    
    def _handle_get_context(self, openclaw_query: Dict[str, Any]) -> Dict[str, Any]:
        """处理获取上下文查询"""
        session_key = openclaw_query.get('session_key')
        if not session_key:
            raise ValueError("session_key is required for get_context query")
        
        # 映射会话键
        session_id = self.session_adapter.map_session_key(session_key, create_if_missing=False)
        if not session_id:
            return {
                'success': True,
                'contexts': [],
                'count': 0,
                'session_key': session_key,
                'message': 'Session not found or no contexts'
            }
        
        # 构建拓扑记忆查询参数
        topology_query = {
            'session_id': session_id,
            'limit': openclaw_query.get('limit', 50),
            'offset': openclaw_query.get('offset', 0),
            'order_by': openclaw_query.get('order_by', 'created_at'),
            'order_direction': openclaw_query.get('order_direction', 'desc'),
            'include_expired': openclaw_query.get('include_expired', False)
        }
        
        # 添加过滤器
        filters = openclaw_query.get('filters', {})
        if filters:
            topology_query['filters'] = self._convert_filters(filters)
        
        # 添加上下文类型过滤
        context_types = openclaw_query.get('context_types')
        if context_types:
            topology_query['context_types'] = context_types
        
        # 执行查询
        result = self.api_client.query_contexts(topology_query)
        
        # 转换结果格式
        contexts = self._convert_query_result(result, openclaw_query)
        
        return {
            'success': True,
            'contexts': contexts,
            'count': len(contexts),
            'total_count': getattr(result, 'total_count', len(contexts)),
            'session_key': session_key,
            'session_id': session_id,
            'has_more': getattr(result, 'has_more', False),
            'next_offset': openclaw_query.get('offset', 0) + len(contexts)
        }
    
    def _handle_search_messages(self, openclaw_query: Dict[str, Any]) -> Dict[str, Any]:
        """处理搜索消息查询"""
        session_key = openclaw_query.get('session_key')
        search_text = openclaw_query.get('search_text', '')
        
        if not search_text:
            raise ValueError("search_text is required for search_messages query")
        
        # 映射会话键
        session_id = None
        if session_key:
            session_id = self.session_adapter.map_session_key(session_key, create_if_missing=False)
        
        # 构建拓扑记忆查询
        topology_query = {
            'search_text': search_text,
            'limit': openclaw_query.get('limit', 100),
            'offset': openclaw_query.get('offset', 0),
            'context_types': openclaw_query.get('context_types', ['user_message', 'assistant_message', 'system_message']),
            'fuzzy': openclaw_query.get('fuzzy', True)
        }
        
        # 如果指定了会话，添加会话过滤
        if session_id:
            topology_query['session_id'] = session_id
        
        # 添加时间范围过滤
        time_range = openclaw_query.get('time_range')
        if time_range:
            topology_query['time_range'] = self._convert_time_range(time_range)
        
        # 执行搜索
        result = self.api_client.search_contexts(topology_query)
        
        # 转换结果格式
        contexts = self._convert_query_result(result, openclaw_query)
        
        return {
            'success': True,
            'results': contexts,
            'count': len(contexts),
            'total_count': getattr(result, 'total_count', len(contexts)),
            'search_text': search_text,
            'session_key': session_key,
            'has_more': getattr(result, 'has_more', False)
        }
    
    def _handle_get_session_history(self, openclaw_query: Dict[str, Any]) -> Dict[str, Any]:
        """处理获取会话历史查询"""
        session_key = openclaw_query.get('session_key')
        if not session_key:
            raise ValueError("session_key is required for get_session_history query")
        
        # 映射会话键
        session_id = self.session_adapter.map_session_key(session_key, create_if_missing=False)
        if not session_id:
            return {
                'success': True,
                'history': [],
                'count': 0,
                'session_key': session_key,
                'message': 'Session not found'
            }
        
        # 获取会话信息
        session_info = self.session_adapter.get_openclaw_session_info(session_id)
        
        # 构建历史查询
        history_query = {
            'session_id': session_id,
            'limit': openclaw_query.get('limit', 100),
            'offset': openclaw_query.get('offset', 0),
            'order_by': 'created_at',
            'order_direction': 'desc',
            'include_all_types': True
        }
        
        # 执行查询
        result = self.api_client.get_session_history(history_query)
        
        # 转换结果格式
        history = self._convert_query_result(result, openclaw_query)
        
        # 添加会话摘要
        session_summary = {
            'session_key': session_key,
            'session_id': session_id,
            'channel': session_info.get('channel') if session_info else 'unknown',
            'account_id': session_info.get('account_id') if session_info else 'unknown',
            'created_at': session_info.get('mapping_created_at') if session_info else None,
            'last_accessed': session_info.get('mapping_last_accessed') if session_info else None
        }
        
        return {
            'success': True,
            'history': history,
            'count': len(history),
            'total_count': getattr(result, 'total_count', len(history)),
            'session_summary': session_summary,
            'has_more': getattr(result, 'has_more', False)
        }
    
    def _handle_get_related_contexts(self, openclaw_query: Dict[str, Any]) -> Dict[str, Any]:
        """处理获取相关上下文查询"""
        context_id = openclaw_query.get('context_id')
        session_key = openclaw_query.get('session_key')
        
        if not context_id and not session_key:
            raise ValueError("Either context_id or session_key is required for get_related_contexts query")
        
        # 构建拓扑记忆查询
        topology_query = {
            'limit': openclaw_query.get('limit', 20),
            'offset': openclaw_query.get('offset', 0),
            'max_degree': openclaw_query.get('max_degree', 2)
        }
        
        # 添加上下文ID或会话ID
        if context_id:
            topology_query['context_id'] = context_id
        elif session_key:
            session_id = self.session_adapter.map_session_key(session_key, create_if_missing=False)
            if session_id:
                topology_query['session_id'] = session_id
        
        # 添加关系类型过滤
        relation_types = openclaw_query.get('relation_types')
        if relation_types:
            topology_query['relation_types'] = relation_types
        
        # 执行查询
        result = self.api_client.get_related_contexts(topology_query)
        
        # 转换结果格式
        related_contexts = self._convert_query_result(result, openclaw_query)
        
        return {
            'success': True,
            'related_contexts': related_contexts,
            'count': len(related_contexts),
            'total_count': getattr(result, 'total_count', len(related_contexts)),
            'context_id': context_id,
            'session_key': session_key,
            'has_more': getattr(result, 'has_more', False)
        }
    
    def _handle_get_recent_sessions(self, openclaw_query: Dict[str, Any]) -> Dict[str, Any]:
        """处理获取最近会话查询"""
        # 获取所有会话映射
        all_mappings = self.session_adapter.get_all_mappings()
        
        # 应用过滤
        filtered_mappings = self._filter_sessions(all_mappings, openclaw_query)
        
        # 排序
        sort_by = openclaw_query.get('sort_by', 'last_accessed')
        sort_direction = openclaw_query.get('sort_direction', 'desc')
        
        filtered_mappings.sort(
            key=lambda x: x.get(sort_by, ''),
            reverse=(sort_direction == 'desc')
        )
        
        # 分页
        limit = openclaw_query.get('limit', 50)
        offset = openclaw_query.get('offset', 0)
        paginated_mappings = filtered_mappings[offset:offset + limit]
        
        # 为每个会话获取额外信息
        sessions = []
        for mapping in paginated_mappings:
            session_info = {
                'session_key': mapping.get('openclaw_key'),
                'session_id': mapping.get('session_id'),
                'channel': mapping.get('channel'),
                'account_id': mapping.get('account_id'),
                'created_at': mapping.get('created_at'),
                'last_accessed': mapping.get('last_accessed'),
                'context_count': 0  # 可以添加实际计数
            }
            sessions.append(session_info)
        
        return {
            'success': True,
            'sessions': sessions,
            'count': len(sessions),
            'total_count': len(filtered_mappings),
            'has_more': (offset + len(sessions)) < len(filtered_mappings)
        }
    
    def _handle_get_context_by_id(self, openclaw_query: Dict[str, Any]) -> Dict[str, Any]:
        """处理根据ID获取上下文查询"""
        context_id = openclaw_query.get('context_id')
        if not context_id:
            raise ValueError("context_id is required for get_context_by_id query")
        
        # 执行查询
        context = self.api_client.get_context_by_id(context_id)
        
        if not context:
            return {
                'success': False,
                'error': f"Context not found: {context_id}",
                'context_id': context_id
            }
        
        # 转换结果格式
        converted_context = self._convert_single_context(context, openclaw_query)
        
        return {
            'success': True,
            'context': converted_context,
            'context_id': context_id
        }
    
    def _handle_get_context_stats(self, openclaw_query: Dict[str, Any]) -> Dict[str, Any]:
        """处理获取上下文统计查询"""
        session_key = openclaw_query.get('session_key')
        
        # 如果有会话键，获取会话ID
        session_id = None
        if session_key:
            session_id = self.session_adapter.map_session_key(session_key, create_if_missing=False)
        
        # 构建统计查询
        stats_query = {}
        if session_id:
            stats_query['session_id'] = session_id
        
        # 添加时间范围
        time_range = openclaw_query.get('time_range')
        if time_range:
            stats_query['time_range'] = self._convert_time_range(time_range)
        
        # 执行统计查询
        stats = self.api_client.get_context_stats(stats_query)
        
        # 添加会话信息
        if session_key and session_id:
            session_info = self.session_adapter.get_openclaw_session_info(session_id)
            if session_info:
                stats['session_info'] = {
                    'session_key': session_key,
                    'channel': session_info.get('channel'),
                    'account_id': session_info.get('account_id'),
                    'created_at': session_info.get('mapping_created_at')
                }
        
        return {
            'success': True,
            'stats': stats,
            'session_key': session_key,
            'timestamp': datetime.now().isoformat()
        }
    
    def _convert_filters(self, openclaw_filters: Dict[str, Any]) -> Dict[str, Any]:
        """转换OpenClaw过滤器为拓扑记忆过滤器"""
        topology_filters = {}
        
        for key, value in openclaw_filters.items():
            # 处理特殊字段映射
            if key == 'sender':
                topology_filters['sender'] = value
            elif key == 'role':
                topology_filters['role'] = value
            elif key == 'message_type':
                # 映射消息类型到上下文类型
                context_types = self._map_message_type_to_context_type(value)
                if context_types:
                    topology_filters['context_types'] = context_types
            elif key == 'created_after':
                topology_filters['created_after'] = value
            elif key == 'created_before':
                topology_filters['created_before'] = value
            elif key == 'has_attachments':
                topology_filters['has_attachments'] = value
            elif key == 'is_important':
                topology_filters['priority_min'] = 8 if value else None
            else:
                # 其他过滤器直接传递
                topology_filters[key] = value
        
        return topology_filters
    
    def _map_message_type_to_context_type(self, message_type: Union[str, List[str]]) -> List[str]:
        """映射OpenClaw消息类型到拓扑记忆上下文类型"""
        type_mapping = {
            'text': ['user_message', 'assistant_message'],
            'command': ['command'],
            'tool_call': ['tool_call'],
            'tool_result': ['tool_result'],
            'system': ['system_message'],
            'error': ['error'],
            'heartbeat': ['heartbeat']
        }
        
        if isinstance(message_type, str):
            message_type = [message_type]
        
        context_types = []
        for mt in message_type:
            if mt in type_mapping:
                context_types.extend(type_mapping[mt])
        
        return list(set(context_types))  # 去重
    
    def _convert_time_range(self, time_range: Dict[str, Any]) -> Dict[str, Any]:
        """转换时间范围"""
        topology_time_range = {}
        
        if 'start' in time_range:
            topology_time_range['start'] = time_range['start']
        if 'end' in time_range:
            topology_time_range['end'] = time_range['end']
        if 'last_hours' in time_range:
            hours = time_range['last_hours']
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            topology_time_range['start'] = start_time.isoformat()
            topology_time_range['end'] = end_time.isoformat()
        if 'last_days' in time_range:
            days = time_range['last_days']
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            topology_time_range['start'] = start_time.isoformat()
            topology_time_range['end'] = end_time.isoformat()
        
        return topology_time_range
    
    def _filter_sessions(self, sessions: List[Dict[str, Any]], 
                        filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """过滤会话列表"""
        filtered = sessions
        
        # 按频道过滤
        channel = filters.get('channel')
        if channel:
            filtered = [s for s in filtered if s.get('channel') == channel]
        
        # 按账户ID过滤
        account_id = filters.get('account_id')
        if account_id:
            filtered = [s for s in filtered if s.get('account_id') == account_id]
        
        # 按创建时间过滤
        created_after = filters.get('created_after')
        if created_after:
            filtered = [s for s in filtered if s.get('created_at', '') >= created_after]
        
        created_before = filters.get('created_before')
        if created_before:
            filtered = [s for s in filtered if s.get('created_at', '') <= created_before]
        
        return filtered