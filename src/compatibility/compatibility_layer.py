"""
兼容层主入口点 - 提供完整的向后兼容支持
"""

import json
from datetime import datetime
from typing import Dict, Optional, Any, List, Union
import logging

from .session_adapter import SessionAdapter, SessionLifecycleManager
from .context_converter import ContextConverter, ReverseContextConverter
from .query_proxy import QueryProxy
from .error_handler import ErrorHandler

logger = logging.getLogger(__name__)


class CompatibilityLayer:
    """
    向后兼容层主类
    
    提供OpenClaw对话系统与拓扑记忆系统之间的完整兼容支持
    """
    
    def __init__(self, topology_context_manager, topology_api_client, 
                 storage_backend=None, config: Optional[Dict[str, Any]] = None):
        """
        初始化兼容层
        
        Args:
            topology_context_manager: 拓扑记忆上下文管理器
            topology_api_client: 拓扑记忆API客户端
            storage_backend: 存储后端（用于会话映射持久化）
            config: 配置字典
        """
        self.config = config or {}
        
        # 初始化组件
        self.session_adapter = SessionAdapter(storage_backend)
        self.context_converter = ContextConverter()
        self.reverse_converter = ReverseContextConverter()
        self.error_handler = ErrorHandler()
        
        # 初始化生命周期管理器
        self.session_lifecycle = SessionLifecycleManager(
            session_adapter=self.session_adapter,
            topology_context_manager=topology_context_manager
        )
        
        # 初始化查询代理
        self.query_proxy = QueryProxy(
            session_adapter=self.session_adapter,
            topology_api_client=topology_api_client,
            context_converter=self.context_converter
        )
        
        # 初始化拓扑记忆组件引用
        self.topology_context_manager = topology_context_manager
        self.topology_api_client = topology_api_client
        
        # 状态跟踪
        self.initialized = False
        self.stats = {
            'sessions_created': 0,
            'messages_converted': 0,
            'queries_executed': 0,
            'errors_handled': 0,
            'last_activity': None
        }
        
        logger.info("Compatibility layer initialized")
    
    def initialize(self) -> Dict[str, Any]:
        """
        初始化兼容层
        
        Returns:
            初始化结果
        """
        try:
            # 加载现有会话映射
            self.session_adapter._load_mappings()
            
            # 验证拓扑记忆系统连接
            if hasattr(self.topology_api_client, 'health_check'):
                health = self.topology_api_client.health_check()
                if not health.get('healthy', False):
                    logger.warning(f"Topology memory system health check failed: {health}")
            
            self.initialized = True
            self.stats['last_activity'] = datetime.now().isoformat()
            
            logger.info("Compatibility layer initialization completed")
            
            return {
                'success': True,
                'initialized': True,
                'session_count': len(self.session_adapter.get_all_mappings()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize compatibility layer: {e}")
            return {
                'success': False,
                'error': str(e),
                'initialized': False
            }
    
    def process_openclaw_message(self, openclaw_message: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理OpenClaw消息
        
        Args:
            openclaw_message: OpenClaw消息对象
            
        Returns:
            处理结果
        """
        try:
            # 更新最后活动时间
            self.stats['last_activity'] = datetime.now().isoformat()
            
            # 提取会话键
            session_key = openclaw_message.get('session_key')
            if not session_key:
                return self.error_handler.handle_error(
                    'missing_session_key',
                    "session_key is required in OpenClaw message",
                    openclaw_message
                )
            
            # 确保会话存在
            session_result = self.session_lifecycle.create_session(session_key)
            if not session_result.get('success', False):
                return session_result
            
            session_id = session_result['session_id']
            
            # 获取会话信息
            session_info = self.session_adapter.get_openclaw_session_info(session_id)
            
            # 转换消息为拓扑记忆上下文
            topology_context = self.context_converter.convert_openclaw_message(
                openclaw_message, 
                session_info
            )
            
            # 保存到拓扑记忆
            context_entry = self.topology_context_manager.create_context(
                session_id=session_id,
                user_id=session_result.get('user_id', 'unknown'),
                context_type=topology_context.get('context_type', 'user_message'),
                content=topology_context.get('content', {}),
                metadata=topology_context.get('metadata', {}),
                priority=topology_context.get('priority', 5),
                ttl=topology_context.get('ttl', 3600)
            )
            
            # 更新统计
            self.stats['messages_converted'] += 1
            
            logger.debug(f"Processed OpenClaw message for session {session_key}")
            
            return {
                'success': True,
                'session_key': session_key,
                'session_id': session_id,
                'context_id': getattr(context_entry, 'id', None),
                'converted_at': datetime.now().isoformat(),
                'message_type': openclaw_message.get('type', 'unknown')
            }
            
        except Exception as e:
            self.stats['errors_handled'] += 1
            logger.error(f"Failed to process OpenClaw message: {e}")
            
            return self.error_handler.handle_error(
                'message_processing_error',
                str(e),
                openclaw_message
            )
    
    def execute_openclaw_query(self, openclaw_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行OpenClaw查询
        
        Args:
            openclaw_query: OpenClaw查询对象
            
        Returns:
            查询结果
        """
        try:
            # 更新最后活动时间
            self.stats['last_activity'] = datetime.now().isoformat()
            
            # 执行查询
            result = self.query_proxy.execute_openclaw_query(openclaw_query)
            
            # 更新统计
            self.stats['queries_executed'] += 1
            
            return result
            
        except Exception as e:
            self.stats['errors_handled'] += 1
            logger.error(f"Failed to execute OpenClaw query: {e}")
            
            return self.error_handler.handle_error(
                'query_execution_error',
                str(e),
                openclaw_query
            )
    
    def get_session_contexts(self, session_key: str, limit: int = 50, 
                            offset: int = 0, **kwargs) -> Dict[str, Any]:
        """
        获取会话上下文（简化接口）
        
        Args:
            session_key: OpenClaw会话键
            limit: 返回数量限制
            offset: 偏移量
            **kwargs: 其他查询参数
            
        Returns:
            上下文列表
        """
        query = {
            'type': 'get_context',
            'session_key': session_key,
            'limit': limit,
            'offset': offset,
            **kwargs
        }
        
        return self.execute_openclaw_query(query)
    
    def search_in_session(self, session_key: str, search_text: str, 
                         limit: int = 100, **kwargs) -> Dict[str, Any]:
        """
        在会话中搜索（简化接口）
        
        Args:
            session_key: OpenClaw会话键
            search_text: 搜索文本
            limit: 返回数量限制
            **kwargs: 其他搜索参数
            
        Returns:
            搜索结果
        """
        query = {
            'type': 'search_messages',
            'session_key': session_key,
            'search_text': search_text,
            'limit': limit,
            **kwargs
        }
        
        return self.execute_openclaw_query(query)
    
    def create_session(self, session_key: str, user_id: Optional[str] = None, 
                      initial_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        创建新会话（简化接口）
        
        Args:
            session_key: OpenClaw会话键
            user_id: 用户ID
            initial_context: 初始上下文
            
        Returns:
            会话创建结果
        """
        try:
            result = self.session_lifecycle.create_session(
                session_key, 
                user_id, 
                initial_context
            )
            
            if result.get('success', False):
                self.stats['sessions_created'] += 1
            
            return result
            
        except Exception as e:
            self.stats['errors_handled'] += 1
            logger.error(f"Failed to create session: {e}")
            
            return self.error_handler.handle_error(
                'session_creation_error',
                str(e),
                {'session_key': session_key}
            )
    
    def close_session(self, session_key: str, reason: str = 'user_ended') -> Dict[str, Any]:
        """
        关闭会话（简化接口）
        
        Args:
            session_key: OpenClaw会话键
            reason: 关闭原因
            
        Returns:
            会话关闭结果
        """
        try:
            return self.session_lifecycle.close_session(session_key, reason)
            
        except Exception as e:
            self.stats['errors_handled'] += 1
            logger.error(f"Failed to close session: {e}")
            
            return self.error_handler.handle_error(
                'session_close_error',
                str(e),
                {'session_key': session_key, 'reason': reason}
            )
    
    def get_session_status(self, session_key: str) -> Dict[str, Any]:
        """
        获取会话状态（简化接口）
        
        Args:
            session_key: OpenClaw会话键
            
        Returns:
            会话状态
        """
        try:
            return self.session_lifecycle.get_session_status(session_key)
            
        except Exception as e:
            self.stats['errors_handled'] += 1
            logger.error(f"Failed to get session status: {e}")
            
            return self.error_handler.handle_error(
                'session_status_error',
                str(e),
                {'session_key': session_key}
            )
    
    def batch_process_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        批量处理消息
        
        Args:
            messages: OpenClaw消息列表
            
        Returns:
            批量处理结果
        """
        results = []
        errors = []
        
        for i, message in enumerate(messages):
            try:
                result = self.process_openclaw_message(message)
                results.append(result)
                
                if not result.get('success', False):
                    errors.append({
                        'index': i,
                        'message_id': message.get('id', f'unknown_{i}'),
                        'error': result.get('error')
                    })
                    
            except Exception as e:
                errors.append({
                    'index': i,
                    'message_id': message.get('id', f'unknown_{i}'),
                    'error': str(e)
                })
                results.append({
                    'success': False,
                    'error': str(e),
                    'message_id': message.get('id', f'unknown_{i}')
                })
        
        # 更新统计
        self.stats['messages_converted'] += len(results)
        self.stats['errors_handled'] += len(errors)
        
        return {
            'success': len(errors) == 0,
            'total_processed': len(messages),
            'successful': len(results) - len(errors),
            'failed': len(errors),
            'results': results,
            'errors': errors if errors else None,
            'timestamp': datetime.now().isoformat()
        }
    
    def export_session_to_openclaw_format(self, session_key: str, 
                                         limit: int = 1000) -> Dict[str, Any]:
        """
        导出会话为OpenClaw格式
        
        Args:
            session_key: OpenClaw会话键
            limit: 导出数量限制
            
        Returns:
            导出结果
        """
        try:
            # 获取所有上下文
            query_result = self.get_session_contexts(session_key, limit=limit)
            
            if not query_result.get('success', False):
                return query_result
            
            contexts = query_result.get('contexts', [])
            
            # 转换为OpenClaw消息格式
            openclaw_messages = []
            for context in contexts:
                try:
                    # 获取会话信息
                    session_id = query_result.get('session_id')
                    session_info = self.session_adapter.get_openclaw_session_info(session_id)
                    
                    # 转换上下文
                    message = self.reverse_converter.convert_topology_context(
                        context, 
                        session_info
                    )
                    
                    openclaw_messages.append(message)
                    
                except Exception as e:
                    logger.warning(f"Failed to convert context: {e}")
                    # 跳过转换失败的消息
            
            return {
                'success': True,
                'session_key': session_key,
                'message_count': len(openclaw_messages),
                'total_contexts': len(contexts),
                'messages': openclaw_messages,
                'exported_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.stats['errors_handled'] += 1
            logger.error(f"Failed to export session: {e}")
            
            return self.error_handler.handle_error(
                'session_export_error',
                str(e),
                {'session_key': session_key}
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取兼容层统计信息
        
        Returns:
            统计信息
        """
        return {
            'initialized': self.initialized,
            'stats': self.stats.copy(),
            'session_count': len(self.session_adapter.get_all_mappings()),
            'component_status': {
                'session_adapter': True,
                'context_converter': True,
                'query_proxy': True,
                'error_handler': True
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def cleanup_old_sessions(self, days_old: int = 30) -> Dict[str, Any]:
        """
        清理旧的会话映射
        
        Args:
            days_old: 清理多少天前的映射
            
        Returns:
            清理结果
        """
        try:
            removed_count = self.session_adapter.cleanup_old_mappings(days_old)
            
            return {
                'success': True,
                'removed_count': removed_count,
                'days_old': days_old,
                'remaining_count': len(self.session_adapter.get_all_mappings()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            
            return self.error_handler.handle_error(
                'cleanup_error',
                str(e),
                {'days_old': days_old}
            )
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态
        """
        try:
            # 检查组件状态
            components_healthy = True
            component_details = {}
            
            # 检查会话适配器
            try:
                session_count = len(self.session_adapter.get_all_mappings())
                component_details['session_adapter'] = {
                    'healthy': True,
                    'session_count': session_count
                }
            except Exception as e:
                components_healthy = False
                component_details['session_adapter'] = {
                    'healthy': False,
                    'error': str(e)
                }
            
            # 检查拓扑记忆连接
            try:
                if hasattr(self.topology_api_client, 'health_check'):
                    topology_health = self.topology_api_client.health_check()
                    component_details['topology_memory'] = topology_health
                    if not topology_health.get('healthy', False):
                        components_healthy = False
                else:
                    component_details['topology_memory'] = {
                        'healthy': True,
                        'message': 'Health check not implemented'
                    }
            except Exception as e:
                components_healthy = False
                component_details['topology_memory'] = {
                    'healthy': False,
                    'error': str(e)
                }
            
            return {
                'healthy': components_healthy and self.initialized,
                'initialized': self.initialized,
                'components': component_details,
                'stats': self.get_stats(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }