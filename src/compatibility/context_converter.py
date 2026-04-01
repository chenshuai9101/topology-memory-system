"""
上下文转换器 - 处理OpenClaw消息和拓扑记忆上下文之间的转换
"""

import uuid
from datetime import datetime
from typing import Dict, Optional, Any, List
import logging

logger = logging.getLogger(__name__)


class ContextConverter:
    """上下文转换器，将OpenClaw消息转换为拓扑记忆上下文"""
    
    def __init__(self):
        # 注册消息类型转换器
        self.message_converters = {
            'text': self._convert_text_message,
            'image': self._convert_image_message,
            'file': self._convert_file_message,
            'command': self._convert_command_message,
            'tool_call': self._convert_tool_call_message,
            'tool_result': self._convert_tool_result_message,
            'system': self._convert_system_message,
            'error': self._convert_error_message,
            'heartbeat': self._convert_heartbeat_message
        }
        
        # 优先级配置
        self.priority_config = {
            'command': 9,
            'tool_call': 8,
            'tool_result': 7,
            'system': 6,
            'error': 6,
            'heartbeat': 1,
            'default': 5
        }
        
        # TTL配置（秒）
        self.ttl_config = {
            'command': 1800,      # 30分钟
            'tool_call': 7200,    # 2小时
            'tool_result': 7200,  # 2小时
            'system': 86400,      # 24小时
            'error': 3600,        # 1小时
            'heartbeat': 300,     # 5分钟
            'default': 3600       # 1小时
        }
    
    def convert_openclaw_message(self, openclaw_message: Dict[str, Any], 
                                session_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        将OpenClaw消息转换为拓扑记忆上下文
        
        Args:
            openclaw_message: OpenClaw消息对象
            session_info: 会话信息
            
        Returns:
            拓扑记忆上下文对象
        """
        try:
            # 确定消息类型
            message_type = openclaw_message.get('type', 'text')
            
            # 获取对应的转换器
            converter = self.message_converters.get(message_type)
            if not converter:
                logger.warning(f"Unknown message type '{message_type}', using text converter")
                converter = self._convert_text_message
            
            # 执行转换
            topology_context = converter(openclaw_message, session_info)
            
            # 添加通用元数据
            metadata = topology_context.get('metadata', {})
            metadata.update({
                'openclaw_message_id': openclaw_message.get('id'),
                'openclaw_timestamp': openclaw_message.get('timestamp'),
                'openclaw_source': openclaw_message.get('source', 'unknown'),
                'openclaw_message_type': message_type,
                'converted_at': datetime.now().isoformat(),
                'original_message': openclaw_message.get('content', '')[:100]  # 保留部分原始内容
            })
            
            topology_context['metadata'] = metadata
            
            # 确保必要的字段存在
            if 'priority' not in topology_context:
                topology_context['priority'] = self._calculate_priority(openclaw_message)
            
            if 'ttl' not in topology_context:
                topology_context['ttl'] = self._calculate_ttl(openclaw_message)
            
            logger.debug(f"Converted OpenClaw message to topology context: {message_type}")
            return topology_context
            
        except Exception as e:
            logger.error(f"Failed to convert OpenClaw message: {e}")
            # 返回一个基本的错误上下文
            return self._create_error_context(
                openclaw_message, 
                session_info, 
                f"Conversion error: {str(e)}"
            )
    
    def _convert_text_message(self, message: Dict[str, Any], session_info: Dict[str, Any]) -> Dict[str, Any]:
        """转换文本消息"""
        content = message.get('content', '')
        sender = message.get('sender', 'unknown')
        role = message.get('role', 'user')
        
        # 确定上下文类型
        if role == 'system':
            context_type = 'system_message'
        elif role == 'assistant':
            context_type = 'assistant_message'
        else:
            context_type = 'user_message'
        
        return {
            'context_type': context_type,
            'content': {
                'text': content,
                'sender': sender,
                'role': role,
                'message_type': 'text'
            },
            'metadata': {
                'has_attachments': bool(message.get('attachments')),
                'is_edited': message.get('is_edited', False),
                'is_important': message.get('is_important', False),
                'language': message.get('language', 'unknown')
            }
        }
    
    def _convert_tool_call_message(self, message: Dict[str, Any], session_info: Dict[str, Any]) -> Dict[str, Any]:
        """转换工具调用消息"""
        tool_name = message.get('tool_name', 'unknown')
        tool_args = message.get('tool_args', {})
        call_id = message.get('call_id', str(uuid.uuid4()))
        
        return {
            'context_type': 'tool_call',
            'content': {
                'tool_name': tool_name,
                'tool_args': tool_args,
                'call_id': call_id,
                'sender': message.get('sender', 'assistant'),
                'role': 'assistant'
            },
            'metadata': {
                'tool_category': message.get('tool_category', 'general'),
                'requires_result': True,
                'is_async': message.get('is_async', False),
                'timeout': message.get('timeout')
            },
            'priority': self.priority_config['tool_call'],
            'ttl': self.ttl_config['tool_call']
        }
    
    def _convert_tool_result_message(self, message: Dict[str, Any], session_info: Dict[str, Any]) -> Dict[str, Any]:
        """转换工具结果消息"""
        call_id = message.get('call_id')
        result = message.get('result', {})
        success = message.get('success', True)
        
        return {
            'context_type': 'tool_result',
            'content': {
                'call_id': call_id,
                'result': result,
                'success': success,
                'error': message.get('error'),
                'sender': message.get('sender', 'system'),
                'role': 'system'
            },
            'metadata': {
                'tool_name': message.get('tool_name'),
                'execution_time': message.get('execution_time'),
                'cached': message.get('cached', False)
            },
            'priority': self.priority_config['tool_result'],
            'ttl': self.ttl_config['tool_result']
        }
    
    def _convert_command_message(self, message: Dict[str, Any], session_info: Dict[str, Any]) -> Dict[str, Any]:
        """转换命令消息"""
        command = message.get('command', '')
        args = message.get('args', [])
        
        return {
            'context_type': 'command',
            'content': {
                'command': command,
                'args': args,
                'sender': message.get('sender', 'user'),
                'role': 'user'
            },
            'metadata': {
                'requires_response': message.get('requires_response', True),
                'is_admin': message.get('is_admin', False),
                'channel': session_info.get('channel', 'unknown')
            },
            'priority': self.priority_config['command'],
            'ttl': self.ttl_config['command']
        }
    
    def _convert_system_message(self, message: Dict[str, Any], session_info: Dict[str, Any]) -> Dict[str, Any]:
        """转换系统消息"""
        return {
            'context_type': 'system_message',
            'content': {
                'text': message.get('content', ''),
                'sender': 'system',
                'role': 'system',
                'message_type': 'system'
            },
            'metadata': {
                'system_event': message.get('event'),
                'is_persistent': message.get('is_persistent', True)
            },
            'priority': self.priority_config['system'],
            'ttl': self.ttl_config['system']
        }
    
    def _convert_error_message(self, message: Dict[str, Any], session_info: Dict[str, Any]) -> Dict[str, Any]:
        """转换错误消息"""
        return {
            'context_type': 'error',
            'content': {
                'error': message.get('error', 'Unknown error'),
                'code': message.get('code'),
                'details': message.get('details'),
                'sender': message.get('sender', 'system'),
                'role': 'system'
            },
            'metadata': {
                'is_fatal': message.get('is_fatal', False),
                'can_retry': message.get('can_retry', True),
                'component': message.get('component', 'unknown')
            },
            'priority': self.priority_config['error'],
            'ttl': self.ttl_config['error']
        }
    
    def _convert_heartbeat_message(self, message: Dict[str, Any], session_info: Dict[str, Any]) -> Dict[str, Any]:
        """转换心跳消息"""
        return {
            'context_type': 'heartbeat',
            'content': {
                'type': 'heartbeat',
                'timestamp': message.get('timestamp', datetime.now().isoformat()),
                'status': message.get('status', 'ok'),
                'sender': 'system',
                'role': 'system'
            },
            'metadata': {
                'interval': message.get('interval'),
                'is_manual': message.get('is_manual', False)
            },
            'priority': self.priority_config['heartbeat'],
            'ttl': self.ttl_config['heartbeat']
        }
    
    def _convert_image_message(self, message: Dict[str, Any], session_info: Dict[str, Any]) -> Dict[str, Any]:
        """转换图片消息（简化版）"""
        return {
            'context_type': 'media_message',
            'content': {
                'media_type': 'image',
                'url': message.get('url'),
                'caption': message.get('caption', ''),
                'sender': message.get('sender', 'user'),
                'role': 'user'
            },
            'metadata': {
                'file_size': message.get('file_size'),
                'dimensions': message.get('dimensions'),
                'format': message.get('format')
            }
        }
    
    def _convert_file_message(self, message: Dict[str, Any], session_info: Dict[str, Any]) -> Dict[str, Any]:
        """转换文件消息（简化版）"""
        return {
            'context_type': 'media_message',
            'content': {
                'media_type': 'file',
                'url': message.get('url'),
                'filename': message.get('filename'),
                'description': message.get('description', ''),
                'sender': message.get('sender', 'user'),
                'role': 'user'
            },
            'metadata': {
                'file_size': message.get('file_size'),
                'file_type': message.get('file_type'),
                'is_downloadable': message.get('is_downloadable', True)
            }
        }
    
    def _create_error_context(self, original_message: Dict[str, Any], 
                             session_info: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """创建错误上下文"""
        return {
            'context_type': 'conversion_error',
            'content': {
                'original_message': original_message,
                'error': error_message,
                'sender': 'system',
                'role': 'system'
            },
            'metadata': {
                'original_type': original_message.get('type', 'unknown'),
                'conversion_failed': True
            },
            'priority': 3,  # 低优先级
            'ttl': 600  # 10分钟
        }
    
    def _calculate_priority(self, message: Dict[str, Any]) -> int:
        """计算上下文优先级"""
        message_type = message.get('type', 'text')
        
        # 检查消息是否标记为重要
        if message.get('is_important', False):
            return 8
        
        # 根据消息类型返回优先级
        return self.priority_config.get(message_type, self.priority_config['default'])
    
    def _calculate_ttl(self, message: Dict[str, Any]) -> int:
        """计算TTL（生存时间）"""
        message_type = message.get('type', 'text')
        
        # 检查消息是否标记为持久
        if message.get('is_persistent', False):
            return 86400  # 24小时
        
        # 根据消息类型返回TTL
        return self.ttl_config.get(message_type, self.ttl_config['default'])
    
    def batch_convert(self, messages: List[Dict[str, Any]], 
                     session_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        批量转换消息
        
        Args:
            messages: OpenClaw消息列表
            session_info: 会话信息
            
        Returns:
            拓扑记忆上下文列表
        """
        results = []
        errors = []
        
        for i, message in enumerate(messages):
            try:
                context = self.convert_openclaw_message(message, session_info)
                results.append(context)
            except Exception as e:
                logger.error(f"Failed to convert message at index {i}: {e}")
                errors.append({
                    'index': i,
                    'message': message.get('id', f'unknown_{i}'),
                    'error': str(e)
                })
                # 创建错误上下文
                error_context = self._create_error_context(
                    message, 
                    session_info, 
                    f"Batch conversion error: {str(e)}"
                )
                results.append(error_context)
        
        if errors:
            logger.warning(f"Batch conversion completed with {len(errors)} errors")
        
        return results


class ReverseContextConverter:
    """反向上下文转换器，将拓扑记忆上下文转换为OpenClaw消息"""
    
    def __init__(self):
        # 注册上下文类型转换器
        self.context_converters = {
            'user_message': self._convert_to_text_message,
            'assistant_message': self._convert_to_text_message,
            'system_message': self._convert_to_system_message,
            'tool_call': self._convert_to_tool_call_message,
            'tool_result': self._convert_to_tool_result_message,
            'command': self._convert_to_command_message,
            'error': self._convert_to_error_message,
            'heartbeat': self._convert_to_heartbeat_message,
            'media_message': self._convert_to_media_message,
            'session_init': self._convert_to_session_message,
            'session_close': self._convert_to_session_message,
            'conversion_error': self._convert_to_error_message
        }
    
    def convert_topology_context(self, topology_context: Dict[str, Any], 
                                session_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        将拓扑记忆上下文转换为OpenClaw消息
        
        Args:
            topology_context: 拓扑记忆上下文对象
            session_info: 会话信息
            
        Returns:
            OpenClaw消息对象
        """
        try:
            context_type = topology_context.get('context_type', 'user_message')
            
            # 获取对应的转换器
            converter = self.context_converters.get(context_type)
            if not converter:
                logger.warning(f"Unknown context type '{context_type}', using text converter")
                converter = self._convert_to_text_message
            
            # 执行转换
            openclaw_message = converter(topology_context, session_info)
            
            # 添加通用字段
            openclaw_message.update({
                'id': topology_context.get('id', str(uuid.uuid4())),
                'timestamp': topology_context.get('created_at', datetime.now().isoformat()),
                'source': 'topology_memory',
                'metadata': {
                    'topology_context_id': topology_context.get('id'),
                    'context_type': context_type,
                    'priority': topology_context.get('priority', 5),
                    'ttl': topology_context.get('ttl'),
                    'converted_at': datetime.now().isoformat()
                }
            })
            
            logger.debug(f"Converted topology context to OpenClaw message: {context_type}")
            return openclaw_message
            
        except Exception as e:
            logger.error(f"Failed to convert topology context: {e}")
            # 返回一个基本的错误消息
            return self._create_error_message(
                topology_context, 
                session_info, 
                f"Reverse conversion error: {str(e)}"
            )
    
    def _convert_to_text_message(self, context: Dict[str, Any], session_info: Dict[str, Any]) -> Dict[str, Any]:
        """转换文本上下文"""
        content = context.get('content', {})
        context_type = context.get('context_type', 'user_message')
        
        # 确定消息类型
        if context_type == 'assistant_message':
            role = 'assistant'
            sender = content.get('sender', 'assistant')
        elif context_type == 'system_message':
            role = 'system'
            sender = content.get('sender', 'system')
        else:
            role = 'user'
            sender = content.get('sender', 'user')
        
        return {
            'type': 'text',
            'content': content.get('text', ''),
            'sender': sender,
            'role': role
        }
    
    def _convert_to_tool_call_message(self, context: Dict[str, Any], session_info: Dict[str, Any]) -> Dict[str, Any]:
        """转换工具调用上下文"""
        content = context.get('content', {})
        
        return {
            'type': 'tool_call',
            'tool_name': content.get('tool_name'),
            'tool_args': content.get('tool_args', {}),
            'call_id': content.get('call_id'),
            'sender': content.get('sender', 'assistant'),
            'role': 'assistant'
        }
    
    def _convert_to_tool_result_message(self, context: Dict[str, Any], session_info: Dict[str, Any]) -> Dict[str, Any]:
        """转换工具结果上下文"""
        content = context.get('content', {})
        
        return