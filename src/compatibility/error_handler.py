"""
错误处理器 - 统一处理兼容层中的错误
"""

import traceback
from datetime import datetime
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ErrorHandler:
    """错误处理器，统一处理兼容层中的错误"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化错误处理器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 错误码映射
        self.error_codes = {
            # 会话相关错误
            'missing_session_key': {
                'code': 4001,
                'message': 'Session key is required',
                'severity': 'error'
            },
            'invalid_session_key': {
                'code': 4002,
                'message': 'Invalid session key format',
                'severity': 'error'
            },
            'session_not_found': {
                'code': 4004,
                'message': 'Session not found',
                'severity': 'warning'
            },
            'session_creation_error': {
                'code': 4005,
                'message': 'Failed to create session',
                'severity': 'error'
            },
            'session_close_error': {
                'code': 4006,
                'message': 'Failed to close session',
                'severity': 'warning'
            },
            
            # 消息处理错误
            'message_processing_error': {
                'code': 4101,
                'message': 'Failed to process message',
                'severity': 'error'
            },
            'message_conversion_error': {
                'code': 4102,
                'message': 'Failed to convert message',
                'severity': 'error'
            },
            'invalid_message_format': {
                'code': 4103,
                'message': 'Invalid message format',
                'severity': 'error'
            },
            
            # 查询相关错误
            'query_execution_error': {
                'code': 4201,
                'message': 'Failed to execute query',
                'severity': 'error'
            },
            'invalid_query_format': {
                'code': 4202,
                'message': 'Invalid query format',
                'severity': 'error'
            },
            'query_timeout': {
                'code': 4203,
                'message': 'Query timeout',
                'severity': 'warning'
            },
            
            # 转换相关错误
            'context_conversion_error': {
                'code': 4301,
                'message': 'Failed to convert context',
                'severity': 'error'
            },
            'reverse_conversion_error': {
                'code': 4302,
                'message': 'Failed to convert back to OpenClaw format',
                'severity': 'warning'
            },
            
            # 拓扑记忆系统错误
            'topology_connection_error': {
                'code': 4401,
                'message': 'Failed to connect to topology memory system',
                'severity': 'error'
            },
            'topology_query_error': {
                'code': 4402,
                'message': 'Topology memory query failed',
                'severity': 'error'
            },
            'topology_storage_error': {
                'code': 4403,
                'message': 'Topology memory storage error',
                'severity': 'error'
            },
            
            # 兼容层内部错误
            'initialization_error': {
                'code': 4501,
                'message': 'Compatibility layer initialization failed',
                'severity': 'error'
            },
            'component_error': {
                'code': 4502,
                'message': 'Component error',
                'severity': 'error'
            },
            'cleanup_error': {
                'code': 4503,
                'message': 'Cleanup operation failed',
                'severity': 'warning'
            },
            
            # 通用错误
            'internal_error': {
                'code': 5000,
                'message': 'Internal server error',
                'severity': 'error'
            },
            'not_implemented': {
                'code': 5001,
                'message': 'Feature not implemented',
                'severity': 'info'
            },
            'validation_error': {
                'code': 5002,
                'message': 'Validation failed',
                'severity': 'error'
            }
        }
        
        # 错误日志配置
        self.log_errors = self.config.get('log_errors', True)
        self.log_stack_traces = self.config.get('log_stack_traces', False)
        
        logger.info("Error handler initialized")
    
    def handle_error(self, error_type: str, error_message: str = None, 
                    context: Optional[Dict[str, Any]] = None, 
                    exception: Optional[Exception] = None) -> Dict[str, Any]:
        """
        处理错误
        
        Args:
            error_type: 错误类型
            error_message: 错误消息（可选，覆盖默认消息）
            context: 错误上下文信息
            exception: 异常对象（可选）
            
        Returns:
            标准化的错误响应
        """
        # 获取错误码信息
        error_info = self.error_codes.get(error_type, self.error_codes['internal_error'])
        
        # 使用提供的消息或默认消息
        if error_message is None:
            error_message = error_info['message']
        
        # 构建错误响应
        error_response = {
            'success': False,
            'error': {
                'code': error_info['code'],
                'type': error_type,
                'message': error_message,
                'severity': error_info['severity'],
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # 添加上下文信息
        if context:
            error_response['error']['context'] = context
        
        # 添加异常信息（如果提供）
        if exception:
            error_response['error']['exception'] = {
                'type': type(exception).__name__,
                'message': str(exception)
            }
            
            # 记录堆栈跟踪（如果配置允许）
            if self.log_stack_traces:
                stack_trace = traceback.format_exc()
                error_response['error']['stack_trace'] = stack_trace
        
        # 记录错误日志
        if self.log_errors:
            self._log_error(error_type, error_message, error_info['severity'], 
                           context, exception)
        
        return error_response
    
    def _log_error(self, error_type: str, error_message: str, severity: str,
                  context: Optional[Dict[str, Any]], exception: Optional[Exception]):
        """记录错误日志"""
        log_message = f"[{severity.upper()}] {error_type}: {error_message}"
        
        if context:
            # 简化上下文信息用于日志
            context_summary = {}
            for key, value in context.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    context_summary[key] = value
                else:
                    context_summary[key] = type(value).__name__
            
            log_message += f" | Context: {context_summary}"
        
        # 根据严重性选择日志级别
        if severity == 'error':
            logger.error(log_message, exc_info=exception)
        elif severity == 'warning':
            logger.warning(log_message, exc_info=exception)
        elif severity == 'info':
            logger.info(log_message)
        else:
            logger.debug(log_message)
    
    def wrap_operation(self, operation_func, error_type: str, 
                      context: Optional[Dict[str, Any]] = None):
        """
        包装操作函数，自动处理异常
        
        Args:
            operation_func: 要执行的操作函数
            error_type: 错误类型
            context: 错误上下文
            
        Returns:
            操作结果或错误响应
        """
        try:
            return operation_func()
        except Exception as e:
            return self.handle_error(error_type, str(e), context, e)
    
    def validate_session_key(self, session_key: str) -> Dict[str, Any]:
        """
        验证会话键格式
        
        Args:
            session_key: 要验证的会话键
            
        Returns:
            验证结果
        """
        if not session_key:
            return self.handle_error(
                'missing_session_key',
                context={'field': 'session_key'}
            )
        
        # 检查会话键格式（channel:accountId:conversationId）
        parts = session_key.split(':')
        if len(parts) != 3:
            return self.handle_error(
                'invalid_session_key',
                f"Expected format 'channel:accountId:conversationId', got '{session_key}'",
                context={'session_key': session_key}
            )
        
        # 检查各部分是否为空
        channel, account_id, conversation_id = parts
        if not channel or not account_id or not conversation_id:
            return self.handle_error(
                'invalid_session_key',
                "Session key parts cannot be empty",
                context={
                    'session_key': session_key,
                    'channel': channel,
                    'account_id': account_id,
                    'conversation_id': conversation_id
                }
            )
        
        return {
            'success': True,
            'valid': True,
            'parsed': {
                'channel': channel,
                'account_id': account_id,
                'conversation_id': conversation_id
            }
        }
    
    def validate_message_format(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证消息格式
        
        Args:
            message: 要验证的消息
            
        Returns:
            验证结果
        """
        if not isinstance(message, dict):
            return self.handle_error(
                'invalid_message_format',
                "Message must be a dictionary",
                context={'message_type': type(message).__name__}
            )
        
        # 检查必需字段
        required_fields = ['type', 'content']
        missing_fields = []
        
        for field in required_fields:
            if field not in message:
                missing_fields.append(field)
        
        if missing_fields:
            return self.handle_error(
                'invalid_message_format',
                f"Missing required fields: {', '.join(missing_fields)}",
                context={'missing_fields': missing_fields}
            )
        
        # 检查会话键（如果存在）
        if 'session_key' in message:
            validation_result = self.validate_session_key(message['session_key'])
            if not validation_result.get('success', False):
                return validation_result
        
        return {
            'success': True,
            'valid': True,
            'message_type': message.get('type')
        }
    
    def validate_query_format(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证查询格式
        
        Args:
            query: 要验证的查询
            
        Returns:
            验证结果
        """
        if not isinstance(query, dict):
            return self.handle_error(
                'invalid_query_format',
                "Query must be a dictionary",
                context={'query_type': type(query).__name__}
            )
        
        # 检查查询类型
        query_type = query.get('type')
        if not query_type:
            return self.handle_error(
                'invalid_query_format',
                "Query type is required",
                context={'query': query}
            )
        
        # 根据查询类型验证必需字段
        validation_rules = {
            'get_context': ['session_key'],
            'search_messages': ['search_text'],
            'get_session_history': ['session_key'],
            'get_context_by_id': ['context_id'],
            'get_related_contexts': []  # 需要context_id或session_key
        }
        
        if query_type in validation_rules:
            required_fields = validation_rules[query_type]
            missing_fields = []
            
            for field in required_fields:
                if field not in query:
                    missing_fields.append(field)
            
            # 特殊处理：get_related_contexts需要context_id或session_key
            if query_type == 'get_related_contexts':
                if 'context_id' not in query and 'session_key' not in query:
                    missing_fields.append('context_id or session_key')
            
            if missing_fields:
                return self.handle_error(
                    'invalid_query_format',
                    f"Query type '{query_type}' requires fields: {', '.join(missing_fields)}",
                    context={
                        'query_type': query_type,
                        'missing_fields': missing_fields
                    }
                )
        
        # 验证会话键（如果存在）
        if 'session_key' in query:
            validation_result = self.validate_session_key(query['session_key'])
            if not validation_result.get('success', False):
                return validation_result
        
        return {
            'success': True,
            'valid': True,
            'query_type': query_type
        }
    
    def create_success_response(self, data: Dict[str, Any] = None, 
                               message: str = None) -> Dict[str, Any]:
        """
        创建成功响应
        
        Args:
            data: 响应数据
            message: 成功消息
            
        Returns:
            标准化的成功响应
        """
        response = {
            'success': True,
            'timestamp': datetime.now().isoformat()
        }
        
        if data:
            response.update(data)
        
        if message:
            response['message'] = message
        
        return response
    
    def get_error_codes(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有错误码定义
        
        Returns:
            错误码定义字典
        """
        return self.error_codes.copy()
    
    def add_custom_error_code(self, error_type: str, code: int, 
                             message: str, severity: str = 'error'):
        """
        添加自定义错误码
        
        Args:
            error_type: 错误类型标识符
            code: 错误码
            message: 错误消息
            severity: 严重性级别（error/warning/info/debug）
        """
        if error_type in self.error_codes:
            logger.warning(f"Overwriting existing error type: {error_type}")
        
        self.error_codes[error_type] = {
            'code': code,
            'message': message,
            'severity': severity
        }
        
        logger.info(f"Added custom error code: {error_type} ({code})")