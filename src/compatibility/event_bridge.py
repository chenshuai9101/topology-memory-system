"""
事件桥接器 - 处理OpenClaw事件和拓扑记忆事件之间的转换
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Optional, Any, List, Callable
import logging

logger = logging.getLogger(__name__)


class EventBridge:
    """事件桥接器，处理OpenClaw事件到拓扑记忆事件的转换"""
    
    def __init__(self, compatibility_layer, config: Optional[Dict[str, Any]] = None):
        """
        初始化事件桥接器
        
        Args:
            compatibility_layer: 兼容层实例
            config: 配置字典
        """
        self.compatibility_layer = compatibility_layer
        self.config = config or {}
        
        # 事件处理器映射
        self.event_handlers = {
            'message_received': self._handle_message_received,
            'session_created': self._handle_session_created,
            'session_closed': self._handle_session_closed,
            'tool_called': self._handle_tool_called,
            'tool_completed': self._handle_tool_completed,
            'error_occurred': self._handle_error_occurred,
            'heartbeat': self._handle_heartbeat
        }
        
        # 事件订阅者
        self.subscribers = []
        
        # 事件队列
        self.event_queue = asyncio.Queue()
        
        # 处理任务
        self.processing_task = None
        
        logger.info("Event bridge initialized")
    
    async def start(self):
        """启动事件桥接器"""
        self.processing_task = asyncio.create_task(self._process_events())
        logger.info("Event bridge started")
    
    async def stop(self):
        """停止事件桥接器"""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Event bridge stopped")
    
    async def process_openclaw_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理OpenClaw事件
        
        Args:
            event: OpenClaw事件对象
            
        Returns:
            处理结果
        """
        try:
            event_type = event.get('type')
            if not event_type:
                return {
                    'success': False,
                    'error': 'Event type is required',
                    'event': event
                }
            
            # 获取事件处理器
            handler = self.event_handlers.get(event_type)
            if not handler:
                logger.warning(f"No handler for event type: {event_type}")
                return {
                    'success': False,
                    'error': f'Unsupported event type: {event_type}',
                    'event': event
                }
            
            # 处理事件
            result = await handler(event)
            
            # 通知订阅者
            await self._notify_subscribers(event_type, event, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process OpenClaw event: {e}")
            return {
                'success': False,
                'error': str(e),
                'event': event
            }
    
    async def _handle_message_received(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """处理消息接收事件"""
        message = event.get('message')
        if not message:
            return {
                'success': False,
                'error': 'Message is required for message_received event'
            }
        
        # 通过兼容层处理消息
        result = self.compatibility_layer.process_openclaw_message(message)
        
        # 创建拓扑记忆事件
        topology_event = {
            'type': 'context_created',
            'context_id': result.get('context_id'),
            'session_id': result.get('session_id'),
            'session_key': result.get('session_key'),
            'message_type': message.get('type'),
            'timestamp': datetime.now().isoformat()
        }
        
        # 将事件放入队列
        await self.event_queue.put({
            'type': 'topology_event',
            'event': topology_event,
            'original_event': event
        })
        
        return {
            'success': True,
            'result': result,
            'topology_event': topology_event
        }
    
    async def _handle_session_created(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """处理会话创建事件"""
        session_key = event.get('session_key')
        if not session_key:
            return {
                'success': False,
                'error': 'session_key is required for session_created event'
            }
        
        # 通过兼容层创建会话
        result = self.compatibility_layer.create_session(session_key)
        
        # 创建拓扑记忆事件
        topology_event = {
            'type': 'session_created',
            'session_id': result.get('session_id'),
            'session_key': session_key,
            'timestamp': datetime.now().isoformat()
        }
        
        # 将事件放入队列
        await self.event_queue.put({
            'type': 'topology_event',
            'event': topology_event,
            'original_event': event
        })
        
        return {
            'success': True,
            'result': result,
            'topology_event': topology_event
        }
    
    async def _handle_session_closed(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """处理会话关闭事件"""
        session_key = event.get('session_key')
        reason = event.get('reason', 'unknown')
        
        if not session_key:
            return {
                'success': False,
                'error': 'session_key is required for session_closed event'
            }
        
        # 通过兼容层关闭会话
        result = self.compatibility_layer.close_session(session_key, reason)
        
        # 创建拓扑记忆事件
        topology_event = {
            'type': 'session_closed',
            'session_id': result.get('session_id'),
            'session_key': session_key,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        
        # 将事件放入队列
        await self.event_queue.put({
            'type': 'topology_event',
            'event': topology_event,
            'original_event': event
        })
        
        return {
            'success': True,
            'result': result,
            'topology_event': topology_event
        }
    
    async def _handle_tool_called(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """处理工具调用事件"""
        tool_call = event.get('tool_call')
        if not tool_call:
            return {
                'success': False,
                'error': 'tool_call is required for tool_called event'
            }
        
        # 创建消息格式的工具调用
        message = {
            'type': 'tool_call',
            'tool_name': tool_call.get('name'),
            'tool_args': tool_call.get('args', {}),
            'call_id': tool_call.get('id'),
            'sender': 'assistant',
            'session_key': event.get('session_key')
        }
        
        # 通过兼容层处理
        result = self.compatibility_layer.process_openclaw_message(message)
        
        # 创建拓扑记忆事件
        topology_event = {
            'type': 'tool_called',
            'context_id': result.get('context_id'),
            'tool_name': tool_call.get('name'),
            'call_id': tool_call.get('id'),
            'timestamp': datetime.now().isoformat()
        }
        
        # 将事件放入队列
        await self.event_queue.put({
            'type': 'topology_event',
            'event': topology_event,
            'original_event': event
        })
        
        return {
            'success': True,
            'result': result,
            'topology_event': topology_event
        }
    
    async def _handle_tool_completed(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """处理工具完成事件"""
        tool_result = event.get('tool_result')
        if not tool_result:
            return {
                'success': False,
                'error': 'tool_result is required for tool_completed event'
            }
        
        # 创建消息格式的工具结果
        message = {
            'type': 'tool_result',
            'call_id': tool_result.get('call_id'),
            'result': tool_result.get('result', {}),
            'success': tool_result.get('success', True),
            'error': tool_result.get('error'),
            'sender': 'system',
            'session_key': event.get('session_key')
        }
        
        # 通过兼容层处理
        result = self.compatibility_layer.process_openclaw_message(message)
        
        # 创建拓扑记忆事件
        topology_event = {
            'type': 'tool_completed',
            'context_id': result.get('context_id'),
            'call_id': tool_result.get('call_id'),
            'success': tool_result.get('success', True),
            'timestamp': datetime.now().isoformat()
        }
        
        # 将事件放入队列
        await self.event_queue.put({
            'type': 'topology_event',
            'event': topology_event,
            'original_event': event
        })
        
        return {
            'success': True,
            'result': result,
            'topology_event': topology_event
        }
    
    async def _handle_error_occurred(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """处理错误发生事件"""
        error = event.get('error', {})
        
        # 创建错误消息
        message = {
            'type': 'error',
            'error': error.get('message', 'Unknown error'),
            'code': error.get('code'),
            'details': error.get('details'),
            'sender': 'system',
            'session_key': event.get('session_key')
        }
        
        # 通过兼容层处理
        result = self.compatibility_layer.process_openclaw_message(message)
        
        # 创建拓扑记忆事件
        topology_event = {
            'type': 'error_occurred',
            'context_id': result.get('context_id'),
            'error_code': error.get('code'),
            'error_message': error.get('message'),
            'timestamp': datetime.now().isoformat()
        }
        
        # 将事件放入队列
        await self.event_queue.put({
            'type': 'topology_event',
            'event': topology_event,
            'original_event': event
        })
        
        return {
            'success': True,
            'result': result,
            'topology_event': topology_event
        }
    
    async def _handle_heartbeat(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """处理心跳事件"""
        # 创建心跳消息
        message = {
            'type': 'heartbeat',
            'timestamp': event.get('timestamp', datetime.now().isoformat()),
            'status': event.get('status', 'ok'),
            'sender': 'system',
            'session_key': event.get('session_key')
        }
        
        # 通过兼容层处理
        result = self.compatibility_layer.process_openclaw_message(message)
        
        # 创建拓扑记忆事件
        topology_event = {
            'type': 'heartbeat',
            'context_id': result.get('context_id'),
            'status': event.get('status', 'ok'),
            'timestamp': datetime.now().isoformat()
        }
        
        # 将事件放入队列
        await self.event_queue.put({
            'type': 'topology_event',
            'event': topology_event,
            'original_event': event
        })
        
        return {
            'success': True,
            'result': result,
            'topology_event': topology_event
        }
    
    async def _process_events(self):
        """处理事件队列"""
        logger.info("Event processing started")
        
        try:
            while True:
                # 从队列获取事件
                queue_item = await self.event_queue.get()
                
                try:
                    # 处理事件
                    await self._process_queue_item(queue_item)
                except Exception as e:
                    logger.error(f"Failed to process queue item: {e}")
                finally:
                    # 标记任务完成
                    self.event_queue.task_done()
                    
        except asyncio.CancelledError:
            logger.info("Event processing cancelled")
        except Exception as e:
            logger.error(f"Event processing error: {e}")
    
    async def _process_queue_item(self, queue_item: Dict[str, Any]):
        """处理队列项"""
        item_type = queue_item.get('type')
        
        if item_type == 'topology_event':
            # 处理拓扑记忆事件
            event = queue_item.get('event')
            original_event = queue_item.get('original_event')
            
            # 这里可以添加将事件发送到拓扑记忆系统的逻辑
            logger.debug(f"Processing topology event: {event.get('type')}")
            
            # 示例：记录事件
            self._log_event(event, original_event)
    
    def _log_event(self, event: Dict[str, Any], original_event: Dict[str, Any]):
        """记录事件"""
        log_entry = {
            'topology_event': event,
            'original_event': original_event,
            'logged_at': datetime.now().isoformat()
        }
        
        # 这里可以添加将日志保存到文件或数据库的逻辑
        logger.debug(f"Event logged: {event.get('type')}")
    
    async def subscribe(self, callback: Callable[[str, Dict, Dict], None]):
        """
        订阅事件
        
        Args:
            callback: 回调函数，接收(event_type, event, result)参数
        """
        self.subscribers.append(callback)
        logger.info(f"New subscriber added, total: {len(self.subscribers)}")
    
    async def unsubscribe(self, callback: Callable[[str, Dict, Dict], None]):
        """取消订阅事件"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.info(f"Subscriber removed, remaining: {len(self.subscribers)}")
    
    async def _notify_subscribers(self, event_type: str, event: Dict[str, Any], 
                                result: Dict[str, Any]):
        """通知订阅者"""
        for callback in self.subscribers:
            try:
                await callback(event_type, event, result)
            except Exception as e:
                logger.error(f"Subscriber callback error: {e}")
    
    def get_queue_size(self) -> int:
        """获取队列大小"""
        return self.event_queue.qsize()
    
    def get_subscriber_count(self) -> int:
        """获取订阅者数量"""
        return len(self.subscribers)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'queue_size': self.get_queue_size(),
            'subscriber_count': self.get_subscriber_count(),
            'event_handlers': list(self.event_handlers.keys()),
            'is_running': self.processing_task is not None and not self.processing_task.done()
        }