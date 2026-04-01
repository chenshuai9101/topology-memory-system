"""
会话适配器 - 处理OpenClaw会话键到拓扑记忆会话ID的映射
"""

import uuid
import hashlib
from datetime import datetime
from typing import Dict, Optional, Any, List
import logging

logger = logging.getLogger(__name__)


class SessionAdapter:
    """会话适配器，管理OpenClaw会话键到拓扑记忆会话ID的映射"""
    
    def __init__(self, storage_backend=None):
        """
        初始化会话适配器
        
        Args:
            storage_backend: 存储后端，用于持久化会话映射
        """
        self.session_mapping: Dict[str, Dict] = {}  # mapping_key -> session_info
        self.reverse_mapping: Dict[str, str] = {}   # session_id -> mapping_key
        self.storage_backend = storage_backend
        
        # 从存储后端加载现有映射
        if storage_backend:
            self._load_mappings()
    
    def _load_mappings(self):
        """从存储后端加载会话映射"""
        try:
            if hasattr(self.storage_backend, 'load_session_mappings'):
                mappings = self.storage_backend.load_session_mappings()
                for mapping in mappings:
                    mapping_key = mapping.get('mapping_key')
                    if mapping_key:
                        self.session_mapping[mapping_key] = mapping
                        self.reverse_mapping[mapping['session_id']] = mapping_key
                logger.info(f"Loaded {len(mappings)} session mappings from storage")
        except Exception as e:
            logger.warning(f"Failed to load session mappings: {e}")
    
    def _save_mapping(self, mapping_key: str, mapping_info: Dict):
        """保存会话映射到存储后端"""
        if self.storage_backend and hasattr(self.storage_backend, 'save_session_mapping'):
            try:
                self.storage_backend.save_session_mapping(mapping_key, mapping_info)
            except Exception as e:
                logger.error(f"Failed to save session mapping: {e}")
    
    def parse_openclaw_session_key(self, session_key: str) -> Dict[str, str]:
        """
        解析OpenClaw会话键
        
        OpenClaw格式: channel:accountId:conversationId
        
        Args:
            session_key: OpenClaw会话键
            
        Returns:
            解析后的会话信息字典
        """
        parts = session_key.split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid OpenClaw session key format: {session_key}")
        
        channel, account_id, conversation_id = parts
        
        return {
            'channel': channel,
            'account_id': account_id,
            'conversation_id': conversation_id,
            'original_key': session_key
        }
    
    def generate_mapping_key(self, session_info: Dict[str, str]) -> str:
        """
        生成映射键
        
        Args:
            session_info: 会话信息字典
            
        Returns:
            映射键字符串
        """
        channel = session_info['channel']
        account_id = session_info['account_id']
        conversation_id = session_info['conversation_id']
        
        # 使用哈希确保键的唯一性和一致性
        hash_input = f"{channel}:{account_id}:{conversation_id}"
        hash_digest = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        
        return f"{channel}_{account_id}_{conversation_id}_{hash_digest}"
    
    def map_session_key(self, openclaw_session_key: str, create_if_missing: bool = True) -> Optional[str]:
        """
        将OpenClaw会话键映射到拓扑记忆会话ID
        
        Args:
            openclaw_session_key: OpenClaw会话键
            create_if_missing: 如果映射不存在是否创建
            
        Returns:
            拓扑记忆会话ID，如果映射不存在且create_if_missing=False则返回None
        """
        try:
            # 解析会话键
            session_info = self.parse_openclaw_session_key(openclaw_session_key)
            mapping_key = self.generate_mapping_key(session_info)
            
            # 检查是否已存在映射
            if mapping_key in self.session_mapping:
                mapping = self.session_mapping[mapping_key]
                return mapping['session_id']
            
            # 如果不存在且不创建，返回None
            if not create_if_missing:
                return None
            
            # 创建新的映射
            session_id = str(uuid.uuid4())
            mapping_info = {
                'session_id': session_id,
                'openclaw_key': openclaw_session_key,
                'mapping_key': mapping_key,
                'channel': session_info['channel'],
                'account_id': session_info['account_id'],
                'conversation_id': session_info['conversation_id'],
                'created_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat()
            }
            
            # 保存映射
            self.session_mapping[mapping_key] = mapping_info
            self.reverse_mapping[session_id] = mapping_key
            
            # 持久化到存储后端
            self._save_mapping(mapping_key, mapping_info)
            
            logger.info(f"Created new session mapping: {openclaw_session_key} -> {session_id}")
            return session_id
            
        except ValueError as e:
            logger.error(f"Failed to map session key {openclaw_session_key}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error mapping session key {openclaw_session_key}: {e}")
            raise
    
    def get_openclaw_session_info(self, topology_session_id: str) -> Optional[Dict[str, Any]]:
        """
        根据拓扑记忆会话ID获取OpenClaw会话信息
        
        Args:
            topology_session_id: 拓扑记忆会话ID
            
        Returns:
            OpenClaw会话信息字典，如果不存在则返回None
        """
        if topology_session_id not in self.reverse_mapping:
            return None
        
        mapping_key = self.reverse_mapping[topology_session_id]
        mapping = self.session_mapping.get(mapping_key)
        
        if not mapping:
            return None
        
        return {
            'session_key': mapping['openclaw_key'],
            'channel': mapping['channel'],
            'account_id': mapping['account_id'],
            'conversation_id': mapping['conversation_id'],
            'mapping_created_at': mapping['created_at'],
            'mapping_last_accessed': mapping.get('last_accessed')
        }
    
    def update_last_accessed(self, session_key_or_id: str):
        """
        更新会话的最后访问时间
        
        Args:
            session_key_or_id: OpenClaw会话键或拓扑记忆会话ID
        """
        mapping_key = None
        
        # 确定是OpenClaw会话键还是拓扑记忆会话ID
        if ':' in session_key_or_id:  # 可能是OpenClaw会话键
            try:
                session_info = self.parse_openclaw_session_key(session_key_or_id)
                mapping_key = self.generate_mapping_key(session_info)
            except ValueError:
                pass
        else:  # 可能是拓扑记忆会话ID
            mapping_key = self.reverse_mapping.get(session_key_or_id)
        
        # 更新最后访问时间
        if mapping_key and mapping_key in self.session_mapping:
            self.session_mapping[mapping_key]['last_accessed'] = datetime.now().isoformat()
            
            # 保存到存储后端
            if self.storage_backend:
                self._save_mapping(mapping_key, self.session_mapping[mapping_key])
    
    def get_all_mappings(self) -> List[Dict[str, Any]]:
        """
        获取所有会话映射
        
        Returns:
            所有会话映射的列表
        """
        return list(self.session_mapping.values())
    
    def cleanup_old_mappings(self, days_old: int = 30):
        """
        清理旧的会话映射
        
        Args:
            days_old: 清理多少天前的映射
        """
        cutoff_date = datetime.now().timestamp() - (days_old * 24 * 3600)
        mappings_to_remove = []
        
        for mapping_key, mapping in self.session_mapping.items():
            last_accessed_str = mapping.get('last_accessed', mapping['created_at'])
            try:
                last_accessed = datetime.fromisoformat(last_accessed_str.replace('Z', '+00:00'))
                if last_accessed.timestamp() < cutoff_date:
                    mappings_to_remove.append(mapping_key)
            except (ValueError, TypeError):
                # 如果无法解析时间，保留映射
                pass
        
        # 移除旧的映射
        for mapping_key in mappings_to_remove:
            mapping = self.session_mapping.pop(mapping_key, None)
            if mapping:
                session_id = mapping['session_id']
                self.reverse_mapping.pop(session_id, None)
                logger.info(f"Removed old session mapping: {mapping_key}")
        
        return len(mappings_to_remove)


class SessionLifecycleManager:
    """会话生命周期管理器"""
    
    def __init__(self, session_adapter: SessionAdapter, topology_context_manager):
        """
        初始化会话生命周期管理器
        
        Args:
            session_adapter: 会话适配器实例
            topology_context_manager: 拓扑记忆上下文管理器
        """
        self.session_adapter = session_adapter
        self.context_manager = topology_context_manager
    
    def _extract_user_id(self, openclaw_session_key: str) -> str:
        """
        从OpenClaw会话键中提取用户ID
        
        Args:
            openclaw_session_key: OpenClaw会话键
            
        Returns:
            用户ID字符串
        """
        try:
            session_info = self.session_adapter.parse_openclaw_session_key(openclaw_session_key)
            # 使用account_id作为用户ID，或者根据需求调整
            return f"user_{session_info['account_id']}"
        except Exception:
            # 如果无法解析，使用默认用户ID
            return "user_unknown"
    
    def create_session(self, openclaw_session_key: str, user_id: Optional[str] = None, 
                      initial_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        创建新会话
        
        Args:
            openclaw_session_key: OpenClaw会话键
            user_id: 用户ID，如果为None则从会话键中提取
            initial_context: 初始上下文内容
            
        Returns:
            会话创建结果
        """
        try:
            # 映射会话键
            session_id = self.session_adapter.map_session_key(openclaw_session_key)
            
            # 确定用户ID
            if user_id is None:
                user_id = self._extract_user_id(openclaw_session_key)
            
            # 准备初始上下文
            if initial_context is None:
                session_info = self.session_adapter.parse_openclaw_session_key(openclaw_session_key)
                initial_context = {
                    'session_key': openclaw_session_key,
                    'action': 'session_create',
                    'created_at': datetime.now().isoformat(),
                    'source': 'openclaw',
                    'metadata': {
                        'channel': session_info['channel'],
                        'account_id': session_info['account_id'],
                        'conversation_id': session_info['conversation_id']
                    }
                }
            
            # 保存到拓扑记忆
            context_entry = self.context_manager.create_context(
                session_id=session_id,
                user_id=user_id,
                context_type='session_init',
                content=initial_context,
                priority=10,  # 高优先级
                ttl=86400  # 24小时
            )
            
            logger.info(f"Created new session: {openclaw_session_key} -> {session_id}")
            
            return {
                'success': True,
                'session_id': session_id,
                'openclaw_session_key': openclaw_session_key,
                'context_id': context_entry.id if hasattr(context_entry, 'id') else None,
                'user_id': user_id,
                'created_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create session for {openclaw_session_key}: {e}")
            return {
                'success': False,
                'error': str(e),
                'openclaw_session_key': openclaw_session_key
            }
    
    def close_session(self, openclaw_session_key: str, reason: str = 'user_ended') -> Dict[str, Any]:
        """
        关闭会话
        
        Args:
            openclaw_session_key: OpenClaw会话键
            reason: 关闭原因
            
        Returns:
            会话关闭结果
        """
        try:
            # 获取会话ID
            session_id = self.session_adapter.map_session_key(openclaw_session_key, create_if_missing=False)
            if not session_id:
                return {
                    'success': False,
                    'error': f"Session not found: {openclaw_session_key}",
                    'openclaw_session_key': openclaw_session_key
                }
            
            # 获取用户ID
            user_id = self._extract_user_id(openclaw_session_key)
            
            # 创建关闭上下文
            close_context = {
                'action': 'session_close',
                'closed_at': datetime.now().isoformat(),
                'reason': reason,
                'source': 'openclaw',
                'metadata': {
                    'close_reason': reason
                }
            }
            
            # 保存到拓扑记忆
            context_entry = self.context_manager.create_context(
                session_id=session_id,
                user_id=user_id,
                context_type='session_close',
                content=close_context,
                priority=5,
                ttl=3600  # 1小时
            )
            
            logger.info(f"Closed session: {openclaw_session_key} (reason: {reason})")
            
            return {
                'success': True,
                'session_id': session_id,
                'openclaw_session_key': openclaw_session_key,
                'context_id': context_entry.id if hasattr(context_entry, 'id') else None,
                'closed_at': datetime.now().isoformat(),
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Failed to close session {openclaw_session_key}: {e}")
            return {
                'success': False,
                'error': str(e),
                'openclaw_session_key': openclaw_session_key
            }
    
    def get_session_status(self, openclaw_session_key: str) -> Dict[str, Any]:
        """
        获取会话状态
        
        Args:
            openclaw_session_key: OpenClaw会话键
            
        Returns:
            会话状态信息
        """
        try:
            # 获取会话ID
            session_id = self.session_adapter.map_session_key(openclaw_session_key, create_if_missing=False)
            if not session_id:
                return {
                    'exists': False,
                    'openclaw_session_key': openclaw_session_key,
                    'message': 'Session not found'
                }
            
            # 获取会话信息
            session_info = self.session_adapter.get_openclaw_session_info(session_id)
            
            # 查询最近的上下文
            recent_contexts = self.context_manager.get_contexts_by_session(
                session_id=session_id,
                limit=5,
                order_by='created_at',
                order_direction='desc'
            )
            
            # 检查是否有关闭上下文
            is_closed = False
            close_reason = None
            for context in recent_contexts:
                if getattr(context, 'context_type', None) == 'session_close':
                    is_closed = True
                    close_reason = getattr(context, 'content', {}).get('reason')
                    break
            
            return {
                'exists': True,
                'is_active': not is_closed,
                'session_id': session_id,
                'openclaw_session_key': openclaw_session_key,
                'session_info': session_info,
                'close_reason': close_reason,
                'recent_context_count': len(recent_contexts),
                'last_updated': session_info.get('mapping_last_accessed') if session_info else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get session status for {openclaw_session_key}: {e}")
            return {
                'exists': False,
                'openclaw_session_key': openclaw_session_key,
                'error': str(e)
            }