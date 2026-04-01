"""
基础仓库类
提供通用的CRUD操作
"""

import logging
from typing import TypeVar, Generic, Type, Optional, List, Dict, Any
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, and_, or_

from ..config.database_manager import db_manager

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BaseRepository(Generic[T]):
    """基础仓库类"""
    
    def __init__(self, model_class: Type[T]):
        self.model_class = model_class
    
    def get_session(self) -> Session:
        """获取数据库会话"""
        return db_manager.session_factory()
    
    def get_by_id(self, id: UUID) -> Optional[T]:
        """根据ID获取记录"""
        with db_manager.get_session() as session:
            return session.query(self.model_class).filter(self.model_class.id == id).first()
    
    def get_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        """获取所有记录"""
        with db_manager.get_session() as session:
            return session.query(self.model_class).offset(skip).limit(limit).all()
    
    def create(self, **kwargs) -> T:
        """创建新记录"""
        with db_manager.get_session() as session:
            instance = self.model_class(**kwargs)
            session.add(instance)
            session.commit()
            session.refresh(instance)
            return instance
    
    def update(self, id: UUID, **kwargs) -> Optional[T]:
        """更新记录"""
        with db_manager.get_session() as session:
            instance = session.query(self.model_class).filter(self.model_class.id == id).first()
            if instance:
                for key, value in kwargs.items():
                    if hasattr(instance, key):
                        setattr(instance, key, value)
                session.commit()
                session.refresh(instance)
            return instance
    
    def delete(self, id: UUID) -> bool:
        """删除记录"""
        with db_manager.get_session() as session:
            instance = session.query(self.model_class).filter(self.model_class.id == id).first()
            if instance:
                session.delete(instance)
                session.commit()
                return True
            return False
    
    def count(self) -> int:
        """统计记录数量"""
        with db_manager.get_session() as session:
            return session.query(self.model_class).count()
    
    def filter_by(self, **kwargs) -> List[T]:
        """根据条件过滤记录"""
        with db_manager.get_session() as session:
            query = session.query(self.model_class)
            for key, value in kwargs.items():
                if hasattr(self.model_class, key):
                    query = query.filter(getattr(self.model_class, key) == value)
            return query.all()
    
    def filter_by_multi(self, filters: Dict[str, Any]) -> List[T]:
        """根据多个条件过滤记录"""
        with db_manager.get_session() as session:
            query = session.query(self.model_class)
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    if isinstance(value, list):
                        query = query.filter(getattr(self.model_class, key).in_(value))
                    else:
                        query = query.filter(getattr(self.model_class, key) == value)
            return query.all()
    
    def search(self, field: str, value: str, case_sensitive: bool = False) -> List[T]:
        """搜索记录"""
        with db_manager.get_session() as session:
            query = session.query(self.model_class)
            if hasattr(self.model_class, field):
                column = getattr(self.model_class, field)
                if case_sensitive:
                    query = query.filter(column.contains(value))
                else:
                    query = query.filter(column.ilike(f"%{value}%"))
            return query.all()
    
    def paginate(self, page: int = 1, page_size: int = 20, order_by: str = "created_at", 
                 descending: bool = True) -> Dict[str, Any]:
        """分页查询"""
        with db_manager.get_session() as session:
            # 计算总数
            total = session.query(self.model_class).count()
            
            # 计算总页数
            total_pages = (total + page_size - 1) // page_size
            
            # 计算偏移量
            offset = (page - 1) * page_size
            
            # 构建查询
            query = session.query(self.model_class)
            
            # 排序
            if hasattr(self.model_class, order_by):
                column = getattr(self.model_class, order_by)
                if descending:
                    query = query.order_by(desc(column))
                else:
                    query = query.order_by(asc(column))
            
            # 分页
            items = query.offset(offset).limit(page_size).all()
            
            return {
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages
            }
    
    def bulk_create(self, items: List[Dict[str, Any]]) -> List[T]:
        """批量创建记录"""
        with db_manager.get_session() as session:
            instances = []
            for item in items:
                instance = self.model_class(**item)
                session.add(instance)
                instances.append(instance)
            session.commit()
            
            # 刷新所有实例
            for instance in instances:
                session.refresh(instance)
            
            return instances
    
    def bulk_update(self, updates: List[Dict[str, Any]]) -> int:
        """批量更新记录"""
        updated_count = 0
        with db_manager.get_session() as session:
            for update in updates:
                id = update.pop('id', None)
                if id:
                    instance = session.query(self.model_class).filter(self.model_class.id == id).first()
                    if instance:
                        for key, value in update.items():
                            if hasattr(instance, key):
                                setattr(instance, key, value)
                        updated_count += 1
            session.commit()
        return updated_count
    
    def exists(self, **kwargs) -> bool:
        """检查记录是否存在"""
        with db_manager.get_session() as session:
            query = session.query(self.model_class)
            for key, value in kwargs.items():
                if hasattr(self.model_class, key):
                    query = query.filter(getattr(self.model_class, key) == value)
            return query.first() is not None
    
    def get_or_create(self, defaults: Dict[str, Any], **kwargs) -> tuple[T, bool]:
        """获取或创建记录"""
        with db_manager.get_session() as session:
            instance = session.query(self.model_class).filter_by(**kwargs).first()
            if instance:
                return instance, False
            
            # 合并参数
            create_kwargs = {**kwargs, **defaults}
            instance = self.model_class(**create_kwargs)
            session.add(instance)
            session.commit()
            session.refresh(instance)
            return instance, True