"""
数据库初始化脚本
"""

import logging
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from database.config.database_manager import db_manager
from database.services.database_service import db_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_database():
    """初始化数据库"""
    try:
        logger.info("开始初始化数据库...")
        
        # 初始化数据库连接
        db_manager.init_all()
        logger.info("数据库连接初始化成功")
        
        # 创建数据库表
        db_manager.create_tables()
        logger.info("数据库表创建成功")
        
        # 健康检查
        health = db_service.health_check()
        logger.info(f"数据库健康检查: {health}")
        
        # 获取系统统计
        stats = db_service.get_system_stats()
        logger.info(f"系统统计: {stats}")
        
        logger.info("数据库初始化完成")
        return True
        
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        return False


def drop_database():
    """删除数据库表"""
    try:
        logger.warning("开始删除数据库表...")
        
        # 初始化数据库连接
        db_manager.init_postgres()
        
        # 删除数据库表
        db_manager.drop_tables()
        logger.info("数据库表删除成功")
        
        return True
        
    except Exception as e:
        logger.error(f"删除数据库表失败: {e}")
        return False


def reset_database():
    """重置数据库"""
    try:
        logger.warning("开始重置数据库...")
        
        # 删除数据库表
        drop_database()
        
        # 重新初始化数据库
        init_database()
        
        logger.info("数据库重置完成")
        return True
        
    except Exception as e:
        logger.error(f"重置数据库失败: {e}")
        return False


def test_database():
    """测试数据库功能"""
    try:
        logger.info("开始测试数据库功能...")
        
        # 初始化数据库
        db_manager.init_all()
        
        # 测试创建上下文
        context_data = {
            "session_id": "test_session_001",
            "user_id": "test_user_001",
            "context_type": "conversation",
            "content": {"message": "Hello, world!"},
            "metadata": {"test": True},
            "priority": 5,
            "ttl": 3600
        }
        
        context = db_service.create_context(**context_data)
        logger.info(f"创建上下文成功: {context['id']}")
        
        # 测试获取上下文
        retrieved = db_service.get_context(context['id'])
        logger.info(f"获取上下文成功: {retrieved['id']}")
        
        # 测试创建记忆节点
        node_data = {
            "node_type": "concept",
            "content": "人工智能",
            "summary": "人工智能相关概念",
            "metadata": {"category": "technology"},
            "tags": ["AI", "technology"]
        }
        
        node = db_service.create_memory_node(**node_data)
        logger.info(f"创建记忆节点成功: {node['id']}")
        
        # 测试创建关联
        association = db_service.create_association(
            source_id=node['id'],
            target_id=node['id'],  # 自关联测试
            relation_type="related_to",
            weight=0.8,
            confidence=0.9,
            description="测试关联"
        )
        logger.info(f"创建关联成功: {association}")
        
        # 测试获取系统统计
        stats = db_service.get_system_stats()
        logger.info(f"系统统计: {stats}")
        
        # 测试性能指标
        metrics = db_service.get_performance_metrics()
        logger.info(f"性能指标: {metrics}")
        
        # 测试健康检查
        health = db_service.health_check()
        logger.info(f"健康检查: {health}")
        
        logger.info("数据库功能测试完成")
        return True
        
    except Exception as e:
        logger.error(f"数据库功能测试失败: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="数据库管理工具")
    parser.add_argument("command", choices=["init", "drop", "reset", "test"], 
                       help="执行命令: init(初始化), drop(删除), reset(重置), test(测试)")
    
    args = parser.parse_args()
    
    if args.command == "init":
        success = init_database()
        sys.exit(0 if success else 1)
    
    elif args.command == "drop":
        success = drop_database()
        sys.exit(0 if success else 1)
    
    elif args.command == "reset":
        success = reset_database()
        sys.exit(0 if success else 1)
    
    elif args.command == "test":
        success = test_database()
        sys.exit(0 if success else 1)