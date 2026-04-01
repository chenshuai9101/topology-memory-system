"""
模型管理器
管理机器学习模型的加载、保存、版本控制和生命周期
"""

import os
import json
import pickle
import shutil
from typing import Dict, List, Any, Optional, Type, Union
from datetime import datetime, timedelta
import hashlib
from dataclasses import dataclass, asdict
import numpy as np

from ..config import ModelType, ModelConfig, ml_config
from ..models.predictor import ContextImportancePredictor
from ..models.classifier import MemoryNodeClassifier
from ..models.cluster import MemoryPatternCluster
from ..models.recommender import PersonalizedRecommender
from ..models.anomaly_detector import AnomalyDetector


@dataclass
class ModelMetadata:
    """模型元数据"""
    model_id: str
    model_type: ModelType
    model_name: str
    version: str
    created_at: str
    last_used: str
    performance_metrics: Dict[str, float]
    training_data_size: int
    feature_count: int
    file_size_bytes: int
    dependencies: List[str]
    is_active: bool = True
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """从字典创建"""
        return cls(**data)


class ModelManager:
    """模型管理器"""
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        初始化模型管理器
        
        Args:
            models_dir: 模型存储目录
        """
        self.models_dir = models_dir or ml_config.models_dir
        self.metadata_file = os.path.join(self.models_dir, "metadata.json")
        self.models: Dict[str, Any] = {}  # 内存中的模型缓存
        self.metadata: Dict[str, ModelMetadata] = {}
        
        # 确保目录存在
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 加载元数据
        self._load_metadata()
    
    def register_model(self, 
                      model: Any,
                      model_type: ModelType,
                      model_name: str,
                      version: str = "1.0.0",
                      performance_metrics: Optional[Dict[str, float]] = None,
                      training_data_size: int = 0,
                      feature_count: int = 0,
                      description: str = "",
                      tags: List[str] = None) -> str:
        """
        注册模型
        
        Args:
            model: 模型对象
            model_type: 模型类型
            model_name: 模型名称
            version: 版本号
            performance_metrics: 性能指标
            training_data_size: 训练数据大小
            feature_count: 特征数量
            description: 模型描述
            tags: 标签列表
            
        Returns:
            模型ID
        """
        # 生成模型ID
        model_id = self._generate_model_id(model_type, model_name, version)
        
        # 保存模型文件
        model_path = self._get_model_path(model_id)
        self._save_model_file(model, model_path)
        
        # 计算文件大小
        file_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
        
        # 创建元数据
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=model_type,
            model_name=model_name,
            version=version,
            created_at=datetime.now().isoformat(),
            last_used=datetime.now().isoformat(),
            performance_metrics=performance_metrics or {},
            training_data_size=training_data_size,
            feature_count=feature_count,
            file_size_bytes=file_size,
            dependencies=self._get_model_dependencies(model),
            is_active=True,
            description=description,
            tags=tags or []
        )
        
        # 保存元数据
        self.metadata[model_id] = metadata
        self._save_metadata()
        
        # 缓存模型
        self.models[model_id] = model
        
        print(f"模型已注册: {model_id}")
        return model_id
    
    def load_model(self, model_id: str, use_cache: bool = True) -> Any:
        """
        加载模型
        
        Args:
            model_id: 模型ID
            use_cache: 是否使用缓存
            
        Returns:
            加载的模型
        """
        # 检查缓存
        if use_cache and model_id in self.models:
            print(f"从缓存加载模型: {model_id}")
            self._update_last_used(model_id)
            return self.models[model_id]
        
        # 检查元数据
        if model_id not in self.metadata:
            raise ValueError(f"模型未注册: {model_id}")
        
        metadata = self.metadata[model_id]
        
        # 检查模型文件是否存在
        model_path = self._get_model_path(model_id)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 根据模型类型加载
        print(f"从文件加载模型: {model_id}")
        model = self._load_model_file(model_path, metadata.model_type)
        
        # 更新最后使用时间
        self._update_last_used(model_id)
        
        # 缓存模型
        if use_cache:
            self.models[model_id] = model
        
        return model
    
    def unload_model(self, model_id: str):
        """
        卸载模型（从缓存中移除）
        
        Args:
            model_id: 模型ID
        """
        if model_id in self.models:
            del self.models[model_id]
            print(f"模型已从缓存卸载: {model_id}")
    
    def delete_model(self, model_id: str):
        """
        删除模型
        
        Args:
            model_id: 模型ID
        """
        if model_id not in self.metadata:
            raise ValueError(f"模型未注册: {model_id}")
        
        # 删除模型文件
        model_path = self._get_model_path(model_id)
        if os.path.exists(model_path):
            os.remove(model_path)
        
        # 删除元数据文件
        metadata_path = f"{model_path}.metadata.json"
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        # 从缓存中移除
        if model_id in self.models:
            del self.models[model_id]
        
        # 从元数据中移除
        del self.metadata[model_id]
        self._save_metadata()
        
        print(f"模型已删除: {model_id}")
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model_id: 模型ID
            
        Returns:
            模型信息
        """
        if model_id not in self.metadata:
            raise ValueError(f"模型未注册: {model_id}")
        
        metadata = self.metadata[model_id]
        info = metadata.to_dict()
        
        # 添加额外信息
        model_path = self._get_model_path(model_id)
        info['file_exists'] = os.path.exists(model_path)
        info['in_cache'] = model_id in self.models
        info['age_days'] = (datetime.now() - datetime.fromisoformat(metadata.created_at)).days
        
        # 计算性能评分
        info['performance_score'] = self._calculate_performance_score(metadata.performance_metrics)
        
        return info
    
    def list_models(self, 
                   model_type: Optional[ModelType] = None,
                   active_only: bool = True,
                   sort_by: str = "last_used") -> List[Dict[str, Any]]:
        """
        列出模型
        
        Args:
            model_type: 过滤模型类型
            active_only: 只显示活跃模型
            sort_by: 排序字段 ('last_used', 'created_at', 'name', 'performance')
            
        Returns:
            模型列表
        """
        models_list = []
        
        for model_id, metadata in self.metadata.items():
            # 过滤
            if active_only and not metadata.is_active:
                continue
            
            if model_type and metadata.model_type != model_type:
                continue
            
            # 获取信息
            info = self.get_model_info(model_id)
            models_list.append(info)
        
        # 排序
        if sort_by == "last_used":
            models_list.sort(key=lambda x: x['last_used'], reverse=True)
        elif sort_by == "created_at":
            models_list.sort(key=lambda x: x['created_at'], reverse=True)
        elif sort_by == "name":
            models_list.sort(key=lambda x: x['model_name'])
        elif sort_by == "performance":
            models_list.sort(key=lambda x: x['performance_score'], reverse=True)
        
        return models_list
    
    def update_model_metadata(self, 
                             model_id: str,
                             updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新模型元数据
        
        Args:
            model_id: 模型ID
            updates: 更新字段
            
        Returns:
            更新后的元数据
        """
        if model_id not in self.metadata:
            raise ValueError(f"模型未注册: {model_id}")
        
        metadata = self.metadata[model_id]
        
        # 更新字段
        for key, value in updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        
        # 更新最后使用时间
        metadata.last_used = datetime.now().isoformat()
        
        # 保存
        self._save_metadata()
        
        return metadata.to_dict()
    
    def update_model_performance(self, 
                                model_id: str,
                                new_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        更新模型性能指标
        
        Args:
            model_id: 模型ID
            new_metrics: 新的性能指标
            
        Returns:
            更新后的元数据
        """
        if model_id not in self.metadata:
            raise ValueError(f"模型未注册: {model_id}")
        
        metadata = self.metadata[model_id]
        
        # 合并性能指标
        metadata.performance_metrics.update(new_metrics)
        
        # 保存
        self._save_metadata()
        
        return metadata.to_dict()
    
    def find_best_model(self, 
                       model_type: ModelType,
                       metric: str = "accuracy") -> Optional[str]:
        """
        查找最佳模型
        
        Args:
            model_type: 模型类型
            metric: 评估指标
            
        Returns:
            最佳模型的ID，如果没有则返回None
        """
        best_model_id = None
        best_score = -float('inf')
        
        for model_id, metadata in self.metadata.items():
            if metadata.model_type != model_type or not metadata.is_active:
                continue
            
            score = metadata.performance_metrics.get(metric, -1)
            if score > best_score:
                best_score = score
                best_model_id = model_id
        
        return best_model_id
    
    def cleanup_old_models(self, 
                          max_age_days: int = 30,
                          min_performance: float = 0.7) -> List[str]:
        """
        清理旧模型
        
        Args:
            max_age_days: 最大保留天数
            min_performance: 最低性能要求
            
        Returns:
            被删除的模型ID列表
        """
        deleted_models = []
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        for model_id, metadata in list(self.metadata.items()):
            created_at = datetime.fromisoformat(metadata.created_at)
            
            # 检查是否应该删除
            should_delete = False
            
            # 检查年龄
            if created_at < cutoff_date:
                should_delete = True
            
            # 检查性能
            performance_score = self._calculate_performance_score(metadata.performance_metrics)
            if performance_score < min_performance:
                should_delete = True
            
            # 删除模型
            if should_delete:
                try:
                    self.delete_model(model_id)
                    deleted_models.append(model_id)
                except Exception as e:
                    print(f"删除模型 {model_id} 失败: {e}")
        
        print(f"清理了 {len(deleted_models)} 个旧模型")
        return deleted_models
    
    def export_model(self, model_id: str, export_dir: str) -> str:
        """
        导出模型
        
        Args:
            model_id: 模型ID
            export_dir: 导出目录
            
        Returns:
            导出文件路径
        """
        if model_id not in self.metadata:
            raise ValueError(f"模型未注册: {model_id}")
        
        # 确保导出目录存在
        os.makedirs(export_dir, exist_ok=True)
        
        # 源文件路径
        source_path = self._get_model_path(model_id)
        source_metadata_path = f"{source_path}.metadata.json"
        
        # 目标文件路径
        export_path = os.path.join(export_dir, f"{model_id}.model")
        export_metadata_path = f"{export_path}.metadata.json"
        
        # 复制文件
        shutil.copy2(source_path, export_path)
        if os.path.exists(source_metadata_path):
            shutil.copy2(source_metadata_path, export_metadata_path)
        
        print(f"模型已导出到: {export_path}")
        return export_path
    
    def import_model(self, import_path: str) -> str:
        """
        导入模型
        
        Args:
            import_path: 导入文件路径
            
        Returns:
            导入后的模型ID
        """
        if not os.path.exists(import_path):
            raise FileNotFoundError(f"导入文件不存在: {import_path}")
        
        # 加载模型以获取信息
        try:
            # 尝试从文件推断模型类型
            with open(import_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # 检查模型类型
            if 'config' in model_data:
                config = model_data['config']
                model_type = config.model_type
                model_name = config.model_name
                version = config.version
            else:
                # 无法推断，使用默认值
                model_type = ModelType.PREDICTOR
                model_name = "imported_model"
                version = "1.0.0"
            
            # 加载模型
            model = self._load_model_file(import_path, model_type)
            
            # 注册模型
            model_id = self.register_model(
                model=model,
                model_type=model_type,
                model_name=model_name,
                version=version,
                description="导入的模型"
            )
            
            return model_id
            
        except Exception as e:
            raise ValueError(f"导入模型失败: {e}")
    
    def _generate_model_id(self, model_type: ModelType, model_name: str, version: str) -> str:
        """生成模型ID"""
        # 使用哈希确保唯一性
        unique_string = f"{model_type.value}_{model_name}_{version}_{datetime.now().timestamp()}"
        hash_obj = hashlib.md5(unique_string.encode())
        return hash_obj.hexdigest()[:12]
    
    def _get_model_path(self, model_id: str) -> str:
        """获取模型文件路径"""
        return os.path.join(self.models_dir, f"{model_id}.model")
    
    def _save_model_file(self, model, path: str):
        """保存模型文件"""
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    
    def _load_model_file(self, path: str, model_type: ModelType) -> Any:
        """加载模型文件"""
        # 根据模型类型使用相应的加载方法
        if model_type == ModelType.PREDICTOR:
            return ContextImportancePredictor.load(path)
        elif model_type == ModelType.CLASSIFIER:
            return MemoryNodeClassifier.load(path)
        elif model_type == ModelType.CLUSTER:
            return MemoryPatternCluster.load(path)
        elif model_type == ModelType.RECOMMENDER:
            return PersonalizedRecommender.load(path)
        elif model_type == ModelType.ANOMALY:
            return AnomalyDetector.load(path)
        else:
            # 通用加载
            with open(path, 'rb') as f:
                return pickle.load(f)
    
    def _get_model_dependencies(self, model) -> List[str]:
        """获取模型依赖"""
        dependencies = []
        
        # 检查模型类型
        model_class = model.__class__.__name__
        dependencies.append(f"model_class:{model_class}")
        
        # 添加scikit-learn依赖
        if hasattr(model, '_estimator_type'):
            dependencies.append(f"sklearn:{model._estimator_type}")
        
        # 添加其他可能依赖
        if hasattr(model, 'n_estimators'):
            dependencies.append(f"ensemble:{model.n_estimators}")
        
        return dependencies
    
    def _load_metadata(self):
        """加载元数据"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                
                # 转换回ModelMetadata对象
                for model_id, metadata_dict in data.items():
                    self.metadata[model_id] = ModelMetadata.from_dict(metadata_dict)
                
                print(f"加载了 {len(self.metadata)} 个模型的元数据")
            except Exception as e:
                print(f"加载元数据失败: {e}")
                self.metadata = {}
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """保存元数据"""
        # 转换为字典
        data = {model_id: metadata.to_dict() 
                for model_id, metadata in self.metadata.items()}
        
        # 保存到文件
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _update_last_used(self, model_id: str):
        """更新最后使用时间"""
        if model_id in self.metadata:
            self.metadata[model_id].last_used = datetime.now().isoformat()
            self._save_metadata()
    
    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """计算性能评分"""
        if not metrics:
            return 0.0
        
        # 根据不同指标计算加权评分        # 根据不同指标计算加权评分
        score_weights = {
            'accuracy': 0.3,
            'precision': 0.2,
            'recall': 0.2,
            'f1_score': 0.3,
            'r2_score': 0.4,
            'mse': -0.2,  # 负权重，越小越好
            'rmse': -0.2,
            'silhouette_score': 0.5,
            'precision@10': 0.3,
            'recall@10': 0.2,
            'inference_time_ms_mean': -0.1  # 推理时间越小越好
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in score_weights.items():
            if metric in metrics:
                value = metrics[metric]
                
                # 处理负权重指标
                if weight < 0:
                    # 对于误差指标，值越小越好
                    if metric in ['mse', 'rmse', 'inference_time_ms_mean']:
                        # 归一化到0-1范围（假设最大合理值）
                        max_value = 100.0 if 'time' in metric else 10.0
                        normalized_value = max(0.0, 1.0 - (value / max_value))
                        total_score += normalized_value * abs(weight)
                        total_weight += abs(weight)
                else:
                    # 正权重指标，值越大越好
                    total_score += value * weight
                    total_weight += weight
        
        # 计算平均分
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.0
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        stats = {
            'total_models': len(self.metadata),
            'active_models': sum(1 for m in self.metadata.values() if m.is_active),
            'models_by_type': {},
            'total_file_size_bytes': 0,
            'avg_performance_score': 0.0,
            'recent_models': 0
        }
        
        # 按类型统计
        for metadata in self.metadata.values():
            model_type = metadata.model_type.value
            stats['models_by_type'][model_type] = stats['models_by_type'].get(model_type, 0) + 1
            
            # 文件大小
            stats['total_file_size_bytes'] += metadata.file_size_bytes
            
            # 性能评分
            performance_score = self._calculate_performance_score(metadata.performance_metrics)
            stats['avg_performance_score'] += performance_score
            
            # 最近模型（7天内）
            last_used = datetime.fromisoformat(metadata.last_used)
            if (datetime.now() - last_used).days <= 7:
                stats['recent_models'] += 1
        
        # 计算平均值
        if stats['total_models'] > 0:
            stats['avg_performance_score'] /= stats['total_models']
            stats['avg_file_size_bytes'] = stats['total_file_size_bytes'] / stats['total_models']
        
        # 添加时间统计
        if self.metadata:
            creation_dates = [datetime.fromisoformat(m.created_at) for m in self.metadata.values()]
            stats['oldest_model_days'] = (datetime.now() - min(creation_dates)).days
            stats['newest_model_days'] = (datetime.now() - max(creation_dates)).days
        
        return stats
    
    def create_model_backup(self, backup_dir: str) -> str:
        """
        创建模型备份
        
        Args:
            backup_dir: 备份目录
            
        Returns:
            备份文件路径
        """
        # 确保备份目录存在
        os.makedirs(backup_dir, exist_ok=True)
        
        # 创建备份文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(backup_dir, f"model_backup_{timestamp}.zip")
        
        # 创建临时目录
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # 复制所有模型文件
            for model_id in self.metadata.keys():
                model_path = self._get_model_path(model_id)
                metadata_path = f"{model_path}.metadata.json"
                
                if os.path.exists(model_path):
                    shutil.copy2(model_path, os.path.join(temp_dir, f"{model_id}.model"))
                
                if os.path.exists(metadata_path):
                    shutil.copy2(metadata_path, os.path.join(temp_dir, f"{model_id}.metadata.json"))
            
            # 复制元数据文件
            shutil.copy2(self.metadata_file, os.path.join(temp_dir, "metadata.json"))
            
            # 创建zip文件
            import zipfile
            with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zipf.write(file_path, arcname)
        
        print(f"模型备份已创建: {backup_file}")
        return backup_file
    
    def restore_from_backup(self, backup_file: str):
        """
        从备份恢复模型
        
        Args:
            backup_file: 备份文件路径
        """
        if not os.path.exists(backup_file):
            raise FileNotFoundError(f"备份文件不存在: {backup_file}")
        
        print(f"从备份恢复模型: {backup_file}")
        
        # 创建临时目录
        import tempfile
        import zipfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 解压备份文件
            with zipfile.ZipFile(backup_file, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # 清空当前模型目录
            for filename in os.listdir(self.models_dir):
                file_path = os.path.join(self.models_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"删除文件失败 {file_path}: {e}")
            
            # 复制备份文件
            for filename in os.listdir(temp_dir):
                src_path = os.path.join(temp_dir, filename)
                dst_path = os.path.join(self.models_dir, filename)
                
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)
            
            # 重新加载元数据
            self._load_metadata()
            self.models.clear()
        
        print("模型恢复完成")
    
    def validate_model(self, model_id: str) -> Dict[str, Any]:
        """
        验证模型
        
        Args:
            model_id: 模型ID
            
        Returns:
            验证结果
        """
        if model_id not in self.metadata:
            return {'valid': False, 'error': '模型未注册'}
        
        metadata = self.metadata[model_id]
        result = {
            'valid': True,
            'model_id': model_id,
            'checks': []
        }
        
        # 检查1: 模型文件是否存在
        model_path = self._get_model_path(model_id)
        if os.path.exists(model_path):
            result['checks'].append({'check': 'file_exists', 'passed': True})
        else:
            result['checks'].append({'check': 'file_exists', 'passed': False, 'error': '模型文件不存在'})
            result['valid'] = False
        
        # 检查2: 文件大小是否合理
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            if file_size > 100:  # 最小合理大小
                result['checks'].append({'check': 'file_size', 'passed': True, 'size_bytes': file_size})
            else:
                result['checks'].append({'check': 'file_size', 'passed': False, 'error': '文件大小异常', 'size_bytes': file_size})
                result['valid'] = False
        
        # 检查3: 能否加载模型
        if result['valid']:
            try:
                model = self.load_model(model_id, use_cache=False)
                result['checks'].append({'check': 'loadable', 'passed': True})
                
                # 检查4: 模型是否有predict方法
                if hasattr(model, 'predict'):
                    result['checks'].append({'check': 'has_predict', 'passed': True})
                else:
                    result['checks'].append({'check': 'has_predict', 'passed': False, 'error': '模型缺少predict方法'})
                    result['valid'] = False
                
                # 卸载模型
                self.unload_model(model_id)
                
            except Exception as e:
                result['checks'].append({'check': 'loadable', 'passed': False, 'error': str(e)})
                result['valid'] = False
        
        # 检查5: 性能指标是否合理
        if metadata.performance_metrics:
            performance_score = self._calculate_performance_score(metadata.performance_metrics)
            if performance_score >= 0.5:
                result['checks'].append({'check': 'performance', 'passed': True, 'score': performance_score})
            else:
                result['checks'].append({'check': 'performance', 'passed': False, 'error': '性能评分过低', 'score': performance_score})
                result['valid'] = False
        
        return result
    
    def get_model_usage_report(self, days: int = 30) -> Dict[str, Any]:
        """
        获取模型使用报告
        
        Args:
            days: 报告天数
            
        Returns:
            使用报告
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        report = {
            'period_days': days,
            'total_models': len(self.metadata),
            'models_used': 0,
            'usage_by_type': {},
            'top_models': [],
            'inactive_models': []
        }
        
        for model_id, metadata in self.metadata.items():
            last_used = datetime.fromisoformat(metadata.last_used)
            
            # 检查是否在报告期内使用过
            if last_used >= cutoff_date:
                report['models_used'] += 1
                
                # 按类型统计
                model_type = metadata.model_type.value
                report['usage_by_type'][model_type] = report['usage_by_type'].get(model_type, 0) + 1
            
            # 收集不活跃模型
            if not metadata.is_active:
                report['inactive_models'].append({
                    'model_id': model_id,
                    'model_name': metadata.model_name,
                    'last_used': metadata.last_used,
                    'reason': '手动标记为不活跃'
                })
        
        # 获取最常用的模型
        sorted_models = sorted(
            self.metadata.items(),
            key=lambda x: datetime.fromisoformat(x[1].last_used),
            reverse=True
        )
        
        for model_id, metadata in sorted_models[:10]:
            report['top_models'].append({
                'model_id': model_id,
                'model_name': metadata.model_name,
                'model_type': metadata.model_type.value,
                'last_used': metadata.last_used,
                'performance_score': self._calculate_performance_score(metadata.performance_metrics)
            })
        
        # 计算使用率
        if report['total_models'] > 0:
            report['usage_rate'] = report['models_used'] / report['total_models']
        else:
            report['usage_rate'] = 0.0
        
        return report