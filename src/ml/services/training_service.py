"""
训练服务
管理机器学习模型的训练流程
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import threading
import queue

from ..config import ModelType, ModelConfig, ml_config
from ..models.predictor import ContextImportancePredictor
from ..models.classifier import MemoryNodeClassifier
from ..models.cluster import MemoryPatternCluster
from ..models.recommender import PersonalizedRecommender
from ..models.anomaly_detector import AnomalyDetector
from .model_manager import ModelManager
from ..utils.feature_extractor import FeatureExtractor
from ..utils.metrics import MLMetrics


@dataclass
class TrainingJob:
    """训练任务"""
    job_id: str
    model_type: ModelType
    model_name: str
    status: str  # pending, running, completed, failed, cancelled
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: float = 0.0
    metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    data_size: int = 0
    model_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingJob':
        """从字典创建"""
        return cls(**data)


class TrainingService:
    """训练服务"""
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        """
        初始化训练服务
        
        Args:
            model_manager: 模型管理器
        """
        self.model_manager = model_manager or ModelManager()
        self.jobs: Dict[str, TrainingJob] = {}
        self.job_queue = queue.Queue()
        self.is_running = False
        self.worker_thread = None
        
        # 训练历史文件
        self.history_file = os.path.join(ml_config.models_dir, "training_history.json")
        self._load_history()
    
    def start(self):
        """启动训练服务"""
        if not self.is_running:
            self.is_running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            print("训练服务已启动")
    
    def stop(self):
        """停止训练服务"""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        print("训练服务已停止")
    
    def submit_training_job(self,
                           model_type: ModelType,
                           training_data: List[Dict[str, Any]],
                           labels: Optional[List] = None,
                           model_name: Optional[str] = None,
                           config: Optional[Dict[str, Any]] = None) -> str:
        """
        提交训练任务
        
        Args:
            model_type: 模型类型
            training_data: 训练数据
            labels: 标签数据（对于有监督学习）
            model_name: 模型名称
            config: 训练配置
            
        Returns:
            任务ID
        """
        # 生成任务ID
        job_id = self._generate_job_id(model_type)
        
        # 创建训练任务
        job = TrainingJob(
            job_id=job_id,
            model_type=model_type,
            model_name=model_name or f"{model_type.value}_model",
            status="pending",
            created_at=datetime.now().isoformat(),
            config=config or {},
            data_size=len(training_data)
        )
        
        # 保存任务
        self.jobs[job_id] = job
        
        # 将任务数据放入队列
        task_data = {
            'job_id': job_id,
            'model_type': model_type,
            'training_data': training_data,
            'labels': labels,
            'config': config
        }
        
        self.job_queue.put(task_data)
        
        print(f"训练任务已提交: {job_id} ({model_type.value})")
        return job_id
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        获取任务状态
        
        Args:
            job_id: 任务ID
            
        Returns:
            任务状态
        """
        if job_id not in self.jobs:
            raise ValueError(f"任务不存在: {job_id}")
        
        job = self.jobs[job_id]
        return job.to_dict()
    
    def cancel_job(self, job_id: str) -> bool:
        """
        取消任务
        
        Args:
            job_id: 任务ID
            
        Returns:
            是否成功取消
        """
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        if job.status in ["pending", "running"]:
            job.status = "cancelled"
            job.completed_at = datetime.now().isoformat()
            return True
        
        return False
    
    def list_jobs(self, 
                 status: Optional[str] = None,
                 model_type: Optional[ModelType] = None) -> List[Dict[str, Any]]:
        """
        列出任务
        
        Args:
            status: 过滤状态
            model_type: 过滤模型类型
            
        Returns:
            任务列表
        """
        jobs_list = []
        
        for job in self.jobs.values():
            # 过滤
            if status and job.status != status:
                continue
            
            if model_type and job.model_type != model_type:
                continue
            
            jobs_list.append(job.to_dict())
        
        # 按创建时间排序
        jobs_list.sort(key=lambda x: x['created_at'], reverse=True)
        
        return jobs_list
    
    def train_predictor(self,
                       training_data: List[Dict[str, Any]],
                       labels: List[float],
                       config: Optional[Dict[str, Any]] = None) -> Tuple[ContextImportancePredictor, Dict[str, Any]]:
        """
        训练预测模型
        
        Args:
            training_data: 训练数据
            labels: 重要性分数
            config: 训练配置
            
        Returns:
            (训练好的模型, 训练结果)
        """
        print(f"训练预测模型，数据量: {len(training_data)}")
        
        # 创建模型
        model_config = ml_config.default_predictor_config
        if config:
            # 更新配置
            for key, value in config.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
        
        predictor = ContextImportancePredictor(config=model_config)
        
        # 训练模型
        start_time = time.time()
        result = predictor.train(training_data, labels, optimize=True)
        training_time = time.time() - start_time
        
        # 添加训练时间
        result['training_time_seconds'] = training_time
        result['training_data_size'] = len(training_data)
        
        return predictor, result
    
    def train_classifier(self,
                        training_data: List[Dict[str, Any]],
                        labels: List[str],
                        config: Optional[Dict[str, Any]] = None) -> Tuple[MemoryNodeClassifier, Dict[str, Any]]:
        """
        训练分类模型
        
        Args:
            training_data: 训练数据
            labels: 类别标签
            config: 训练配置
            
        Returns:
            (训练好的模型, 训练结果)
        """
        print(f"训练分类模型，数据量: {len(training_data)}，类别数: {len(set(labels))}")
        
        # 创建模型
        model_config = ml_config.default_classifier_config
        if config:
            for key, value in config.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
        
        classifier = MemoryNodeClassifier(config=model_config)
        
        # 训练模型
        start_time = time.time()
        result = classifier.train(training_data, labels, optimize=True)
        training_time = time.time() - start_time
        
        result['training_time_seconds'] = training_time
        result['training_data_size'] = len(training_data)
        result['num_classes'] = len(set(labels))
        
        return classifier, result
    
    def train_cluster(self,
                     training_data: List[Dict[str, Any]],
                     config: Optional[Dict[str, Any]] = None) -> Tuple[MemoryPatternCluster, Dict[str, Any]]:
        """
        训练聚类模型
        
        Args:
            training_data: 训练数据
            config: 训练配置
            
        Returns:
            (训练好的模型, 训练结果)
        """
        print(f"训练聚类模型，数据量: {len(training_data)}")
        
        # 创建模型
        model_config = ml_config.default_cluster_config
        if config:
            for key, value in config.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
        
        cluster = MemoryPatternCluster(config=model_config)
        
        # 训练模型
        start_time = time.time()
        result = cluster.train(training_data, method='kmeans', reduce_dimensions=True)
        training_time = time.time() - start_time
        
        result['training_time_seconds'] = training_time
        result['training_data_size'] = len(training_data)
        
        return cluster, result
    
    def train_recommender(self,
                         user_interactions: List[Dict[str, Any]],
                         user_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
                         item_features: Optional[Dict[str, Dict[str, Any]]] = None,
                         config: Optional[Dict[str, Any]] = None) -> Tuple[PersonalizedRecommender, Dict[str, Any]]:
        """
        训练推荐系统
        
        Args:
            user_interactions: 用户交互数据
            user_profiles: 用户画像
            item_features: 物品特征
            config: 训练配置
            
        Returns:
            (训练好的模型, 训练结果)
        """
        print(f"训练推荐系统，交互数据量: {len(user_interactions)}")
        
        # 创建模型
        model_config = ml_config.default_recommender_config
        if config:
            for key, value in config.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
        
        recommender = PersonalizedRecommender(config=model_config)
        
        # 训练模型
        start_time = time.time()
        result = recommender.train(user_interactions, user_profiles, item_features, method='hybrid')
        training_time = time.time() - start_time
        
        result['training_time_seconds'] = training_time
        result['num_interactions'] = len(user_interactions)
        
        return recommender, result
    
    def train_anomaly_detector(self,
                              training_data: List[Dict[str, Any]],
                              labels: Optional[List[int]] = None,
                              config: Optional[Dict[str, Any]] = None) -> Tuple[AnomalyDetector, Dict[str, Any]]:
        """
        训练异常检测模型
        
        Args:
            training_data: 训练数据
            labels: 异常标签（可选）
            config: 训练配置
            
        Returns:
            (训练好的模型, 训练结果)
        """
        print(f"训练异常检测模型，数据量: {len(training_data)}")
        
        # 创建模型
        model_config = ml_config.default_anomaly_config
        if config:
            for key, value in config.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
        
        detector = AnomalyDetector(config=model_config)
        
        # 训练模型
        start_time = time.time()
        result = detector.train(training_data, labels, method='isolation_forest')
        training_time = time.time() - start_time
        
        result['training_time_seconds'] = training_time
        result['training_data_size'] = len(training_data)
        
        return detector, result
    
    def evaluate_model(self,
                      model: Any,
                      test_data: List[Dict[str, Any]],
                      test_labels: Optional[List] = None,
                      model_type: Optional[ModelType] = None) -> Dict[str, Any]:
        """
        评估模型
        
        Args:
            model: 要评估的模型
            test_data: 测试数据
            test_labels: 测试标签
            model_type: 模型类型
            
        Returns:
            评估结果
        """
        print(f"评估模型，测试数据量: {len(test_data)}")
        
        # 提取特征
        feature_extractor = FeatureExtractor()
        
        if model_type == ModelType.PREDICTOR:
            # 预测模型评估
            X_test = feature_extractor.prepare_training_data(test_data, feature_type='combined')
            y_test = test_labels if test_labels else []
            
            if len(y_test) > 0:
                return model.evaluate(test_data, y_test)
            else:
                return {"error": "预测模型评估需要标签数据"}
        
        elif model_type == ModelType.CLASSIFIER:
            # 分类模型评估
            X_test = feature_extractor.prepare_training_data(test_data, feature_type='node')
            y_test = test_labels if test_labels else []
            
            if len(y_test) > 0:
                return model.evaluate(test_data, y_test)
            else:
                return {"error": "分类模型评估需要标签数据"}
        
        elif model_type == ModelType.CLUSTER:
            # 聚类模型评估
            return model.evaluate(test_data)
        
        elif model_type == ModelType.RECOMMENDER:
            # 推荐系统评估
            if test_labels:
                # test_labels应该是交互数据
                return model.evaluate(test_labels)
            else:
                return {"error": "推荐系统评估需要交互数据"}
        
        elif model_type == ModelType.ANOMALY:
            # 异常检测评估
            X_test = feature_extractor.prepare_training_data(test_data, feature_type='combined')
            y_test = test_labels if test_labels else []
            
            if len(y_test) > 0:
                return model.evaluate(test_data, y_test)
            else:
                return {"error": "异常检测评估需要标签数据"}
        
        else:
            return {"error": f"未知的模型类型: {model_type}"}
    
    def _worker_loop(self):
        """工作线程循环"""
        while self.is_running:
            try:
                # 获取任务（非阻塞）
                try:
                    task_data = self.job_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                job_id = task_data['job_id']
                model_type = task_data['model_type']
                training_data = task_data['training_data']
                labels = task_data.get('labels')
                config = task_data.get('config', {})
                
                # 更新任务状态
                job = self.jobs[job_id]
                job.status = "running"
                job.started_at = datetime.now().isoformat()
                job.progress = 0.1
                
                print(f"开始执行训练任务: {job_id}")
                
                try:
                    # 执行训练
                    if model_type == ModelType.PREDICTOR:
                        if labels is None:
                            raise ValueError("预测模型训练需要标签数据")
                        model, result = self.train_predictor(training_data, labels, config)
                    
                    elif model_type == ModelType.CLASSIFIER:
                        if labels is None:
                            raise ValueError("分类模型训练需要标签数据")
                        model, result = self.train_classifier(training_data, labels, config)
                    
                    elif model_type == ModelType.CLUSTER:
                        model, result = self.train_cluster(training_data, config)
                    
                    elif model_type == ModelType.RECOMMENDER:
                        model, result = self.train_recommender(training_data, config=config)
                    
                    elif model_type == ModelType.ANOMALY:
                        model, result = self.train_anomaly_detector(training_data, labels, config)
                    
                    else:
                        raise ValueError(f"不支持的模型类型: {model_type}")
                    
                    # 更新进度
                    job.progress = 0.9
                    
                    # 注册模型
                    model_id = self.model_manager.register_model(
                        model=model,
                        model_type=model_type,
                        model_name=job.model_name,
                        version="1.0.0",
                        performance_metrics=result.get('metrics', {}),
                        training_data_size=len(training_data),
                        feature_count=self._estimate_feature_count(model_type, training_data),
                        description=f"自动训练于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    
                    # 更新任务结果
                    job.model_id = model_id
                    job.metrics = result.get('metrics', {})
                    job.status = "completed"
                    job.progress = 1.0
                    
                    print(f"训练任务完成: {job_id} -> 模型ID: {model_id}")
                    
                except Exception as e:
                    # 训练失败
                    job.status = "failed"
                    job.error_message = str(e)
                    print(f"训练任务失败: {job_id}, 错误: {e}")
                
                finally:
                    job.completed_at = datetime.now().isoformat()
                    self._save_history()
                    self.job_queue.task_done()
                    
            except Exception as e:
                print(f"训练服务工作线程错误: {e}")
                time.sleep(5)
    
    def _generate_job_id(self, model_type: ModelType) -> str:
        """生成任务ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = os.urandom(3).hex()
        return f"{model_type.value}_{timestamp}_{random_suffix}"
    
    def _estimate_feature_count(self, model_type: ModelType, data: List[Dict[str, Any]]) -> int:
        """估计特征数量"""
        if not data:
            return 0
        
        # 提取一个样本的特征
        feature_extractor = FeatureExtractor()
        
        if model_type in [ModelType.PREDICTOR, ModelType.ANOMALY]:
            # 需要节点和上下文
            if len(data) > 0 and isinstance(data[0], tuple) and len(data        if model_type in [ModelType.PREDICTOR, ModelType.ANOMALY]:
            # 需要节点和上下文
            if len(data) > 0 and isinstance(data[0], tuple) and len(data[0]) == 2:
                node_data, context_data = data[0]
                features = feature_extractor.extract_combined_features(node_data, context_data)
                return len(features)
        
        # 其他模型类型使用节点特征
        if len(data) > 0:
            features = feature_extractor.extract_from_memory_node(data[0])
            return len(features)
        
        return 0
    
    def _load_history(self):
        """加载训练历史"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    history_data = json.load(f)
                
                # 转换回TrainingJob对象
                for job_id, job_dict in history_data.items():
                    self.jobs[job_id] = TrainingJob.from_dict(job_dict)
                
                print(f"加载了 {len(self.jobs)} 个训练任务历史")
            except Exception as e:
                print(f"加载训练历史失败: {e}")
                self.jobs = {}
        else:
            self.jobs = {}
    
    def _save_history(self):
        """保存训练历史"""
        # 只保存最近100个任务
        all_jobs = list(self.jobs.values())
        all_jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        recent_jobs = all_jobs[:100]
        
        # 转换为字典
        history_data = {job.job_id: job.to_dict() for job in recent_jobs}
        
        # 保存到文件
        with open(self.history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        stats = {
            'total_jobs': len(self.jobs),
            'jobs_by_status': {},
            'jobs_by_type': {},
            'success_rate': 0.0,
            'avg_training_time_seconds': 0.0,
            'recent_activity': {}
        }
        
        completed_jobs = 0
        successful_jobs = 0
        total_training_time = 0.0
        
        # 按状态和类型统计
        for job in self.jobs.values():
            # 按状态统计
            stats['jobs_by_status'][job.status] = stats['jobs_by_status'].get(job.status, 0) + 1
            
            # 按类型统计
            model_type = job.model_type.value
            stats['jobs_by_type'][model_type] = stats['jobs_by_type'].get(model_type, 0) + 1
            
            # 计算成功率和训练时间
            if job.status == "completed":
                completed_jobs += 1
                successful_jobs += 1
                
                if job.metrics and 'training_time_seconds' in job.metrics:
                    total_training_time += job.metrics['training_time_seconds']
            
            elif job.status == "failed":
                completed_jobs += 1
        
        # 计算成功率
        if completed_jobs > 0:
            stats['success_rate'] = successful_jobs / completed_jobs
        
        # 计算平均训练时间
        if successful_jobs > 0:
            stats['avg_training_time_seconds'] = total_training_time / successful_jobs
        
        # 最近活动（最近7天）
        week_ago = datetime.now().timestamp() - (7 * 24 * 3600)
        
        for job in self.jobs.values():
            created_time = datetime.fromisoformat(job.created_at).timestamp()
            if created_time > week_ago:
                day = datetime.fromisoformat(job.created_at).strftime('%Y-%m-%d')
                stats['recent_activity'][day] = stats['recent_activity'].get(day, 0) + 1
        
        return stats
    
    def create_training_report(self, job_id: str) -> Dict[str, Any]:
        """
        创建训练报告
        
        Args:
            job_id: 任务ID
            
        Returns:
            训练报告
        """
        if job_id not in self.jobs:
            raise ValueError(f"任务不存在: {job_id}")
        
        job = self.jobs[job_id]
        
        report = {
            'job_id': job_id,
            'model_type': job.model_type.value,
            'model_name': job.model_name,
            'status': job.status,
            'timeline': {
                'created_at': job.created_at,
                'started_at': job.started_at,
                'completed_at': job.completed_at
            },
            'data_info': {
                'data_size': job.data_size,
                'config': job.config or {}
            }
        }
        
        # 添加训练结果
        if job.status == "completed":
            report['result'] = {
                'model_id': job.model_id,
                'metrics': job.metrics or {},
                'success': True
            }
            
            # 计算训练时长
            if job.started_at and job.completed_at:
                start_time = datetime.fromisoformat(job.started_at)
                end_time = datetime.fromisoformat(job.completed_at)
                report['training_duration_seconds'] = (end_time - start_time).total_seconds()
        
        elif job.status == "failed":
            report['result'] = {
                'success': False,
                'error_message': job.error_message
            }
        
        # 添加性能评估
        if job.metrics:
            report['performance_summary'] = self._create_performance_summary(job.metrics, job.model_type)
        
        return report
    
    def _create_performance_summary(self, metrics: Dict[str, float], model_type: ModelType) -> Dict[str, Any]:
        """创建性能摘要"""
        summary = {
            'overall_score': 0.0,
            'key_metrics': {},
            'meets_requirements': False
        }
        
        # 根据模型类型选择关键指标
        key_metrics_map = {
            ModelType.PREDICTOR: ['r2_score', 'mse', 'inference_time_ms_mean'],
            ModelType.CLASSIFIER: ['accuracy', 'f1_score', 'inference_time_ms_mean'],
            ModelType.CLUSTER: ['silhouette_score', 'inference_time_ms_mean'],
            ModelType.RECOMMENDER: ['precision@10', 'recall@10', 'inference_time_ms_mean'],
            ModelType.ANOMALY: ['accuracy', 'precision_anomaly', 'inference_time_ms_mean']
        }
        
        key_metrics = key_metrics_map.get(model_type, [])
        
        for metric in key_metrics:
            if metric in metrics:
                summary['key_metrics'][metric] = metrics[metric]
        
        # 计算总体分数
        if summary['key_metrics']:
            # 简单平均
            summary['overall_score'] = np.mean(list(summary['key_metrics'].values()))
        
        # 检查是否满足要求
        accuracy_metric = None
        if model_type == ModelType.PREDICTOR:
            accuracy_metric = 'r2_score'
        elif model_type in [ModelType.CLASSIFIER, ModelType.ANOMALY]:
            accuracy_metric = 'accuracy'
        elif model_type == ModelType.CLUSTER:
            accuracy_metric = 'silhouette_score'
        elif model_type == ModelType.RECOMMENDER:
            accuracy_metric = 'precision@10'
        
        if accuracy_metric and accuracy_metric in metrics:
            accuracy = metrics[accuracy_metric]
            inference_time = metrics.get('inference_time_ms_mean', 100)
            
            summary['meets_requirements'] = (
                accuracy >= ml_config.min_accuracy and
                inference_time <= ml_config.max_inference_time_ms
            )
        
        return summary
    
    def retrain_model(self, 
                     model_id: str,
                     new_data: List[Dict[str, Any]],
                     new_labels: Optional[List] = None,
                     incremental: bool = True) -> str:
        """
        重新训练模型
        
        Args:
            model_id: 模型ID
            new_data: 新数据
            new_labels: 新标签
            incremental: 是否增量训练
            
        Returns:
            新训练任务ID
        """
        # 获取原模型信息
        model_info = self.model_manager.get_model_info(model_id)
        model_type = ModelType(model_info['model_type'])
        
        if incremental:
            print(f"增量重新训练模型: {model_id}")
            # 这里可以实现增量训练逻辑
            # 简化版本：提交新的训练任务
            return self.submit_training_job(
                model_type=model_type,
                training_data=new_data,
                labels=new_labels,
                model_name=f"{model_info['model_name']}_v2",
                config=model_info.get('config', {})
            )
        else:
            print(f"完全重新训练模型: {model_id}")
            # 完全重新训练
            return self.submit_training_job(
                model_type=model_type,
                training_data=new_data,
                labels=new_labels,
                model_name=f"{model_info['model_name']}_retrained",
                config=model_info.get('config', {})
            )
    
    def optimize_hyperparameters(self,
                                model_type: ModelType,
                                training_data: List[Dict[str, Any]],
                                labels: Optional[List] = None,
                                param_grid: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """
        超参数优化
        
        Args:
            model_type: 模型类型
            training_data: 训练数据
            labels: 标签数据
            param_grid: 参数网格
            
        Returns:
            优化结果
        """
        print(f"开始超参数优化: {model_type.value}")
        
        # 这里可以实现更复杂的超参数优化
        # 简化版本：使用默认配置训练
        
        if model_type == ModelType.PREDICTOR:
            if labels is None:
                return {"error": "预测模型需要标签数据"}
            model, result = self.train_predictor(training_data, labels, {})
        
        elif model_type == ModelType.CLASSIFIER:
            if labels is None:
                return {"error": "分类模型需要标签数据"}
            model, result = self.train_classifier(training_data, labels, {})
        
        elif model_type == ModelType.CLUSTER:
            model, result = self.train_cluster(training_data, {})
        
        elif model_type == ModelType.RECOMMENDER:
            model, result = self.train_recommender(training_data, config={})
        
        elif model_type == ModelType.ANOMALY:
            model, result = self.train_anomaly_detector(training_data, labels, {})
        
        else:
            return {"error": f"不支持的模型类型: {model_type}"}
        
        return {
            "status": "completed",
            "best_model_metrics": result.get('metrics', {}),
            "optimization_method": "default_training"
        }