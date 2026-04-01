"""
机器学习配置
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum


class ModelType(Enum):
    """模型类型枚举"""
    PREDICTOR = "predictor"
    CLASSIFIER = "classifier"
    CLUSTER = "cluster"
    RECOMMENDER = "recommender"
    ANOMALY = "anomaly"


@dataclass
class ModelConfig:
    """模型配置"""
    model_type: ModelType
    model_name: str
    model_path: str
    version: str = "1.0.0"
    hyperparameters: Optional[Dict[str, Any]] = None
    feature_columns: Optional[list] = None
    target_column: Optional[str] = None
    metrics_thresholds: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.hyperparameters is None:
            self.hyperparameters = {}
        if self.metrics_thresholds is None:
            self.metrics_thresholds = {}


@dataclass
class MLConfig:
    """机器学习全局配置"""
    
    # 模型存储路径
    models_dir: str = os.path.join(os.path.dirname(__file__), "saved_models")
    
    # 训练配置
    train_test_split: float = 0.8
    random_state: int = 42
    cv_folds: int = 5
    
    # 性能要求
    min_accuracy: float = 0.85  # 85%准确率
    max_inference_time_ms: float = 50.0  # 50ms推理时间
    
    # 特征工程配置
    text_embedding_dim: int = 128
    numerical_features: list = None
    categorical_features: list = None
    text_features: list = None
    
    # 模型默认配置
    default_predictor_config: ModelConfig = None
    default_classifier_config: ModelConfig = None
    default_cluster_config: ModelConfig = None
    default_recommender_config: ModelConfig = None
    default_anomaly_config: ModelConfig = None
    
    def __post_init__(self):
        # 确保模型目录存在
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 初始化特征列表
        if self.numerical_features is None:
            self.numerical_features = ["access_count", "importance_score", "recency_days"]
        if self.categorical_features is None:
            self.categorical_features = ["node_type", "category"]
        if self.text_features is None:
            self.text_features = ["content", "tags"]
        
        # 初始化默认模型配置
        if self.default_predictor_config is None:
            self.default_predictor_config = ModelConfig(
                model_type=ModelType.PREDICTOR,
                model_name="context_importance_predictor",
                model_path=os.path.join(self.models_dir, "predictor"),
                hyperparameters={
                    "n_estimators": 100,
                    "max_depth": 10,
                    "learning_rate": 0.1,
                    "random_state": 42
                },
                metrics_thresholds={
                    "accuracy": 0.85,
                    "r2_score": 0.8,
                    "inference_time_ms": 50.0
                }
            )
        
        if self.default_classifier_config is None:
            self.default_classifier_config = ModelConfig(
                model_type=ModelType.CLASSIFIER,
                model_name="memory_node_classifier",
                model_path=os.path.join(self.models_dir, "classifier"),
                hyperparameters={
                    "n_estimators": 100,
                    "max_depth": 15,
                    "class_weight": "balanced",
                    "random_state": 42
                },
                metrics_thresholds={
                    "accuracy": 0.85,
                    "f1_score": 0.8,
                    "inference_time_ms": 50.0
                }
            )
        
        if self.default_cluster_config is None:
            self.default_cluster_config = ModelConfig(
                model_type=ModelType.CLUSTER,
                model_name="memory_pattern_cluster",
                model_path=os.path.join(self.models_dir, "cluster"),
                hyperparameters={
                    "n_clusters": 8,
                    "random_state": 42,
                    "n_init": 10
                },
                metrics_thresholds={
                    "silhouette_score": 0.5,
                    "inference_time_ms": 50.0
                }
            )
        
        if self.default_recommender_config is None:
            self.default_recommender_config = ModelConfig(
                model_type=ModelType.RECOMMENDER,
                model_name="personalized_recommender",
                model_path=os.path.join(self.models_dir, "recommender"),
                hyperparameters={
                    "factors": 50,
                    "regularization": 0.02,
                    "iterations": 20,
                    "random_state": 42
                },
                metrics_thresholds={
                    "precision_at_k": 0.7,
                    "recall_at_k": 0.6,
                    "inference_time_ms": 50.0
                }
            )
        
        if self.default_anomaly_config is None:
            self.default_anomaly_config = ModelConfig(
                model_type=ModelType.ANOMALY,
                model_name="anomaly_detector",
                model_path=os.path.join(self.models_dir, "anomaly"),
                hyperparameters={
                    "contamination": 0.1,
                    "random_state": 42,
                    "n_estimators": 100
                },
                metrics_thresholds={
                    "precision": 0.8,
                    "recall": 0.7,
                    "inference_time_ms": 50.0
                }
            )
    
    def get_model_config(self, model_type: ModelType) -> ModelConfig:
        """获取指定类型的模型配置"""
        config_map = {
            ModelType.PREDICTOR: self.default_predictor_config,
            ModelType.CLASSIFIER: self.default_classifier_config,
            ModelType.CLUSTER: self.default_cluster_config,
            ModelType.RECOMMENDER: self.default_recommender_config,
            ModelType.ANOMALY: self.default_anomaly_config,
        }
        return config_map.get(model_type)


# 全局配置实例
ml_config = MLConfig()