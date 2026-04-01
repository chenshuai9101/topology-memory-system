"""
记忆节点分类模型
自动分类记忆节点并生成标签
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import pickle
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from ..config import ModelType, ModelConfig, ml_config
from ..utils.feature_extractor import FeatureExtractor
from ..utils.metrics import MLMetrics


class MemoryNodeClassifier:
    """记忆节点分类模型"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        初始化分类模型
        
        Args:
            config: 模型配置
        """
        self.config = config or ml_config.default_classifier_config
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.label_encoder = LabelEncoder()
        self.class_labels = []
        self.is_trained = False
        self.training_metrics = {}
        self.feature_importance = {}
        self.last_trained = None
        
    def train(self, 
              X_data: List[Dict[str, Any]],
              y_data: List[str],
              validation_split: float = 0.2,
              optimize: bool = True) -> Dict[str, Any]:
        """
        训练分类模型
        
        Args:
            X_data: 训练数据（记忆节点数据）
            y_data: 目标标签
            validation_split: 验证集比例
            optimize: 是否进行超参数优化
            
        Returns:
            训练结果字典
        """
        print(f"开始训练记忆节点分类模型，数据量: {len(X_data)}，类别数: {len(set(y_data))}")
        
        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y_data)
        self.class_labels = self.label_encoder.classes_.tolist()
        
        # 提取特征
        X_processed = self.feature_extractor.prepare_training_data(
            X_data, y_encoded, feature_type='node'
        )
        
        # 分割训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y_encoded, 
            test_size=validation_split,
            random_state=ml_config.random_state,
            stratify=y_encoded
        )
        
        # 选择模型
        if optimize:
            self.model = self._optimize_model(X_train, y_train)
        else:
            self.model = self._create_default_model()
        
        # 训练模型
        print("训练模型中...")
        self.model.fit(X_train, y_train)
        
        # 评估模型
        print("评估模型性能...")
        self.training_metrics = self._evaluate_model(X_val, y_val)
        
        # 计算特征重要性
        self._calculate_feature_importance(X_train)
        
        # 更新状态
        self.is_trained = True
        self.last_trained = datetime.now()
        
        # 保存模型
        self.save()
        
        print(f"训练完成！验证集准确率: {self.training_metrics.get('accuracy', 0):.4f}")
        
        return {
            'status': 'success',
            'metrics': self.training_metrics,
            'feature_importance': self.feature_importance,
            'class_labels': self.class_labels,
            'model_info': self.get_model_info()
        }
    
    def predict(self, 
                node_data: Dict[str, Any],
                include_probabilities: bool = False) -> Dict[str, Any]:
        """
        预测记忆节点类别
        
        Args:
            node_data: 记忆节点数据
            include_probabilities: 是否包含类别概率
            
        Returns:
            预测结果字典
        """
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用train()方法")
        
        # 提取特征
        features = self.feature_extractor.extract_from_memory_node(node_data)
        
        # 转换为DataFrame
        features_df = pd.DataFrame([features])
        
        # 编码分类特征
        features_encoded = self.feature_extractor._encode_categorical_features(features_df)
        
        # 标准化数值特征
        features_scaled = self.feature_extractor._scale_numerical_features(features_encoded)
        
        # 预测
        predicted_class_idx = int(self.model.predict(features_scaled)[0])
        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        result = {
            'predicted_class': predicted_class,
            'class_index': predicted_class_idx,
            'confidence': 0.0,
            'model_version': self.config.version,
            'prediction_time': datetime.now().isoformat()
        }
        
        # 添加概率信息
        if include_probabilities and hasattr(self.model, 'predict_proba'):
            try:
                probabilities = self.model.predict_proba(features_scaled)[0]
                result['confidence'] = float(probabilities[predicted_class_idx])
                result['probabilities'] = {
                    self.class_labels[i]: float(prob) 
                    for i, prob in enumerate(probabilities)
                }
                
                # 添加top-3预测
                top_indices = np.argsort(probabilities)[-3:][::-1]
                result['top_predictions'] = [
                    {
                        'class': self.class_labels[idx],
                        'probability': float(probabilities[idx]),
                        'index': int(idx)
                    }
                    for idx in top_indices
                ]
            except Exception as e:
                print(f"获取概率失败: {e}")
        
        # 添加特征贡献度
        result['feature_contributions'] = self._get_feature_contributions(features_scaled, predicted_class_idx)
        
        return result
    
    def batch_predict(self, 
                     node_data_list: List[Dict[str, Any]],
                     include_probabilities: bool = False) -> List[Dict[str, Any]]:
        """
        批量预测
        
        Args:
            node_data_list: 记忆节点数据列表
            include_probabilities: 是否包含类别概率
            
        Returns:
            预测结果列表
        """
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用train()方法")
        
        results = []
        
        for node_data in node_data_list:
            try:
                prediction = self.predict(node_data, include_probabilities)
                results.append(prediction)
            except Exception as e:
                print(f"预测失败: {e}")
                results.append({
                    'predicted_class': 'unknown',
                    'confidence': 0.0,
                    'error': str(e),
                    'model_version': self.config.version
                })
        
        return results
    
    def suggest_labels(self, 
                      node_data: Dict[str, Any],
                      top_k: int = 3) -> List[Dict[str, Any]]:
        """
        为记忆节点建议标签
        
        Args:
            node_data: 记忆节点数据
            top_k: 返回的标签数量
            
        Returns:
            标签建议列表
        """
        prediction = self.predict(node_data, include_probabilities=True)
        
        if 'top_predictions' in prediction:
            return prediction['top_predictions'][:top_k]
        elif 'probabilities' in prediction:
            # 从概率中提取top-k
            probs = prediction['probabilities']
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            
            suggestions = []
            for i, (class_name, prob) in enumerate(sorted_probs[:top_k]):
                suggestions.append({
                    'class': class_name,
                    'probability': prob,
                    'index': i
                })
            return suggestions
        else:
            # 返回默认建议
            return [{
                'class': prediction['predicted_class'],
                'probability': prediction.get('confidence', 0.5),
                'index': 0
            }]
    
    def evaluate(self, 
                X_test: List[Dict[str, Any]],
                y_test: List[str]) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            X_test: 测试数据
            y_test: 测试标签
            
        Returns:
            评估结果
        """
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        # 编码标签
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # 提取特征
        X_processed = self.feature_extractor.prepare_training_data(
            X_test, feature_type='node'
        )
        
        # 预测
        y_pred = self.model.predict(X_processed)
        
        # 获取概率（如果可用）
        y_prob = None
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(X_processed)
        
        # 计算指标
        metrics = MLMetrics.calculate_classification_metrics(y_test_encoded, y_pred, y_prob)
        
        # 测量推理时间
        time_metrics = MLMetrics.measure_inference_time(self.model, X_processed)
        metrics.update(time_metrics)
        
        # 检查性能要求
        metrics['meets_requirements'] = (
            metrics.get('accuracy', 0) >= ml_config.min_accuracy and
            metrics.get('inference_time_ms_mean', 100) <= ml_config.max_inference_time_ms
        )
        
        # 添加类别详细信息
        metrics['class_labels'] = self.class_labels
        metrics['class_distribution'] = {
            label: int(np.sum(y_test_encoded == i))
            for i, label in enumerate(self.class_labels)
        }
        
        return metrics
    
    def save(self, path: Optional[str] = None):
        """
        保存模型
        
        Args:
            path: 保存路径（可选）
        """
        if not self.is_trained:
            print("模型未训练，跳过保存")
            return
        
        save_path = path or self.config.model_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 准备保存数据
        model_data = {
            'model': self.model,
            'config': self.config,
            'feature_extractor': self.feature_extractor,
            'label_encoder': self.label_encoder,
            'class_labels': self.class_labels,
            'training_metrics': self.training_metrics,
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained,
            'last_trained': self.last_trained,
            'version': self.config.version
        }
        
        # 保存模型
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # 保存元数据
        metadata_path = f"{save_path}.metadata.json"
        metadata = {
            'model_type': 'MemoryNodeClassifier',
            'version': self.config.version,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'class_labels': self.class_labels,
            'training_metrics': self.training_metrics,
            'feature_importance': self.feature_importance,
            'performance_requirements': {
                'min_accuracy': ml_config.min_accuracy,
                'max_inference_time_ms': ml_config.max_inference_time_ms
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"模型已保存到: {save_path}")
    
    @classmethod
    def load(cls, path: str) -> 'MemoryNodeClassifier':
        """
        加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            加载的模型实例
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # 创建实例
        classifier = cls(config=model_data['config'])
        
        # 恢复状态
        classifier.model = model_data['model']
        classifier.feature_extractor = model_data['feature_extractor']
        classifier.label_encoder = model_data['label_encoder']
        classifier.class_labels = model_data['class_labels']
        classifier.training_metrics = model_data['training_metrics']
        classifier.feature_importance = model_data['feature_importance']
        classifier.is_trained = model_data['is_trained']
        classifier.last_trained = model_data['last_trained']
        
        return classifier
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_type': 'MemoryNodeClassifier',
            'version': self.config.version,
            'is_trained': self.is_trained,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'class_labels': self.class_labels,
            'num_classes': len(self.class_labels),
            'training_metrics': self.training_metrics,
            'feature_importance_summary': self._get_feature_importance_summary(),
            'performance': {
                'meets_accuracy_requirement': self.training_metrics.get('accuracy', 0) >= ml_config.min_accuracy,
                'meets_speed_requirement': self.training_metrics.get('inference_time_ms_mean', 100) <= ml_config.max_inference_time_ms
            }
        }
    
    def _create_default_model(self):
        """创建默认模型"""
        model_type = self.config.hyperparameters.get('model_type', 'random_forest')
        
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=self.config.hyperparameters.get('n_estimators', 100),
                max_depth=self.config.hyperparameters.get('max_depth', 15),
                class_weight=self.config.hyperparameters.get('class_weight', 'balanced'),
                random_state=self.config.hyperparameters.get('random_state', 42),
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=self.config.hyperparameters.get('n_estimators', 100),
                max_depth=self.config.hyperparameters.get('max_depth', 5),
                learning_rate=self.config.hyperparameters.get('learning_rate', 0.1),
                random_state=self.config.hyperparameters.get('random_state', 42)
            )
        elif model_type == 'logistic_regression':
            return LogisticRegression(
                C=self.config.hyperparameters.get('C', 1.0),
                multi_class='ovr',
                random_state=self.config.hyperparameters.get('random_state', 42),
                max_iter=self.config.hyperparameters.get('max_iter', 1000)
            )
        elif model_type == 'svm':
            return SVC(
                C=self.config.hyperparameters.get('C', 1.0),
                kernel=self.config.hyperparameters.get('kernel', 'rbf'),
                probability=True,
                random_state=self.config.hyperparameters.get('random_state', 42)
            )
        elif model_type == 'neural_network':
            return MLPClassifier(
                hidden_layer_sizes=self.config.hyperparameters.get('hidden_layer_sizes', (100, 50)),
                activation=self.config.hyperparameters.get('activation', 'relu'),
                random_state=self.config.hyperparameters.get('random_state', 42),
                max_iter=self.config.hyperparameters.get('max_iter', 500)
            )
        else:
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
    
    def _optimize_model(self, X_train, y_train):
        """优化模型超参数"""
        print("进行超参数优化...")
        
        # 定义参数网格
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        model_type = self.config.hyperparameters.get('model_type', 'random_forest')
        
        if model_type in param_grids:
            # 创建基础模型
            if model_type == 'random_forest':
                base_model = RandomForestClassifier(
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
            else:  # logistic_regression
                base_model = LogisticRegression(
                    multi_class='ovr',
                    random_state=42,
                    max_iter=1000
                )
            
            # 网格搜索
            grid_search = GridSearchCV(
                base_model,
                param_grids[model_type],
                cv=ml_config.cv_folds,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"最佳参数: {grid_search.best_params_}")
            print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
        else:
            print(f"模型类型 {model_type} 未配置超参数优化，使用默认参数")
            return self._create_default_model()
    
    def _evaluate_model(self, X_val, y_val):
        """评估模型"""
        # 预测
        y_pred = self.model.predict(X_val)
        
        # 获取概率（如果可用）
        y_prob = None
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(X_val)
        
        # 计算分类指标
        metrics = MLMetrics.calculate_classification_metrics(y_val, y_pred, y_prob)
        
        # 测量推理时间
        time_metrics = MLMetrics.measure_inference_time(self.model, X_val)
        metrics.update(time_metrics)
        
        return metrics
    
    def _calculate_feature_importance(self, X_train):
        """计算特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = self.feature_extractor.get_feature_names('node')
            
            # 确保长度匹配
            min_len = min(len(importances), len(feature_names))
            importances = importances[:min_len]
            feature_names = feature_names[:min_len]
            
            # 创建重要性字典
            self.feature_importance = dict(zip(feature_names, importances))
            
            # 排序
            self.feature_importance = dict(sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
        else:
            self.feature_importance = {'not_available': 1.0}
    
    def _get_feature_contributions(self, features, predicted_class_idx):
        """获取特征贡献度"""
        if not self.feature_importance:
            return {}
        
        # 取前10个最重要的特征
        top_features = list(self.feature_importance.items())[:10]
        
        contributions = {}
        for feature_name, importance in top_features:
            if feature_name in features.columns:
                feature_value = features[feature_name].iloc[0]
                contributions[feature_name] = {
                    'importance': float(importance),
                    'value': float(feature_value),
                    'contribution': float(importance * feature_value)
                }
        
        return contributions
    
    def _get_feature_importance_summary(self):
        """获取特征重要性摘要"""
        if not self.feature_importance:
            return "未计算"
        
        top_features = list(self.feature_importance.items())[:5]
        summary = []
        
        for feature, importance in top_features:
            summary.append(f"{feature}: {importance:.4f}")
        
        return "; ".join(summary)