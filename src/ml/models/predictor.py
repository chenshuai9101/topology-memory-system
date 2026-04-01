"""
上下文重要性预测模型
基于历史数据预测记忆节点的上下文重要性
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

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from ..config import ModelType, ModelConfig, ml_config
from ..utils.feature_extractor import FeatureExtractor
from ..utils.metrics import MLMetrics


class ContextImportancePredictor:
    """上下文重要性预测模型"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        初始化预测模型
        
        Args:
            config: 模型配置
        """
        self.config = config or ml_config.default_predictor_config
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_metrics = {}
        self.feature_importance = {}
        self.last_trained = None
        
    def train(self, 
              X_data: List[Dict[str, Any]],
              y_data: List[float],
              validation_split: float = 0.2,
              optimize: bool = True) -> Dict[str, Any]:
        """
        训练预测模型
        
        Args:
            X_data: 训练数据（记忆节点和上下文数据）
            y_data: 目标值（重要性分数）
            validation_split: 验证集比例
            optimize: 是否进行超参数优化
            
        Returns:
            训练结果字典
        """
        print(f"开始训练上下文重要性预测模型，数据量: {len(X_data)}")
        
        # 提取特征
        X_processed = self.feature_extractor.prepare_training_data(
            X_data, y_data, feature_type='combined'
        )
        
        # 分割训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y_data, 
            test_size=validation_split,
            random_state=ml_config.random_state
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
        
        print(f"训练完成！验证集R²分数: {self.training_metrics.get('r2_score', 0):.4f}")
        
        return {
            'status': 'success',
            'metrics': self.training_metrics,
            'feature_importance': self.feature_importance,
            'model_info': self.get_model_info()
        }
    
    def predict(self, 
                node_data: Dict[str, Any],
                context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        预测上下文重要性
        
        Args:
            node_data: 记忆节点数据
            context_data: 上下文数据
            
        Returns:
            预测结果字典
        """
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用train()方法")
        
        # 提取特征
        features = self.feature_extractor.extract_combined_features(node_data, context_data)
        
        # 转换为DataFrame
        features_df = pd.DataFrame([features])
        
        # 编码分类特征
        features_encoded = self.feature_extractor._encode_categorical_features(features_df)
        
        # 标准化数值特征
        features_scaled = self.feature_extractor._scale_numerical_features(features_encoded)
        
        # 预测
        importance_score = float(self.model.predict(features_scaled)[0])
        
        # 计算置信度
        confidence = self._calculate_confidence(features_scaled)
        
        # 获取特征贡献度
        feature_contributions = self._get_feature_contributions(features_scaled)
        
        return {
            'importance_score': max(0.0, min(1.0, importance_score)),  # 限制在0-1之间
            'confidence': confidence,
            'feature_contributions': feature_contributions,
            'model_version': self.config.version,
            'prediction_time': datetime.now().isoformat()
        }
    
    def batch_predict(self, 
                     data_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        批量预测
        
        Args:
            data_pairs: (节点数据, 上下文数据)对列表
            
        Returns:
            预测结果列表
        """
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用train()方法")
        
        results = []
        
        for node_data, context_data in data_pairs:
            try:
                prediction = self.predict(node_data, context_data)
                results.append(prediction)
            except Exception as e:
                print(f"预测失败: {e}")
                results.append({
                    'importance_score': 0.5,
                    'confidence': 0.0,
                    'error': str(e),
                    'model_version': self.config.version
                })
        
        return results
    
    def evaluate(self, 
                X_test: List[Dict[str, Any]],
                y_test: List[float]) -> Dict[str, Any]:
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
        
        # 提取特征
        X_processed = self.feature_extractor.prepare_training_data(
            X_test, feature_type='combined'
        )
        
        # 预测
        y_pred = self.model.predict(X_processed)
        
        # 计算指标
        metrics = MLMetrics.calculate_regression_metrics(np.array(y_test), y_pred)
        
        # 测量推理时间
        time_metrics = MLMetrics.measure_inference_time(self.model, X_processed)
        metrics.update(time_metrics)
        
        # 检查性能要求
        metrics['meets_requirements'] = (
            metrics.get('r2_score', 0) >= ml_config.min_accuracy and
            metrics.get('inference_time_ms_mean', 100) <= ml_config.max_inference_time_ms
        )
        
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
            'model_type': 'ContextImportancePredictor',
            'version': self.config.version,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
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
    def load(cls, path: str) -> 'ContextImportancePredictor':
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
        predictor = cls(config=model_data['config'])
        
        # 恢复状态
        predictor.model = model_data['model']
        predictor.feature_extractor = model_data['feature_extractor']
        predictor.training_metrics = model_data['training_metrics']
        predictor.feature_importance = model_data['feature_importance']
        predictor.is_trained = model_data['is_trained']
        predictor.last_trained = model_data['last_trained']
        
        return predictor
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_type': 'ContextImportancePredictor',
            'version': self.config.version,
            'is_trained': self.is_trained,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'training_metrics': self.training_metrics,
            'feature_importance_summary': self._get_feature_importance_summary(),
            'performance': {
                'meets_accuracy_requirement': self.training_metrics.get('r2_score', 0) >= ml_config.min_accuracy,
                'meets_speed_requirement': self.training_metrics.get('inference_time_ms_mean', 100) <= ml_config.max_inference_time_ms
            }
        }
    
    def _create_default_model(self):
        """创建默认模型"""
        model_type = self.config.hyperparameters.get('model_type', 'random_forest')
        
        if model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=self.config.hyperparameters.get('n_estimators', 100),
                max_depth=self.config.hyperparameters.get('max_depth', 10),
                random_state=self.config.hyperparameters.get('random_state', 42),
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=self.config.hyperparameters.get('n_estimators', 100),
                max_depth=self.config.hyperparameters.get('max_depth', 5),
                learning_rate=self.config.hyperparameters.get('learning_rate', 0.1),
                random_state=self.config.hyperparameters.get('random_state', 42)
            )
        elif model_type == 'linear':
            return Ridge(
                alpha=self.config.hyperparameters.get('alpha', 1.0),
                random_state=self.config.hyperparameters.get('random_state', 42)
            )
        elif model_type == 'svr':
            return SVR(
                C=self.config.hyperparameters.get('C', 1.0),
                kernel=self.config.hyperparameters.get('kernel', 'rbf')
            )
        elif model_type == 'neural_network':
            return MLPRegressor(
                hidden_layer_sizes=self.config.hyperparameters.get('hidden_layer_sizes', (100, 50)),
                activation=self.config.hyperparameters.get('activation', 'relu'),
                random_state=self.config.hyperparameters.get('random_state', 42),
                max_iter=self.config.hyperparameters.get('max_iter', 500)
            )
        else:
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
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
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
        
        model_type = self.config.hyperparameters.get('model_type', 'random_forest')
        
        if model_type in param_grids:
            # 创建基础模型
            if model_type == 'random_forest':
                base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            else:  # gradient_boosting
                base_model = GradientBoostingRegressor(random_state=42)
            
            # 网格搜索
            grid_search = GridSearchCV(
                base_model,
                param_grids[model_type],
                cv=ml_config.cv_folds,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"最佳参数: {grid_search.best_params_}")
            print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
        else:
            print(f"模型类型 {model_type} 未配置超参数优化，使用默认参数")
            return self._create_default_model()
    
    def _evaluate_model(self, X_val, y_val):
        """评估模型"""
        # 预测
        y_pred = self.model.predict(X_val)
        
        # 计算回归指标
        metrics = MLMetrics.calculate_regression_metrics(np.array(y_val), y_pred)
        
        # 测量推理时间
        time_metrics = MLMetrics.measure_inference_time(self.model, X_val)
        metrics.update(time_metrics)
        
        return metrics
    
    def _calculate_feature_importance(self, X_train):
        """计算特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = self.feature_extractor.get_feature_names('combined')
            
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
    
    def _calculate_confidence(self, features):
        """计算预测置信度"""
        # 简单实现：基于特征数量和模型类型
        if hasattr(self.model, 'predict_proba'):
            # 如果有概率预测，使用概率方差
            try:
                proba = self.model.predict_proba(features)
                confidence = np.max(proba, axis=1)[0]
                return float(confidence)
            except:
                pass
        
        # 默认置信度计算
        n_features = features.shape[1]
        base_confidence = min(0.9, 0.5 + (n_features / 100))
        
        # 如果有训练指标，基于R²调整
        if 'r2_score' in self.training_metrics:
            base_confidence *= self.training_metrics['r2_score']
        
        return float(base_confidence)
    
    def _get_feature_contributions(self, features):
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