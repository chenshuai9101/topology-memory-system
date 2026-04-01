"""
异常检测模型
识别异常记忆模式和上下文
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

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ..config import ModelType, ModelConfig, ml_config
from ..utils.feature_extractor import FeatureExtractor
from ..utils.metrics import MLMetrics


class AnomalyDetector:
    """异常检测模型"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        初始化异常检测模型
        
        Args:
            config: 模型配置
        """
        self.config = config or ml_config.default_anomaly_config
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None
        self.is_trained = False
        self.training_metrics = {}
        self.threshold = 0.5
        self.normal_patterns = {}
        self.last_trained = None
        
    def train(self, 
              X_data: List[Dict[str, Any]],
              y_data: Optional[List[int]] = None,
              contamination: Optional[float] = None,
              method: str = 'isolation_forest') -> Dict[str, Any]:
        """
        训练异常检测模型
        
        Args:
            X_data: 训练数据（正常数据）
            y_data: 标签数据（可选，1=异常，0=正常）
            contamination: 异常比例估计
            method: 检测方法 ('isolation_forest', 'lof', 'svm', 'elliptic')
            
        Returns:
            训练结果字典
        """
        print(f"开始训练异常检测模型，数据量: {len(X_data)}")
        
        # 提取特征
        X_processed = self.feature_extractor.prepare_training_data(
            X_data, feature_type='combined'
        )
        
        # 如果有标签数据，使用监督学习
        if y_data is not None:
            print("使用监督学习训练异常检测模型")
            return self._train_supervised(X_processed, y_data, method)
        
        # 无监督学习
        print("使用无监督学习训练异常检测模型")
        
        # 设置异常比例
        if contamination is None:
            contamination = self.config.hyperparameters.get('contamination', 0.1)
        
        # 选择并训练模型
        self.model = self._create_model(method, contamination)
        
        print(f"使用 {method} 方法训练模型...")
        self.model.fit(X_processed)
        
        # 评估模型（如果没有标签，使用模型自身的预测）
        y_pred = self.model.predict(X_processed)
        y_scores = self.model.decision_function(X_processed) if hasattr(self.model, 'decision_function') else None
        
        # 转换标签：1=正常，-1=异常 -> 0=正常，1=异常
        y_pred_binary = np.where(y_pred == 1, 0, 1)
        
        # 如果没有真实标签，使用预测标签作为参考
        y_ref = y_pred_binary if y_data is None else y_data
        
        # 评估
        self.training_metrics = self._evaluate_model(X_processed, y_ref, y_pred_binary, y_scores)
        
        # 学习正常模式
        self._learn_normal_patterns(X_processed, y_pred_binary)
        
        # 确定阈值
        self.threshold = self._determine_threshold(y_scores, contamination)
        
        # 更新状态
        self.is_trained = True
        self.last_trained = datetime.now()
        
        # 保存模型
        self.save()
        
        detected_anomalies = np.sum(y_pred_binary)
        print(f"训练完成！检测到 {detected_anomalies}/{len(X_data)} 个异常 ({detected_anomalies/len(X_data)*100:.1f}%)")
        
        return {
            'status': 'success',
            'metrics': self.training_metrics,
            'detected_anomalies': int(detected_anomalies),
            'anomaly_rate': float(detected_anomalies / len(X_data)),
            'threshold': float(self.threshold),
            'model_info': self.get_model_info()
        }
    
    def detect(self, 
               node_data: Dict[str, Any],
               context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        检测异常
        
        Args:
            node_data: 记忆节点数据
            context_data: 上下文数据
            
        Returns:
            检测结果字典
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
        
        # 检测异常
        prediction = self.model.predict(features_scaled)[0]
        is_anomaly = prediction == -1  # 1=正常，-1=异常
        
        # 获取异常分数
        anomaly_score = 0.0
        if hasattr(self.model, 'decision_function'):
            score = self.model.decision_function(features_scaled)[0]
            # 转换分数：值越小越异常
            anomaly_score = 1.0 - (1.0 / (1.0 + np.exp(-score)))  # sigmoid转换
        elif hasattr(self.model, 'score_samples'):
            score = self.model.score_samples(features_scaled)[0]
            anomaly_score = 1.0 - (1.0 / (1.0 + np.exp(-score)))
        else:
            # 简单分数
            anomaly_score = 1.0 if is_anomaly else 0.0
        
        # 计算置信度
        confidence = self._calculate_confidence(features_scaled, anomaly_score)
        
        # 分析异常原因
        reasons = self._analyze_anomaly_reasons(features_scaled, features)
        
        # 严重程度评估
        severity = self._assess_severity(anomaly_score, reasons)
        
        return {
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': float(anomaly_score),
            'confidence': float(confidence),
            'severity': severity,
            'reasons': reasons,
            'threshold': float(self.threshold),
            'model_version': self.config.version,
            'detection_time': datetime.now().isoformat()
        }
    
    def batch_detect(self, 
                    data_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        批量检测异常
        
        Args:
            data_pairs: (节点数据, 上下文数据)对列表
            
        Returns:
            检测结果列表
        """
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用train()方法")
        
        results = []
        
        for node_data, context_data in data_pairs:
            try:
                detection = self.detect(node_data, context_data)
                results.append(detection)
            except Exception as e:
                print(f"异常检测失败: {e}")
                results.append({
                    'is_anomaly': False,
                    'anomaly_score': 0.0,
                    'error': str(e),
                    'model_version': self.config.version
                })
        
        return results
    
    def detect_context_anomalies(self, 
                                context_sequence: List[Dict[str, Any]],
                                window_size: int = 10) -> Dict[str, Any]:
        """
        检测上下文序列中的异常
        
        Args:
            context_sequence: 上下文序列
            window_size: 滑动窗口大小
            
        Returns:
            序列异常检测结果
        """
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        if len(context_sequence) < window_size:
            print(f"序列长度 ({len(context_sequence)}) 小于窗口大小 ({window_size})")
            window_size = len(context_sequence)
        
        anomalies = []
        anomaly_scores = []
        
        # 滑动窗口检测
        for i in range(len(context_sequence) - window_size + 1):
            window = context_sequence[i:i + window_size]
            
            # 提取窗口特征
            window_features = []
            for context in window:
                features = self.feature_extractor.extract_from_context(context)
                window_features.append(list(features.values()))
            
            # 计算窗口统计特征
            window_array = np.array(window_features)
            window_mean = np.mean(window_array, axis=0)
            window_std = np.std(window_array, axis=0)
            
            # 创建虚拟节点数据用于检测
            dummy_node = {
                'content': f'window_{i}',
                'node_type': 'context_window',
                'importance_score': 0.5
            }
            
            dummy_context = {
                'context_type': 'sequence_window',
                'timestamp': window[-1].get('timestamp', datetime.now().isoformat()),
                'window_stats': {
                    'mean_features': window_mean.tolist(),
                    'std_features': window_std.tolist(),
                    'start_index': i,
                    'end_index': i + window_size - 1
                }
            }
            
            # 检测异常
            detection = self.detect(dummy_node, dummy_context)
            
            if detection['is_anomaly']:
                anomalies.append({
                    'window_index': i,
                    'start_time': window[0].get('timestamp'),
                    'end_time': window[-1].get('timestamp'),
                    'anomaly_score': detection['anomaly_score'],
                    'severity': detection['severity'],
                    'reasons': detection['reasons']
                })
            
            anomaly_scores.append(detection['anomaly_score'])
        
        # 分析序列模式
        sequence_analysis = self._analyze_sequence_pattern(anomaly_scores, context_sequence)
        
        return {
            'total_windows': len(context_sequence) - window_size + 1,
            'anomalous_windows': len(anomalies),
            'anomaly_rate': len(anomalies) / max(1, len(context_sequence) - window_size + 1),
            'anomalies': anomalies,
            'anomaly_scores': anomaly_scores,
            'sequence_analysis': sequence_analysis,
            'window_size': window_size
        }
    
    def evaluate(self, 
                X_test: List[Dict[str, Any]],
                y_test: List[int]) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            X_test: 测试数据
            y_test: 测试标签（1=异常，0=正常）
            
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
        y_pred_binary = np.where(y_pred == 1, 0, 1)  # 转换标签
        
        # 获取异常分数
        y_score = None
        if hasattr(self.model, 'decision_function'):
            y_score = self.model.decision_function(X_processed)
        elif hasattr(self.model, 'score_samples'):
            y_score = self.model.score_samples(X_processed)
        
        # 计算指标
        metrics = MLMetrics.calculate_anomaly_metrics(
            np.array(y_test), y_pred_binary, y_score
        )
        
        # 测量推理时间
        time_metrics = MLMetrics.measure_inference_time(self, X_test)
        metrics.update(time_metrics)
        
        # 检查性能要求
        metrics['meets_requirements'] = (
            metrics.get('accuracy', 0) >= ml_config.min_accuracy and
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
            'scaler': self.scaler,
            'pca': self.pca,
            'threshold': self.threshold,
            'normal_patterns': self.normal_patterns,
            'training_metrics': self.training_metrics,
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
            'model_type': 'AnomalyDetector',
            'version': self.config.version,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'threshold': self.threshold,
            'training_metrics': self.training_metrics,
            'normal_patterns_summary': self._get_normal_patterns_summary(),
            'performance_requirements': {
                'min_accuracy': ml_config.min_accuracy,
                'max_inference_time_ms': ml_config.max_inference_time_ms
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"模型已保存到: {save_path}")
    
    @classmethod
    def load(cls, path: str) -> 'AnomalyDetector':
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
        detector = cls(config=model_data['config'])
        
        # 恢复状态
        detector.model = model_data['model']
        detector.feature_extractor = model_data['feature_extractor']
        detector.scaler = model_data['scaler']
        detector.pca = model_data['pca']
        detector.threshold = model_data['threshold']
        detector.normal_patterns = model_data['normal_patterns']
        detector.training_metrics = model_data['training_metrics']
        detector.is_trained = model_data['is_trained']
        detector.last_trained = model_data['last_trained']
        
        return detector
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_type': 'AnomalyDetector',
            'version': self.config.version,
            'is_trained': self.is_trained,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'threshold': self.threshold,
            'training_metrics': self.training_metrics,
            'normal_patterns_summary': self._get_normal_patterns_summary(),
            'performance': {
                'meets_accuracy_requirement': self.training_metrics.get('accuracy', 0) >= ml_config.min_accuracy,
                'meets_speed_requirement': self.training_metrics.get('inference_time_ms_mean', 100) <= ml_config.max_inference_time_ms
            }
        }
    
    def _create_model(self, method: str, contamination: float):
        """创建异常检测模型"""
        if method == 'isolation_forest':
            return IsolationForest(
                contamination=contamination,
                n_estimators=self.config.hyperparameters.get('n_estimators', 100),
                max_samples=self.config.hyperparameters.get('max_samples', 'auto'),
                random_state=self.config.hyperparameters.get('random_state', 42),
                n_jobs=-1
            )
        elif method == 'lof':
            return LocalOutlierFactor(
                contamination=contamination,
                n_neighbors=self.config.hyperparameters.get('n_neighbors', 20),
                algorithm='auto',
                novelty=True,  # 用于预测新样本
                n_jobs=-1
            )
        elif method == 'svm':
            return OneClassSVM(
                nu=contamination,  # nu大致对应异常比例
                kernel=self.config.hyperparameters.get('kernel', 'rbf'),
                gamma=self.config.hyperparameters.get('gamma', 'auto')
            )
        elif method == 'elliptic':
            return EllipticEnvelope(
                contamination=contamination,
                random_state=self.config.hyperparameters.get('random_state', 42)
            )
        else:
            return IsolationForest(
                contamination=contamination,
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
    
    def _train_supervised(self, X_processed, y_data, method):
        """监督学习训练（如果有标签）"""
        # 这里可以实现监督学习版本
        # 简化处理：仍然使用无监督学习
        print("监督学习训练（简化版）")
        
        # 分离正常和异常样本
        normal_indices = np.where(np.array(y_data) == 0)[0]
        anomaly_indices = np.where(np.array(y_data) ==        # 分离正常和异常样本
        normal_indices = np.where(np.array(y_data) == 0)[0]
        anomaly_indices = np.where(np.array(y_data) == 1)[0]
        
        print(f"正常样本: {len(normal_indices)}，异常样本: {len(anomaly_indices)}")
        
        # 只使用正常样本训练（半监督）
        X_normal = X_processed[normal_indices]
        
        # 训练模型
        contamination = len(anomaly_indices) / len(X_processed)
        self.model = self._create_model(method, contamination)
        self.model.fit(X_normal)
        
        # 在所有数据上评估
        y_pred = self.model.predict(X_processed)
        y_pred_binary = np.where(y_pred == 1, 0, 1)
        
        y_scores = None
        if hasattr(self.model, 'decision_function'):
            y_scores = self.model.decision_function(X_processed)
        
        # 评估
        self.training_metrics = self._evaluate_model(X_processed, y_data, y_pred_binary, y_scores)
        
        # 学习正常模式
        self._learn_normal_patterns(X_normal, np.zeros(len(X_normal)))
        
        # 确定阈值
        self.threshold = self._determine_threshold(y_scores, contamination)
        
        # 更新状态
        self.is_trained = True
        self.last_trained = datetime.now()
        
        return {
            'status': 'success',
            'metrics': self.training_metrics,
            'detected_anomalies': len(anomaly_indices),
            'anomaly_rate': contamination,
            'threshold': float(self.threshold),
            'supervised': True
        }
    
    def _evaluate_model(self, X, y_true, y_pred, y_scores):
        """评估模型"""
        metrics = MLMetrics.calculate_anomaly_metrics(y_true, y_pred, y_scores)
        
        # 添加基本统计
        metrics['num_samples'] = len(X)
        metrics['num_features'] = X.shape[1]
        metrics['anomaly_rate'] = np.mean(y_true) if y_true is not None else np.mean(y_pred)
        
        # 测量推理时间
        import time
        start_time = time.perf_counter()
        _ = self.model.predict(X[:100])
        end_time = time.perf_counter()
        metrics['inference_time_ms_mean'] = (end_time - start_time) * 1000 / 100
        
        return metrics
    
    def _learn_normal_patterns(self, X_normal, y_pred):
        """学习正常模式"""
        # 计算正常数据的统计特征
        if len(X_normal) > 0:
            self.normal_patterns = {
                'mean': np.mean(X_normal, axis=0).tolist(),
                'std': np.std(X_normal, axis=0).tolist(),
                'min': np.min(X_normal, axis=0).tolist(),
                'max': np.max(X_normal, axis=0).tolist(),
                'median': np.median(X_normal, axis=0).tolist(),
                'num_samples': len(X_normal)
            }
        else:
            self.normal_patterns = {}
    
    def _determine_threshold(self, scores, contamination):
        """确定异常阈值"""
        if scores is None:
            return 0.5
        
        # 根据异常比例确定阈值
        if len(scores) > 0:
            # 分数越小越异常（对于decision_function）
            sorted_scores = np.sort(scores)
            threshold_idx = int(contamination * len(scores))
            
            if threshold_idx < len(sorted_scores):
                threshold = sorted_scores[threshold_idx]
                
                # 转换为0-1范围
                min_score = np.min(scores)
                max_score = np.max(scores)
                
                if max_score > min_score:
                    normalized_threshold = (threshold - min_score) / (max_score - min_score)
                    return float(normalized_threshold)
        
        return 0.5
    
    def _calculate_confidence(self, features, anomaly_score):
        """计算检测置信度"""
        # 基于特征数量和异常分数
        n_features = features.shape[1]
        base_confidence = min(0.9, 0.6 + (n_features / 100))
        
        # 基于异常分数调整
        if anomaly_score > 0.8 or anomaly_score < 0.2:
            # 非常明显或非常正常的样本，置信度高
            confidence = base_confidence * 1.2
        elif 0.4 < anomaly_score < 0.6:
            # 边界样本，置信度低
            confidence = base_confidence * 0.7
        else:
            confidence = base_confidence
        
        return min(1.0, max(0.0, confidence))
    
    def _analyze_anomaly_reasons(self, features_scaled, original_features):
        """分析异常原因"""
        reasons = []
        
        if not self.normal_patterns:
            return ["无法分析原因：正常模式未学习"]
        
        # 检查每个特征
        feature_names = self.feature_extractor.get_feature_names('combined')
        
        for i, (feature_name, feature_value) in enumerate(zip(feature_names, features_scaled.values[0])):
            if i >= len(self.normal_patterns['mean']):
                break
            
            normal_mean = self.normal_patterns['mean'][i]
            normal_std = self.normal_patterns['std'][i]
            
            if normal_std > 0:
                # 计算Z-score
                z_score = abs((feature_value - normal_mean) / normal_std)
                
                if z_score > 3.0:
                    original_value = list(original_features.values())[i] if i < len(original_features) else feature_value
                    
                    if feature_value > normal_mean:
                        reasons.append(f"{feature_name} 异常高 (Z={z_score:.1f}, 值={original_value})")
                    else:
                        reasons.append(f"{feature_name} 异常低 (Z={z_score:.1f}, 值={original_value})")
        
        # 如果没有找到具体原因，提供一般性原因
        if not reasons:
            # 检查组合特征
            if 'node_context_similarity' in original_features:
                similarity = original_features['node_context_similarity']
                if similarity < 0.3:
                    reasons.append("节点与上下文相似度低")
            
            if 'time_alignment_hours' in original_features:
                time_diff = original_features['time_alignment_hours']
                if time_diff > 24:
                    reasons.append("时间对齐差异大")
            
            # 默认原因
            if not reasons:
                reasons.append("多特征组合异常")
        
        return reasons[:5]  # 返回前5个原因
    
    def _assess_severity(self, anomaly_score, reasons):
        """评估异常严重程度"""
        # 基于异常分数
        if anomaly_score > 0.9:
            base_severity = 'critical'
        elif anomaly_score > 0.7:
            base_severity = 'high'
        elif anomaly_score > 0.5:
            base_severity = 'medium'
        else:
            base_severity = 'low'
        
        # 基于原因调整
        reason_keywords = {
            'critical': ['严重', '崩溃', '错误', '失败'],
            'high': ['高', '重大', '重要', '安全'],
            'medium': ['中', '一般', '普通'],
            'low': ['低', '轻微', '小']
        }
        
        for severity, keywords in reason_keywords.items():
            for reason in reasons:
                if any(keyword in reason for keyword in keywords):
                    # 如果原因中包含关键词，提升严重程度
                    severity_levels = ['low', 'medium', 'high', 'critical']
                    current_idx = severity_levels.index(base_severity)
                    keyword_idx = severity_levels.index(severity)
                    
                    if keyword_idx > current_idx:
                        base_severity = severity
                        break
        
        return base_severity
    
    def _analyze_sequence_pattern(self, anomaly_scores, context_sequence):
        """分析序列模式"""
        if len(anomaly_scores) < 2:
            return {'pattern': 'insufficient_data'}
        
        # 计算统计
        mean_score = np.mean(anomaly_scores)
        std_score = np.std(anomaly_scores)
        max_score = np.max(anomaly_scores)
        min_score = np.min(anomaly_scores)
        
        # 检测趋势
        from scipy import stats
        if len(anomaly_scores) > 5:
            x = np.arange(len(anomaly_scores))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, anomaly_scores)
            
            if abs(slope) > 0.01:
                trend = 'increasing' if slope > 0 else 'decreasing'
                trend_strength = abs(r_value)
            else:
                trend = 'stable'
                trend_strength = 0.0
        else:
            trend = 'unknown'
            trend_strength = 0.0
        
        # 检测周期性（简化版）
        autocorrelation = 0.0
        if len(anomaly_scores) > 10:
            try:
                from statsmodels.tsa.stattools import acf
                autocorr = acf(anomaly_scores, nlags=min(5, len(anomaly_scores)//2))
                autocorrelation = np.max(np.abs(autocorr[1:])) if len(autocorr) > 1 else 0.0
            except:
                pass
        
        # 上下文变化分析
        context_changes = 0
        if len(context_sequence) > 1:
            for i in range(1, len(context_sequence)):
                if context_sequence[i].get('context_type') != context_sequence[i-1].get('context_type'):
                    context_changes += 1
        
        return {
            'mean_anomaly_score': float(mean_score),
            'std_anomaly_score': float(std_score),
            'max_anomaly_score': float(max_score),
            'min_anomaly_score': float(min_score),
            'trend': trend,
            'trend_strength': float(trend_strength),
            'autocorrelation': float(autocorrelation),
            'context_changes': context_changes,
            'pattern_summary': self._generate_pattern_summary(mean_score, std_score, trend, autocorrelation)
        }
    
    def _generate_pattern_summary(self, mean_score, std_score, trend, autocorrelation):
        """生成模式摘要"""
        summary_parts = []
        
        if mean_score > 0.7:
            summary_parts.append("高频异常")
        elif mean_score < 0.3:
            summary_parts.append("低频异常")
        else:
            summary_parts.append("中等频率异常")
        
        if std_score > 0.2:
            summary_parts.append("波动大")
        
        if trend == 'increasing':
            summary_parts.append("趋势上升")
        elif trend == 'decreasing':
            summary_parts.append("趋势下降")
        
        if autocorrelation > 0.5:
            summary_parts.append("可能具有周期性")
        
        return "，".join(summary_parts) if summary_parts else "无明显模式"
    
    def _get_normal_patterns_summary(self):
        """获取正常模式摘要"""
        if not self.normal_patterns:
            return "未学习正常模式"
        
        summary = []
        summary.append(f"样本数: {self.normal_patterns.get('num_samples', 0)}")
        
        if 'mean' in self.normal_patterns and len(self.normal_patterns['mean']) > 0:
            mean_values = self.normal_patterns['mean']
            if len(mean_values) >= 3:
                summary.append(f"特征均值范围: [{min(mean_values[:3]):.2f}, {max(mean_values[:3]):.2f}]")
        
        return "; ".join(summary)