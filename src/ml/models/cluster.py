"""
记忆模式聚类模型
无监督学习发现记忆模式
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

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

from ..config import ModelType, ModelConfig, ml_config
from ..utils.feature_extractor import FeatureExtractor
from ..utils.metrics import MLMetrics


class MemoryPatternCluster:
    """记忆模式聚类模型"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        初始化聚类模型
        
        Args:
            config: 模型配置
        """
        self.config = config or ml_config.default_cluster_config
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None
        self.is_trained = False
        self.cluster_labels = []
        self.cluster_centers = []
        self.cluster_stats = {}
        self.training_metrics = {}
        self.last_trained = None
        
    def train(self, 
              X_data: List[Dict[str, Any]],
              n_clusters: Optional[int] = None,
              method: str = 'kmeans',
              reduce_dimensions: bool = True) -> Dict[str, Any]:
        """
        训练聚类模型
        
        Args:
            X_data: 训练数据（记忆节点数据）
            n_clusters: 聚类数量（如果为None则自动确定）
            method: 聚类方法 ('kmeans', 'dbscan', 'hierarchical', 'gmm')
            reduce_dimensions: 是否降维
            
        Returns:
            训练结果字典
        """
        print(f"开始训练记忆模式聚类模型，数据量: {len(X_data)}")
        
        # 提取特征
        X_processed = self.feature_extractor.prepare_training_data(
            X_data, feature_type='node'
        )
        
        # 降维（可选）
        if reduce_dimensions and X_processed.shape[1] > 50:
            print(f"特征维度较高 ({X_processed.shape[1]})，进行降维...")
            X_processed = self._reduce_dimensions(X_processed)
        
        # 确定聚类数量
        if n_clusters is None:
            n_clusters = self._determine_optimal_clusters(X_processed)
            print(f"自动确定最优聚类数量: {n_clusters}")
        
        # 选择并训练模型
        self.model = self._create_model(method, n_clusters)
        print(f"使用 {method} 方法进行聚类...")
        
        self.cluster_labels = self.model.fit_predict(X_processed)
        
        # 处理噪声标签（DBSCAN可能返回-1）
        if method == 'dbscan':
            # 将噪声点分配到单独的簇
            noise_mask = self.cluster_labels == -1
            if np.any(noise_mask):
                max_label = np.max(self.cluster_labels)
                self.cluster_labels[noise_mask] = max_label + 1
        
        # 计算聚类中心
        self._calculate_cluster_centers(X_processed)
        
        # 计算聚类统计
        self._calculate_cluster_stats(X_processed, X_data)
        
        # 评估聚类质量
        self.training_metrics = self._evaluate_clustering(X_processed)
        
        # 更新状态
        self.is_trained = True
        self.last_trained = datetime.now()
        
        # 保存模型
        self.save()
        
        print(f"训练完成！发现 {len(set(self.cluster_labels))} 个聚类")
        print(f"轮廓系数: {self.training_metrics.get('silhouette_score', 0):.4f}")
        
        return {
            'status': 'success',
            'metrics': self.training_metrics,
            'cluster_stats': self.cluster_stats,
            'num_clusters': len(set(self.cluster_labels)),
            'model_info': self.get_model_info()
        }
    
    def predict(self, 
                node_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        预测记忆节点所属聚类
        
        Args:
            node_data: 记忆节点数据
            
        Returns:
            聚类结果字典
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
        
        # 如果需要，应用相同的降维
        if self.pca is not None:
            features_scaled = self.pca.transform(features_scaled)
        
        # 预测聚类
        cluster_label = int(self.model.predict(features_scaled)[0])
        
        # 获取聚类信息
        cluster_info = self.cluster_stats.get(str(cluster_label), {})
        
        # 计算到聚类中心的距离
        distance_to_center = 0.0
        if cluster_label < len(self.cluster_centers):
            center = self.cluster_centers[cluster_label]
            distance_to_center = np.linalg.norm(features_scaled - center)
        
        # 计算异常分数（距离越远，越可能是异常）
        max_distance = max([np.linalg.norm(center) for center in self.cluster_centers]) if self.cluster_centers else 1.0
        anomaly_score = min(1.0, distance_to_center / (max_distance + 1e-10))
        
        return {
            'cluster_label': cluster_label,
            'cluster_name': cluster_info.get('name', f'Cluster_{cluster_label}'),
            'cluster_size': cluster_info.get('size', 0),
            'distance_to_center': float(distance_to_center),
            'anomaly_score': float(anomaly_score),
            'is_outlier': anomaly_score > 0.7,  # 阈值可调整
            'cluster_characteristics': cluster_info.get('characteristics', {}),
            'model_version': self.config.version,
            'prediction_time': datetime.now().isoformat()
        }
    
    def batch_predict(self, 
                     node_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量预测
        
        Args:
            node_data_list: 记忆节点数据列表
            
        Returns:
            聚类结果列表
        """
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用train()方法")
        
        results = []
        
        for node_data in node_data_list:
            try:
                prediction = self.predict(node_data)
                results.append(prediction)
            except Exception as e:
                print(f"预测失败: {e}")
                results.append({
                    'cluster_label': -1,
                    'error': str(e),
                    'model_version': self.config.version
                })
        
        return results
    
    def discover_patterns(self, 
                         node_data_list: List[Dict[str, Any]],
                         min_cluster_size: int = 5) -> Dict[str, Any]:
        """
        发现记忆模式
        
        Args:
            node_data_list: 记忆节点数据列表
            min_cluster_size: 最小聚类大小
            
        Returns:
            模式发现结果
        """
        # 批量预测
        predictions = self.batch_predict(node_data_list)
        
        # 按聚类分组
        clusters = {}
        for i, (node_data, pred) in enumerate(zip(node_data_list, predictions)):
            cluster_label = pred['cluster_label']
            
            if cluster_label not in clusters:
                clusters[cluster_label] = {
                    'nodes': [],
                    'predictions': [],
                    'size': 0
                }
            
            clusters[cluster_label]['nodes'].append(node_data)
            clusters[cluster_label]['predictions'].append(pred)
            clusters[cluster_label]['size'] += 1
        
        # 分析每个聚类
        patterns = {}
        for cluster_label, cluster_data in clusters.items():
            if cluster_data['size'] < min_cluster_size:
                continue  # 跳过太小的聚类
            
            # 分析聚类特征
            pattern = self._analyze_cluster_pattern(
                cluster_label, 
                cluster_data['nodes'],
                cluster_data['predictions']
            )
            
            patterns[cluster_label] = pattern
        
        # 识别常见模式
        common_patterns = self._identify_common_patterns(patterns)
        
        return {
            'patterns': patterns,
            'common_patterns': common_patterns,
            'cluster_distribution': {label: data['size'] for label, data in clusters.items()},
            'total_patterns_discovered': len(patterns)
        }
    
    def visualize_clusters(self, 
                          node_data_list: List[Dict[str, Any]],
                          save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        可视化聚类结果
        
        Args:
            node_data_list: 记忆节点数据列表
            save_path: 保存路径（可选）
            
        Returns:
            可视化结果
        """
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        # 提取特征
        X_processed = self.feature_extractor.prepare_training_data(
            node_data_list, feature_type='node'
        )
        
        # 降维到2D用于可视化
        if X_processed.shape[1] > 2:
            print("降维到2D用于可视化...")
            X_2d = self._reduce_to_2d(X_processed)
        else:
            X_2d = X_processed
        
        # 获取聚类标签
        predictions = self.batch_predict(node_data_list)
        labels = [pred['cluster_label'] for pred in predictions]
        
        # 创建可视化
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 散点图
        scatter = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab20', alpha=0.6)
        axes[0].set_title('记忆节点聚类可视化')
        axes[0].set_xlabel('维度 1')
        axes[0].set_ylabel('维度 2')
        plt.colorbar(scatter, ax=axes[0], label='聚类标签')
        
        # 添加聚类中心
        if hasattr(self.model, 'cluster_centers_'):
            centers_2d = self._reduce_to_2d(self.model.cluster_centers_)
            axes[0].scatter(centers_2d[:, 0], centers_2d[:, 1], 
                          c='red', marker='X', s=200, label='聚类中心')
            axes[0].legend()
        
        # 聚类大小分布
        unique_labels, counts = np.unique(labels, return_counts=True)
        axes[1].bar(unique_labels, counts)
        axes[1].set_title('聚类大小分布')
        axes[1].set_xlabel('聚类标签')
        axes[1].set_ylabel('节点数量')
        axes[1].set_xticks(unique_labels)
        
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化已保存到: {save_path}")
        
        plt.close()
        
        return {
            'visualization_generated': True,
            'save_path': save_path,
            'cluster_distribution': dict(zip(unique_labels.astype(str), counts.tolist())),
            'num_clusters': len(unique_labels)
        }
    
    def evaluate(self, 
                X_test: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估聚类质量
        
        Args:
            X_test: 测试数据
            
        Returns:
            评估结果
        """
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        # 提取特征
        X_processed = self.feature_extractor.prepare_training_data(
            X_test, feature_type='node'
        )
        
        # 预测聚类
        predictions = self.batch_predict(X_test)
        labels = np.array([pred['cluster_label'] for pred in predictions])
        
        # 计算聚类指标
        metrics = MLMetrics.calculate_clustering_metrics(X_processed, labels)
        
        # 测量推理时间
        time_metrics = MLMetrics.measure_inference_time(self, X_test)
        metrics.update(time_metrics)
        
        # 检查性能要求
        metrics['meets_requirements'] = (
            metrics.get('silhouette_score', -1) >= 0.5 and  # 轮廓系数阈值
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
            'cluster_labels': self.cluster_labels,
            'cluster_centers': self.cluster_centers,
            'cluster_stats': self.cluster_stats,
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
            'model_type': 'MemoryPatternCluster',
            'version': self.config.version,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'num_clusters': len(set(self.cluster_labels)),
            'cluster_stats_summary': self._get_cluster_stats_summary(),
            'training_metrics': self.training_metrics,
            'performance_requirements': {
                'min_silhouette_score': 0.5,
                'max_inference_time_ms': ml_config.max_inference_time_ms
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"模型已保存到: {save_path}")
    
    @classmethod
    def load(cls, path: str) -> 'MemoryPatternCluster':
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
        cluster = cls(config=model_data['config'])
        
        # 恢复状态
        cluster.model = model_data['model']
        cluster.feature_extractor = model_data['feature_extractor']
        cluster.scaler = model_data['scaler']
        cluster.pca = model_data['pca']
        cluster.cluster_labels = model_data['cluster_labels']
        cluster.cluster_centers = model_data['cluster_centers']
        cluster.cluster_stats = model_data['cluster_stats']
        cluster.training_metrics = model_data['training_metrics']
        cluster.is_trained = model_data['is_trained']
        cluster.last_trained = model_data['last_trained']
        
        return cluster
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_type': 'MemoryPatternCluster',
            'version': self.config.version,
            'is_trained': self.is_trained,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'num_clusters': len(set(self.cluster_labels)),
            'cluster_stats_summary': self._get_cluster_stats_summary(),
            'training_metrics': self.training_metrics,
            'performance': {
                'meets_quality_requirement': self.training_metrics.get('silhouette_score', -1) >= 0.5,
                'meets_speed_requirement': self.training_metrics.get('inference_time_ms_mean', 100) <= ml_config.max_inference_time_ms
            }
        }
    
    def _create_model(self, method: str, n_clusters: int):
        """创建聚类模型"""
        if method == 'kmeans':
            return KMeans(
                n_clusters=n_clusters,
                random_state=self.config.hyperparameters.get('random_state', 42),
                n_init=self.config.hyperparameters.get('n_init', 10)
            )
        elif method == 'dbscan':
            return DBSCAN(
                eps=self.config.hyperparameters.get('eps', 0.5),
                min_samples=self.config.hyperparameters.get('min_samples', 5)
            )
        elif method == 'hierarchical':
            return AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=self.config.hyperparameters.get('linkage', 'ward')
            )
        elif method == 'gmm':
            return GaussianMixture(
                n_components=n_clusters,
                random_state=self.config.hyper                random_state=self.config.hyperparameters.get('random_state', 42)
            )
        else:
            return KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
    
    def _determine_optimal_clusters(self, X: np.ndarray, max_clusters: int = 15) -> int:
        """确定最优聚类数量"""
        if len(X) < 10:
            return min(3, len(X))
        
        silhouette_scores = []
        max_clusters = min(max_clusters, len(X) - 1)
        
        for n in range(2, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=n, random_state=42, n_init=5)
                labels = kmeans.fit_predict(X)
                
                if len(set(labels)) > 1:
                    score = silhouette_score(X, labels)
                    silhouette_scores.append(score)
                else:
                    silhouette_scores.append(-1)
            except:
                silhouette_scores.append(-1)
        
        # 找到轮廓系数最高的聚类数量
        if silhouette_scores:
            best_n = np.argmax(silhouette_scores) + 2  # +2因为从2开始
            return best_n
        else:
            return 3
    
    def _reduce_dimensions(self, X: np.ndarray, n_components: int = 50) -> np.ndarray:
        """降维"""
        if X.shape[1] <= n_components:
            return X
        
        self.pca = PCA(n_components=min(n_components, X.shape[0]), random_state=42)
        X_reduced = self.pca.fit_transform(X)
        
        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        print(f"PCA降维: {X.shape[1]} -> {X_reduced.shape[1]} 维")
        print(f"保留方差: {explained_variance:.2%}")
        
        return X_reduced
    
    def _reduce_to_2d(self, X: np.ndarray) -> np.ndarray:
        """降维到2D用于可视化"""
        if X.shape[1] <= 2:
            return X
        
        # 使用t-SNE或PCA
        if X.shape[0] < 1000:
            # t-SNE对于小数据集效果更好
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X.shape[0] - 1))
            X_2d = tsne.fit_transform(X)
        else:
            # PCA对于大数据集更快
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X)
        
        return X_2d
    
    def _calculate_cluster_centers(self, X: np.ndarray):
        """计算聚类中心"""
        if hasattr(self.model, 'cluster_centers_'):
            self.cluster_centers = self.model.cluster_centers_
        else:
            # 对于没有中心概念的算法，计算均值
            unique_labels = np.unique(self.cluster_labels)
            self.cluster_centers = []
            
            for label in unique_labels:
                mask = self.cluster_labels == label
                if np.any(mask):
                    center = np.mean(X[mask], axis=0)
                    self.cluster_centers.append(center)
                else:
                    self.cluster_centers.append(np.zeros(X.shape[1]))
    
    def _calculate_cluster_stats(self, X: np.ndarray, original_data: List[Dict[str, Any]]):
        """计算聚类统计"""
        unique_labels = np.unique(self.cluster_labels)
        
        for label in unique_labels:
            mask = self.cluster_labels == label
            cluster_data = [original_data[i] for i in range(len(original_data)) if mask[i]]
            
            # 基本统计
            stats = {
                'size': int(np.sum(mask)),
                'name': f'Cluster_{label}',
                'indices': np.where(mask)[0].tolist()
            }
            
            # 特征统计
            if np.any(mask):
                cluster_features = X[mask]
                
                # 均值特征
                mean_features = np.mean(cluster_features, axis=0)
                stats['mean_features'] = mean_features.tolist()
                
                # 方差特征
                std_features = np.std(cluster_features, axis=0)
                stats['std_features'] = std_features.tolist()
                
                # 特征范围
                min_features = np.min(cluster_features, axis=0)
                max_features = np.max(cluster_features, axis=0)
                stats['feature_range'] = {
                    'min': min_features.tolist(),
                    'max': max_features.tolist()
                }
            
            # 内容分析
            if cluster_data:
                # 提取常见标签
                all_tags = []
                for data in cluster_data:
                    tags = data.get('tags', [])
                    if isinstance(tags, list):
                        all_tags.extend(tags)
                
                # 统计标签频率
                from collections import Counter
                tag_counter = Counter(all_tags)
                stats['common_tags'] = tag_counter.most_common(10)
                
                # 提取常见类别
                categories = [data.get('category', 'unknown') for data in cluster_data]
                category_counter = Counter(categories)
                stats['common_categories'] = category_counter.most_common(5)
                
                # 内容长度统计
                content_lengths = [len(str(data.get('content', ''))) for data in cluster_data]
                stats['content_stats'] = {
                    'avg_length': np.mean(content_lengths),
                    'min_length': np.min(content_lengths),
                    'max_length': np.max(content_lengths)
                }
            
            # 聚类特征
            stats['characteristics'] = self._extract_cluster_characteristics(cluster_data)
            
            self.cluster_stats[str(label)] = stats
    
    def _extract_cluster_characteristics(self, cluster_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """提取聚类特征"""
        if not cluster_data:
            return {}
        
        characteristics = {}
        
        # 节点类型分布
        node_types = [data.get('node_type', 'unknown') for data in cluster_data]
        from collections import Counter
        type_counter = Counter(node_types)
        characteristics['node_type_distribution'] = dict(type_counter.most_common())
        
        # 平均重要性分数
        importance_scores = [data.get('importance_score', 0.5) for data in cluster_data]
        characteristics['avg_importance'] = np.mean(importance_scores) if importance_scores else 0.5
        
        # 平均访问次数
        access_counts = [data.get('access_count', 0) for data in cluster_data]
        characteristics['avg_access_count'] = np.mean(access_counts) if access_counts else 0
        
        # 时间特征
        created_dates = []
        for data in cluster_data:
            created_at = data.get('created_at')
            if created_at:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    created_dates.append(dt)
                except:
                    pass
        
        if created_dates:
            characteristics['avg_age_days'] = np.mean([
                (datetime.now() - dt).days for dt in created_dates
            ])
        
        # 关联特征
        association_counts = [len(data.get('associations', [])) for data in cluster_data]
        characteristics['avg_associations'] = np.mean(association_counts) if association_counts else 0
        
        return characteristics
    
    def _evaluate_clustering(self, X: np.ndarray) -> Dict[str, float]:
        """评估聚类质量"""
        metrics = {}
        
        unique_labels = np.unique(self.cluster_labels)
        n_clusters = len(unique_labels)
        
        if n_clusters > 1 and n_clusters < len(X):
            try:
                metrics['silhouette_score'] = silhouette_score(X, self.cluster_labels)
                metrics['davies_bouldin_score'] = davies_bouldin_score(X, self.cluster_labels)
                
                # 计算聚类平衡度
                cluster_sizes = [np.sum(self.cluster_labels == label) for label in unique_labels]
                metrics['cluster_balance'] = 1 - (np.std(cluster_sizes) / np.mean(cluster_sizes))
                
            except Exception as e:
                print(f"聚类评估失败: {e}")
                metrics['silhouette_score'] = -1
                metrics['davies_bouldin_score'] = float('inf')
                metrics['cluster_balance'] = 0
        else:
            metrics['silhouette_score'] = -1
            metrics['davies_bouldin_score'] = float('inf')
            metrics['cluster_balance'] = 0
        
        # 基本统计
        metrics['num_clusters'] = n_clusters
        metrics['total_samples'] = len(X)
        
        # 测量推理时间（简化版）
        import time
        start_time = time.perf_counter()
        _ = self.model.predict(X[:10])
        end_time = time.perf_counter()
        metrics['inference_time_ms_mean'] = (end_time - start_time) * 1000 / 10
        
        return metrics
    
    def _analyze_cluster_pattern(self, 
                                cluster_label: int,
                                nodes: List[Dict[str, Any]],
                                predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析聚类模式"""
        pattern = {
            'cluster_label': cluster_label,
            'size': len(nodes),
            'name': f'Pattern_{cluster_label}',
            'description': '',
            'key_characteristics': [],
            'typical_nodes': [],
            'anomaly_rate': 0.0
        }
        
        # 计算异常率
        anomaly_scores = [pred.get('anomaly_score', 0) for pred in predictions]
        pattern['anomaly_rate'] = np.mean([score > 0.7 for score in anomaly_scores])
        
        # 提取关键特征
        if nodes:
            # 分析常见特征
            common_categories = {}
            common_tags = {}
            
            for node in nodes:
                category = node.get('category', 'unknown')
                common_categories[category] = common_categories.get(category, 0) + 1
                
                tags = node.get('tags', [])
                for tag in tags:
                    common_tags[tag] = common_tags.get(tag, 0) + 1
            
            # 找出最常见的类别和标签
            if common_categories:
                top_category = max(common_categories.items(), key=lambda x: x[1])
                pattern['key_characteristics'].append(f"主要类别: {top_category[0]} ({top_category[1]}/{len(nodes)})")
            
            if common_tags:
                top_tags = sorted(common_tags.items(), key=lambda x: x[1], reverse=True)[:3]
                pattern['key_characteristics'].extend([f"常见标签: {tag} ({count})" for tag, count in top_tags])
            
            # 分析重要性分布
            importance_scores = [node.get('importance_score', 0.5) for node in nodes]
            avg_importance = np.mean(importance_scores)
            pattern['key_characteristics'].append(f"平均重要性: {avg_importance:.2f}")
            
            # 选择典型节点（重要性最高的）
            if nodes:
                sorted_nodes = sorted(nodes, key=lambda x: x.get('importance_score', 0), reverse=True)
                pattern['typical_nodes'] = [
                    {
                        'id': node.get('id', f'node_{i}'),
                        'content_preview': str(node.get('content', ''))[:100] + '...',
                        'importance': node.get('importance_score', 0.5)
                    }
                    for i, node in enumerate(sorted_nodes[:3])
                ]
        
        # 生成描述
        if pattern['key_characteristics']:
            pattern['description'] = f"包含 {pattern['size']} 个节点的聚类。特征包括: " + "; ".join(pattern['key_characteristics'][:3])
        else:
            pattern['description'] = f"包含 {pattern['size']} 个节点的聚类"
        
        return pattern
    
    def _identify_common_patterns(self, patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别常见模式"""
        common_patterns = []
        
        for pattern in patterns.values():
            # 根据大小和特征判断是否为常见模式
            if pattern['size'] >= 10 and pattern['anomaly_rate'] < 0.3:
                common_patterns.append(pattern)
        
        # 按大小排序
        common_patterns.sort(key=lambda x: x['size'], reverse=True)
        
        return common_patterns
    
    def _get_cluster_stats_summary(self):
        """获取聚类统计摘要"""
        if not self.cluster_stats:
            return "无统计信息"
        
        summary = []
        for label, stats in self.cluster_stats.items():
            summary.append(f"聚类{label}: {stats['size']}个节点")
        
        return "; ".join(summary[:5]) + (f" ... 共{len(self.cluster_stats)}个聚类" if len(self.cluster_stats) > 5 else "")