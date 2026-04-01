"""
机器学习指标计算工具
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    precision_recall_curve, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')


class MLMetrics:
    """机器学习指标计算器"""
    
    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, 
                                        y_pred: np.ndarray,
                                        y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        计算分类指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率（可选）
            
        Returns:
            指标字典
        """
        metrics = {}
        
        # 基础指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # 多分类指标
        unique_classes = np.unique(y_true)
        if len(unique_classes) > 2:
            # 计算每个类的指标
            for cls in unique_classes:
                y_true_binary = (y_true == cls).astype(int)
                y_pred_binary = (y_pred == cls).astype(int)
                
                if len(np.unique(y_true_binary)) > 1:
                    metrics[f'precision_class_{cls}'] = precision_score(
                        y_true_binary, y_pred_binary, zero_division=0
                    )
                    metrics[f'recall_class_{cls}'] = recall_score(
                        y_true_binary, y_pred_binary, zero_division=0
                    )
                    metrics[f'f1_class_{cls}'] = f1_score(
                        y_true_binary, y_pred_binary, zero_division=0
                    )
        
        # 概率相关指标
        if y_prob is not None:
            try:
                if len(unique_classes) == 2:
                    # 二分类
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    # 多分类
                    metrics['roc_auc_ovr'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average='weighted'
                    )
                    metrics['roc_auc_ovo'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovo', average='weighted'
                    )
            except:
                pass
        
        # 混淆矩阵统计
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # 计算每个类的支持度
        class_support = np.sum(cm, axis=1)
        for i, support in enumerate(class_support):
            metrics[f'support_class_{i}'] = int(support)
        
        return metrics
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, 
                                    y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算回归指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            指标字典
        """
        metrics = {}
        
        # 基础指标
        metrics['r2_score'] = r2_score(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        # 统计指标
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        metrics['max_residual'] = np.max(np.abs(residuals))
        
        # 分位数误差
        for q in [0.25, 0.5, 0.75, 0.9]:
            metrics[f'quantile_error_{q}'] = np.percentile(np.abs(residuals), q * 100)
        
        return metrics
    
    @staticmethod
    def calculate_clustering_metrics(X: np.ndarray, 
                                    labels: np.ndarray) -> Dict[str, float]:
        """
        计算聚类指标
        
        Args:
            X: 特征数据
            labels: 聚类标签
            
        Returns:
            指标字典
        """
        metrics = {}
        
        n_clusters = len(np.unique(labels))
        n_samples = len(X)
        
        if n_clusters > 1 and n_clusters < n_samples:
            try:
                metrics['silhouette_score'] = silhouette_score(X, labels)
                metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
            except:
                metrics['silhouette_score'] = -1
                metrics['davies_bouldin_score'] = float('inf')
                metrics['calinski_harabasz_score'] = 0
        
        # 聚类统计
        cluster_sizes = np.bincount(labels + np.min(labels))  # 处理负标签
        metrics['n_clusters'] = n_clusters
        metrics['avg_cluster_size'] = np.mean(cluster_sizes)
        metrics['std_cluster_size'] = np.std(cluster_sizes)
        metrics['min_cluster_size'] = np.min(cluster_sizes)
        metrics['max_cluster_size'] = np.max(cluster_sizes)
        
        # 聚类平衡度
        if n_clusters > 1:
            metrics['cluster_balance'] = 1 - (np.std(cluster_sizes) / np.mean(cluster_sizes))
        else:
            metrics['cluster_balance'] = 1.0
        
        return metrics
    
    @staticmethod
    def calculate_recommendation_metrics(y_true: List[List[Any]],
                                        y_pred: List[List[Any]],
                                        k: int = 10) -> Dict[str, float]:
        """
        计算推荐指标
        
        Args:
            y_true: 真实项目列表（每个用户）
            y_pred: 预测项目列表（每个用户）
            k: Top-K指标
            
        Returns:
            指标字典
        """
        metrics = {}
        
        precision_at_k_list = []
        recall_at_k_list = []
        ndcg_at_k_list = []
        
        for true_items, pred_items in zip(y_true, y_pred):
            if not true_items or not pred_items:
                continue
            
            # 取前k个预测
            pred_items_k = pred_items[:k]
            
            # 计算命中数
            hits = len(set(true_items).intersection(set(pred_items_k)))
            
            # Precision@K
            precision_at_k = hits / len(pred_items_k) if pred_items_k else 0
            precision_at_k_list.append(precision_at_k)
            
            # Recall@K
            recall_at_k = hits / len(true_items) if true_items else 0
            recall_at_k_list.append(recall_at_k)
            
            # NDCG@K
            dcg = 0
            for i, item in enumerate(pred_items_k):
                if item in true_items:
                    dcg += 1 / np.log2(i + 2)  # i+2因为索引从0开始
            
            # 理想DCG
            idcg = 0
            min_len = min(len(true_items), k)
            for i in range(min_len):
                idcg += 1 / np.log2(i + 2)
            
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_at_k_list.append(ndcg)
        
        # 聚合指标
        if precision_at_k_list:
            metrics[f'precision@{k}'] = np.mean(precision_at_k_list)
            metrics[f'recall@{k}'] = np.mean(recall_at_k_list)
            metrics[f'ndcg@{k}'] = np.mean(ndcg_at_k_list)
            metrics[f'coverage@{k}'] = len(set().union(*[set(p[:k]) for p in y_pred])) / len(set().union(*[set(t) for t in y_true]))
        else:
            metrics[f'precision@{k}'] = 0
            metrics[f'recall@{k}'] = 0
            metrics[f'ndcg@{k}'] = 0
            metrics[f'coverage@{k}'] = 0
        
        return metrics
    
    @staticmethod
    def calculate_anomaly_metrics(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 y_score: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        计算异常检测指标
        
        Args:
            y_true: 真实标签（0=正常，1=异常）
            y_pred: 预测标签
            y_score: 异常分数（可选）
            
        Returns:
            指标字典
        """
        metrics = {}
        
        # 基础分类指标
        base_metrics = MLMetrics.calculate_classification_metrics(y_true, y_pred)
        metrics.update({k: v for k, v in base_metrics.items() 
                       if not k.startswith('confusion') and not k.startswith('support')})
        
        # 异常检测特定指标
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            metrics['true_positives'] = int(tp)
            metrics['false_positives'] = int(fp)
            metrics['true_negatives'] = int(tn)
            metrics['false_negatives'] = int(fn)
            
            # 异常检测指标
            metrics['detection_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['precision_anomaly'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['f1_anomaly'] = 2 * metrics['precision_anomaly'] * metrics['detection_rate'] / (
                metrics['precision_anomaly'] + metrics['detection_rate'] + 1e-10
            )
        
        # 分数相关指标
        if y_score is not None:
            try:
                metrics['auc_pr'] = MLMetrics._calculate_auc_pr(y_true, y_score)
                metrics['auc_roc'] = roc_auc_score(y_true, y_score)
            except:
                metrics['auc_pr'] = 0
                metrics['auc_roc'] = 0
        
        return metrics
    
    @staticmethod
    def measure_inference_time(model, X: np.ndarray, n_runs: int = 100) -> Dict[str, float]:
        """
        测量推理时间
        
        Args:
            model: 机器学习模型
            X: 输入数据
            n_runs: 运行次数
            
        Returns:
            时间指标字典
        """
        times = []
        
        # 预热
        for _ in range(10):
            _ = model.predict(X[:1])
        
        # 测量时间
        for _ in range(n_runs):
            start_time = time.perf_counter()
            _ = model.predict(X)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        # 计算统计量
        times_array = np.array(times)
        
        metrics = {
            'inference_time_ms_mean': np.mean(times_array),
            'inference_time_ms_std': np.std(times_array),
            'inference_time_ms_min': np.min(times_array),
            'inference_time_ms_max': np.max(times_array),
            'inference_time_ms_p95': np.percentile(times_array, 95),
            'inference_time_ms_p99': np.percentile(times_array, 99),
            'throughput_samples_per_second': 1000 / np.mean(times_array) * X.shape[0] if np.mean(times_array) > 0 else 0
        }
        
        return metrics
    
    @staticmethod
    def evaluate_model_performance(model, 
                                  X_test: np.ndarray,
                                  y_test: np.ndarray,
                                  task_type: str = 'classification',
                                  **kwargs) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            model: 机器学习模型
            X_test: 测试特征
            y_test: 测试标签
            task_type: 任务类型 ('classification', 'regression', 'clustering', 'anomaly')
            
        Returns:
            性能评估字典
        """
        results = {}
        
        # 预测
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = (time.time() - start_time) * 1000  # 毫秒
        
        results['inference_time_ms'] = inference_time
        
        # 根据任务类型计算指标
        if task_type == 'classification':
            y_prob = None
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
            
            metrics = MLMetrics.calculate_classification_metrics(y_test, y_pred, y_prob)
            results.update(metrics)
            
        elif task_type == 'regression':
            metrics = MLMetrics.calculate_regression_metrics(y_test, y_pred)
            results.update(metrics)
            
        elif task_type == 'clustering':
            metrics = MLMetrics.calculate_clustering_metrics(X_test, y_pred)
            results.update(metrics)
            
        elif task_type == 'anomaly':
            y_score = None
            if hasattr(model, 'decision_function'):
                y_score = model.decision_function(X_test)
            elif hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
                y_score = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0]
            
            metrics = MLMetrics.calculate_anomaly_metrics(y_test, y_pred, y_score)
            results.update(metrics)
        
        # 详细的时间测量
        time_metrics = MLMetrics.measure_inference_time(model, X_test, n_runs=kwargs.get('n_runs', 50))
        results.update(time_metrics)
        
        # 性能总结
        results['meets_performance_requirements'] = MLMetrics._check_performance_requirements(
            results, task_type
        )
        
        return results
    
    @staticmethod
    def generate_report(metrics: Dict[str, Any], 
                       task_type: str = 'classification') -> str:
        """
        生成性能报告
        
        Args:
            metrics: 指标字典
            task_type: 任务类型
            
        Returns:
            报告字符串
        """
        report_lines = []
        
        report_lines.append("=" * 60)
        report_lines.append(f"机器学习模型性能报告 - {task_type.upper()}")
        report_lines.append("=" * 60)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 主要指标
        report_lines.append("主要指标:")
        report_lines.append("-" * 40)
        
        if task_type == 'classification':
            main_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        elif task_type == 'regression':
            main_metrics = ['r2_score', 'mse', 'rmse', 'mae']
        elif task_type == 'clustering':
            main_metrics = ['silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score']
        elif task_type == 'anomaly':
            main_metrics = ['accuracy', 'precision_anomaly', 'detection_rate', 'false_alarm_rate']
        else:
            main_metrics = []
        
        for metric in main_metrics:
            if metric in metrics:
                report_lines.append(f"  {metric}: {metrics[metric]:.4f}")
        
        # 推理时间
        report_lines.append("")
        report_lines.append("推理性能:")
        report_lines.append("-" * 40)
        time_metrics = ['inference_time_ms_mean', 'inference_time_ms_p95', 'throughput_samples_per_second']
        for metric in time_metrics:
            if metric in metrics:
                if 'throughput' in metric:
                    report_lines.append(f"  {metric}: {metrics[metric]:.2f}")
                else:
                    report_lines.append(f"  {metric}: {metrics[metric]:.2f} ms")
        
        # 性能要求检查
        report_lines.append("")
        report_lines.append("性能要求检查:")
        report_lines.append("-" * 40)
        if 'meets_performance_requirements' in metrics:
            status = "通过" if metrics['meets_performance_requirements'] else "未通过"
            report_lines.append(f"  整体状态: {status}")
        
        # 详细指标
        report_lines.append("")
        report_lines.append("详细指标:")
        report_lines.append("-" * 40)
        
        # 按类别组织指标
        metric_categories = {
            '基础指标': ['accuracy', 'precision', 'recall', 'f1_score', 'r2_score', 'mse', 'rmse'],
            '时间指标': ['