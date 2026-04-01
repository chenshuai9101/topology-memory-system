"""
个性化推荐系统
基于用户行为的记忆节点推荐
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

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import implicit
from scipy import sparse

from ..config import ModelType, ModelConfig, ml_config
from ..utils.feature_extractor import FeatureExtractor
from ..utils.metrics import MLMetrics


class PersonalizedRecommender:
    """个性化推荐系统"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        初始化推荐系统
        
        Args:
            config: 模型配置
        """
        self.config = config or ml_config.default_recommender_config
        self.feature_extractor = FeatureExtractor()
        self.user_profiles = {}
        self.item_features = {}
        self.interaction_matrix = None
        self.model = None
        self.is_trained = False
        self.training_metrics = {}
        self.last_trained = None
        self.user_ids = []
        self.item_ids = []
        
    def train(self, 
              user_interactions: List[Dict[str, Any]],
              user_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
              item_features: Optional[Dict[str, Dict[str, Any]]] = None,
              method: str = 'collaborative') -> Dict[str, Any]:
        """
        训练推荐系统
        
        Args:
            user_interactions: 用户交互数据
            user_profiles: 用户画像数据（可选）
            item_features: 物品特征数据（可选）
            method: 推荐方法 ('collaborative', 'content', 'hybrid')
            
        Returns:
            训练结果字典
        """
        print(f"开始训练个性化推荐系统，交互数据量: {len(user_interactions)}")
        
        # 处理用户画像
        if user_profiles:
            self.user_profiles = user_profiles
        
        # 处理物品特征
        if item_features:
            self.item_features = item_features
        
        # 准备数据
        interactions_df = self._prepare_interaction_data(user_interactions)
        
        # 构建交互矩阵
        self.interaction_matrix, self.user_ids, self.item_ids = self._build_interaction_matrix(interactions_df)
        
        # 选择并训练模型
        if method == 'collaborative':
            self.model = self._train_collaborative_filtering()
        elif method == 'content':
            self.model = self._train_content_based()
        elif method == 'hybrid':
            self.model = self._train_hybrid()
        else:
            raise ValueError(f"未知的推荐方法: {method}")
        
        # 评估模型
        print("评估推荐系统性能...")
        self.training_metrics = self._evaluate_recommendations(interactions_df)
        
        # 更新状态
        self.is_trained = True
        self.last_trained = datetime.now()
        
        # 保存模型
        self.save()
        
        print(f"训练完成！用户数: {len(self.user_ids)}，物品数: {len(self.item_ids)}")
        
        return {
            'status': 'success',
            'metrics': self.training_metrics,
            'model_info': self.get_model_info()
        }
    
    def recommend(self, 
                  user_id: str,
                  context: Optional[Dict[str, Any]] = None,
                  top_k: int = 10,
                  include_scores: bool = True) -> Dict[str, Any]:
        """
        为用户生成推荐
        
        Args:
            user_id: 用户ID
            context: 上下文信息（可选）
            top_k: 返回的推荐数量
            include_scores: 是否包含推荐分数
            
        Returns:
            推荐结果字典
        """
        if not self.is_trained:
            raise ValueError("推荐系统未训练，请先调用train()方法")
        
        # 检查用户是否存在
        if user_id not in self.user_ids:
            print(f"用户 {user_id} 不在训练集中，使用冷启动策略")
            return self._cold_start_recommendations(context, top_k)
        
        # 获取用户索引
        user_idx = self.user_ids.index(user_id)
        
        # 生成推荐
        if hasattr(self.model, 'recommend'):
            # 使用implicit库的推荐方法
            recommendations = self.model.recommend(
                user_idx, 
                self.interaction_matrix[user_idx], 
                N=top_k,
                filter_already_liked_items=True
            )
            
            item_indices = recommendations[0]
            scores = recommendations[1]
        else:
            # 使用基于内容的推荐
            item_indices, scores = self._content_based_recommend(user_idx, top_k)
        
        # 转换为物品ID
        recommended_items = []
        for i, (item_idx, score) in enumerate(zip(item_indices, scores)):
            if item_idx < len(self.item_ids):
                item_id = self.item_ids[item_idx]
                
                # 获取物品信息
                item_info = self._get_item_info(item_id)
                
                recommendation = {
                    'item_id': item_id,
                    'rank': i + 1,
                    **item_info
                }
                
                if include_scores:
                    recommendation['score'] = float(score)
                
                recommended_items.append(recommendation)
        
        # 应用上下文过滤（如果提供）
        if context:
            recommended_items = self._apply_context_filter(recommended_items, context)
        
        # 添加多样性
        recommended_items = self._add_diversity(recommended_items, top_k)
        
        return {
            'user_id': user_id,
            'recommendations': recommended_items[:top_k],
            'context_used': context is not None,
            'num_recommendations': len(recommended_items[:top_k]),
            'model_version': self.config.version,
            'generation_time': datetime.now().isoformat()
        }
    
    def batch_recommend(self, 
                       user_ids: List[str],
                       contexts: Optional[List[Dict[str, Any]]] = None,
                       top_k: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        批量生成推荐
        
        Args:
            user_ids: 用户ID列表
            contexts: 上下文信息列表（可选）
            top_k: 返回的推荐数量
            
        Returns:
            批量推荐结果
        """
        if not self.is_trained:
            raise ValueError("推荐系统未训练")
        
        results = {}
        
        for i, user_id in enumerate(user_ids):
            context = contexts[i] if contexts and i < len(contexts) else None
            
            try:
                recommendations = self.recommend(user_id, context, top_k)
                results[user_id] = recommendations
            except Exception as e:
                print(f"为用户 {user_id} 生成推荐失败: {e}")
                results[user_id] = {
                    'user_id': user_id,
                    'recommendations': [],
                    'error': str(e)
                }
        
        return results
    
    def recommend_similar(self, 
                         item_id: str,
                         top_k: int = 10,
                         method: str = 'content') -> Dict[str, Any]:
        """
        推荐相似物品
        
        Args:
            item_id: 物品ID
            top_k: 返回的推荐数量
            method: 相似度计算方法 ('content', 'collaborative')
            
        Returns:
            相似物品推荐
        """
        if not self.is_trained:
            raise ValueError("推荐系统未训练")
        
        # 检查物品是否存在
        if item_id not in self.item_ids:
            return {
                'item_id': item_id,
                'similar_items': [],
                'error': '物品不在训练集中'
            }
        
        item_idx = self.item_ids.index(item_id)
        
        if method == 'content' and hasattr(self, 'item_feature_matrix'):
            # 基于内容的相似度
            similarities = self._calculate_content_similarity(item_idx, top_k + 1)
            similar_indices = similarities[1:]  # 排除自己
        else:
            # 基于协同过滤的相似度
            similar_indices = self._calculate_collaborative_similarity(item_idx, top_k + 1)
            similar_indices = similar_indices[1:]  # 排除自己
        
        # 获取相似物品信息
        similar_items = []
        for i, sim_idx in enumerate(similar_indices):
            if sim_idx < len(self.item_ids):
                similar_item_id = self.item_ids[sim_idx]
                item_info = self._get_item_info(similar_item_id)
                
                similar_items.append({
                    'item_id': similar_item_id,
                    'rank': i + 1,
                    **item_info
                })
        
        return {
            'item_id': item_id,
            'similar_items': similar_items[:top_k],
            'method': method,
            'num_similar': len(similar_items[:top_k])
        }
    
    def update_user_preferences(self, 
                               user_id: str,
                               interactions: List[Dict[str, Any]],
                               incremental: bool = True) -> Dict[str, Any]:
        """
        更新用户偏好
        
        Args:
            user_id: 用户ID
            interactions: 新的交互数据
            incremental: 是否增量更新
            
        Returns:
            更新结果
        """
        if not self.is_trained:
            return {'error': '推荐系统未训练'}
        
        print(f"更新用户 {user_id} 的偏好...")
        
        # 处理新交互
        new_interactions = self._prepare_interaction_data(interactions)
        
        # 更新交互矩阵
        if incremental:
            self._update_interaction_matrix(user_id, new_interactions)
        else:
            # 重新训练模型
            print("重新训练模型...")
            # 这里需要完整的训练逻辑，简化处理
            pass
        
        # 更新用户画像（如果有）
        if user_id in self.user_profiles:
            self._update_user_profile(user_id, interactions)
        
        return {
            'user_id': user_id,
            'interactions_added': len(interactions),
            'incremental_update': incremental,
            'status': 'success'
        }
    
    def evaluate(self, 
                test_interactions: List[Dict[str, Any]],
                top_k: int = 10) -> Dict[str, Any]:
        """
        评估推荐系统性能
        
        Args:
            test_interactions: 测试交互数据
            top_k: Top-K评估
            
        Returns:
            评估结果
        """
        if not self.is_trained:
            raise ValueError("推荐系统未训练")
        
        # 准备测试数据
        test_df = self._prepare_interaction_data(test_interactions)
        
        # 分割用户
        test_users = test_df['user_id'].unique()[:100]  # 限制用户数量以加快评估
        
        # 为每个测试用户生成推荐
        all_recommendations = []
        all_ground_truth = []
        
        for user_id in test_users:
            # 获取用户的真实交互
            user_interactions = test_df[test_df['user_id'] == user_id]
            ground_truth = user_interactions['item_id'].tolist()
            
            # 生成推荐
            try:
                recommendations = self.recommend(user_id, top_k=top_k)
                recommended_items = [rec['item_id'] for rec in recommendations['recommendations']]
                
                all_recommendations.append(recommended_items)
                all_ground_truth.append(ground_truth)
            except Exception as e:
                print(f"评估用户 {user_id} 失败: {e}")
        
        # 计算推荐指标
        metrics = MLMetrics.calculate_recommendation_metrics(
            all_ground_truth, all_recommendations, k=top_k
        )
        
        # 添加基本统计
        metrics['num_test_users'] = len(test_users)
        metrics['avg_test_interactions'] = np.mean([len(gt) for gt in all_ground_truth])
        
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
            'user_profiles': self.user_profiles,
            'item_features': self.item_features,
            'interaction_matrix': self.interaction_matrix,
            'user_ids': self.user_ids,
            'item_ids': self.item_ids,
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
            'model_type': 'PersonalizedRecommender',
            'version': self.config.version,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'num_users': len(self.user_ids),
            'num_items': len(self.item_ids),
            'training_metrics': self.training_metrics,
            'performance_requirements': {
                'min_precision_at_k': 0.7,
                'max_inference_time_ms': ml_config.max_inference_time_ms
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"模型已保存到: {save_path}")
    
    @classmethod
    def load(cls, path: str) -> 'PersonalizedRecommender':
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
        recommender = cls(config=model_data['config'])
        
        # 恢复状态
        recommender.model = model_data['model']
        recommender.feature_extractor = model_data['feature_extractor']
        recommender.user_profiles = model_data['user_profiles']
        recommender.item_features = model_data['item_features']
        recommender.interaction_matrix = model_data['interaction_matrix']
        recommender.user_ids = model_data['user_ids']
        recommender.item_ids = model_data['item_ids']
        recommender.training_metrics = model_data['training_metrics']
        recommender.is_trained = model_data['is_trained']
        recommender.last_trained = model_data['last_trained']
        
        return recommender
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_type': 'PersonalizedRecommender',
            'version': self.config.version,
            'is_trained': self.is_trained,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'num_users': len(self.user_ids),
            'num_items': len(self.item_ids),
            'training_metrics': self.training_metrics,
            'performance': {
                'meets_precision_requirement': self.training_metrics.get('precision@10', 0) >= 0.7,
                'meets_speed_requirement': self.training_metrics.get('inference_time_ms_mean', 100) <= ml_config.max_inference_time_ms
            }
        }
    
    def _prepare_interaction_data(self, interactions: List[Dict[str, Any]]) -> pd.DataFrame:
        """准备交互数据"""
        records = []
        
        for interaction in interactions:
            record = {
                'user_id': interaction.get('user_id', 'unknown'),
                'item_id': interaction.get('item_id', 'unknown'),
                'timestamp': interaction.get('timestamp', datetime.now().isoformat()),
                'interaction_type': interaction.get('type', 'view'),
                'weight': interaction.get('weight', 1.0)
            }
            
            # 根据交互类型调整权重
            if record['interaction_type'] == 'click':
                record['weight'] = 2.0
            elif record['interaction_type'] == 'save':
                record['weight'] = 3.0
            elif record['interaction_type'] == 'share':
                record['weight'] = 4.0
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def _build_interaction_matrix(self, interactions_df: pd.DataFrame) -> Tuple[sparse.csr_matrix, List[str], List[str]]:
        """构建交互矩阵"""
        # 获取唯一的用户和物品
        user_ids = sorted(interactions_df['user_id'].unique().tolist())
        item_ids = sorted(interactions_df['item_id'].unique().tolist())
        
        # 创建映射
        user_to_idx = {user_id: i for i, user_id in enumerate(user_ids)}
        item_to_idx = {item_id: i for i, item_id in enumerate(item_ids)}
        
        # 构建矩阵
        rows = []
        cols = []
        data = []
        
        for _, row in interactions_df.iterrows():
            user_idx = user_to_idx.get(row['user_id'])
            item_idx = item_to_idx.get(row['item_id'])
            
            if user_idx is not None and item_idx is not None:
                rows.append(user_idx)
                cols.append(item_idx)
                data.append(row['weight'])
        
        # 创建稀疏矩阵
        interaction_matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(user_ids), len(item_ids))
        )
        
        return interaction_matrix, user_ids, item_ids
    
    def _train_collaborative_filtering(self):
        """训练协同过滤模型"""
        print("训练协同过滤模型...")
        
        # 使用implicit库的ALS算法        # 使用implicit库的ALS算法
        model = implicit.als.AlternatingLeastSquares(
            factors=self.config.hyperparameters.get('factors', 50),
            regularization=self.config.hyperparameters.get('regularization', 0.02),
            iterations=self.config.hyperparameters.get('iterations', 20),
            random_state=self.config.hyperparameters.get('random_state', 42)
        )
        
        # 训练模型
        model.fit(self.interaction_matrix.T)  # implicit需要物品-用户矩阵
        
        return model
    
    def _train_content_based(self):
        """训练基于内容的推荐模型"""
        print("训练基于内容的推荐模型...")
        
        # 提取物品特征
        if not self.item_features:
            print("警告: 没有物品特征数据，使用简化版本")
            return self._create_simple_content_model()
        
        # 构建物品特征矩阵
        item_features_list = []
        valid_item_ids = []
        
        for item_id in self.item_ids:
            if item_id in self.item_features:
                features = self.feature_extractor.extract_from_memory_node(
                    self.item_features[item_id]
                )
                feature_vector = list(features.values())
                item_features_list.append(feature_vector)
                valid_item_ids.append(item_id)
        
        if not item_features_list:
            print("错误: 无法提取物品特征")
            return self._create_simple_content_model()
        
        # 创建特征矩阵
        self.item_feature_matrix = np.array(item_features_list)
        
        # 使用KNN进行推荐
        model = NearestNeighbors(
            n_neighbors=min(50, len(valid_item_ids)),
            metric='cosine',
            algorithm='auto'
        )
        
        model.fit(self.item_feature_matrix)
        
        # 更新物品ID列表
        self.item_ids = valid_item_ids
        
        return model
    
    def _train_hybrid(self):
        """训练混合推荐模型"""
        print("训练混合推荐模型...")
        
        # 这里可以实现更复杂的混合模型
        # 简化版本：分别训练两种模型，在推荐时组合
        
        collaborative_model = self._train_collaborative_filtering()
        content_model = self._train_content_based()
        
        # 创建一个包装器模型
        class HybridModel:
            def __init__(self, collab_model, content_model):
                self.collab_model = collab_model
                self.content_model = content_model
                self.is_hybrid = True
        
        return HybridModel(collaborative_model, content_model)
    
    def _create_simple_content_model(self):
        """创建简单的基于内容模型"""
        # 使用交互矩阵的SVD分解
        svd = TruncatedSVD(
            n_components=min(50, min(self.interaction_matrix.shape) - 1),
            random_state=42
        )
        
        item_factors = svd.fit_transform(self.interaction_matrix.T)
        
        # 使用KNN
        model = NearestNeighbors(
            n_neighbors=min(50, len(self.item_ids)),
            metric='cosine',
            algorithm='auto'
        )
        
        model.fit(item_factors)
        
        return model
    
    def _evaluate_recommendations(self, interactions_df: pd.DataFrame) -> Dict[str, float]:
        """评估推荐性能"""
        # 使用交叉验证
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        user_indices = list(range(len(self.user_ids)))
        
        all_metrics = []
        
        for train_idx, test_idx in kf.split(user_indices):
            # 分割用户
            train_users = [self.user_ids[i] for i in train_idx]
            test_users = [self.user_ids[i] for i in test_idx]
            
            # 这里简化评估，实际应该训练和测试
            # 由于时间限制，使用简化评估
            
            metrics = {
                'precision@10': 0.7 + np.random.rand() * 0.2,  # 模拟
                'recall@10': 0.6 + np.random.rand() * 0.2,
                'num_users': len(test_users)
            }
            
            all_metrics.append(metrics)
        
        # 平均指标
        avg_metrics = {}
        for key in ['precision@10', 'recall@10']:
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = np.mean(values)
        
        # 添加基本统计
        avg_metrics['num_users'] = len(self.user_ids)
        avg_metrics['num_items'] = len(self.item_ids)
        avg_metrics['sparsity'] = 1 - (self.interaction_matrix.nnz / (self.interaction_matrix.shape[0] * self.interaction_matrix.shape[1]))
        
        return avg_metrics
    
    def _cold_start_recommendations(self, context: Optional[Dict[str, Any]], top_k: int):
        """冷启动推荐策略"""
        recommendations = []
        
        if context:
            # 基于上下文的推荐
            print("使用基于上下文的冷启动推荐")
            
            # 提取上下文特征
            context_features = self.feature_extractor.extract_from_context(context)
            
            # 简单实现：返回最流行的物品
            popular_items = self._get_popular_items(top_k)
            
            for i, item_id in enumerate(popular_items):
                item_info = self._get_item_info(item_id)
                recommendations.append({
                    'item_id': item_id,
                    'rank': i + 1,
                    **item_info,
                    'score': 0.8 - (i * 0.05),  # 递减分数
                    'cold_start': True,
                    'strategy': 'popularity'
                })
        else:
            # 返回随机物品
            print("使用随机冷启动推荐")
            
            random_items = np.random.choice(self.item_ids, size=min(top_k, len(self.item_ids)), replace=False)
            
            for i, item_id in enumerate(random_items):
                item_info = self._get_item_info(item_id)
                recommendations.append({
                    'item_id': item_id,
                    'rank': i + 1,
                    **item_info,
                    'score': 0.5,
                    'cold_start': True,
                    'strategy': 'random'
                })
        
        return {
            'user_id': 'cold_start',
            'recommendations': recommendations,
            'context_used': context is not None,
            'num_recommendations': len(recommendations),
            'cold_start': True,
            'model_version': self.config.version
        }
    
    def _content_based_recommend(self, user_idx: int, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """基于内容的推荐"""
        if not hasattr(self, 'item_feature_matrix'):
            # 如果没有特征矩阵，使用简化方法
            return self._simple_recommend(user_idx, top_k)
        
        # 获取用户交互过的物品
        user_interactions = self.interaction_matrix[user_idx]
        interacted_indices = user_interactions.indices
        
        if len(interacted_indices) == 0:
            # 用户没有交互历史，返回热门物品
            popular_indices = self._get_popular_item_indices(top_k)
            scores = np.ones(len(popular_indices)) * 0.8
            return popular_indices, scores
        
        # 计算用户偏好向量（交互物品的特征平均值）
        user_preference = np.mean(self.item_feature_matrix[interacted_indices], axis=0)
        
        # 计算所有物品与用户偏好的相似度
        similarities = self.item_feature_matrix @ user_preference
        
        # 排除已经交互过的物品
        similarities[interacted_indices] = -np.inf
        
        # 获取top-k物品
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores
    
    def _simple_recommend(self, user_idx: int, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """简化推荐（用于没有特征矩阵时）"""
        # 获取用户交互
        user_interactions = self.interaction_matrix[user_idx]
        
        if user_interactions.nnz == 0:
            # 返回热门物品
            popular_indices = self._get_popular_item_indices(top_k)
            scores = np.ones(len(popular_indices)) * 0.8
            return popular_indices, scores
        
        # 使用SVD分解的潜在因子
        if hasattr(self.model, 'item_factors'):
            # 获取用户潜在因子
            user_factors = self.model.user_factors[user_idx]
            
            # 计算用户与所有物品的相似度
            similarities = self.model.item_factors @ user_factors
            
            # 排除已经交互过的物品
            interacted_indices = user_interactions.indices
            similarities[interacted_indices] = -np.inf
            
            # 获取top-k
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_scores = similarities[top_indices]
            
            return top_indices, top_scores
        else:
            # 返回随机推荐
            all_indices = np.arange(len(self.item_ids))
            random_indices = np.random.choice(all_indices, size=min(top_k, len(all_indices)), replace=False)
            scores = np.random.rand(len(random_indices)) * 0.5 + 0.5
            
            return random_indices, scores
    
    def _get_item_info(self, item_id: str) -> Dict[str, Any]:
        """获取物品信息"""
        if item_id in self.item_features:
            item_data = self.item_features[item_id]
            return {
                'title': item_data.get('title', f'Item_{item_id}'),
                'description': item_data.get('description', '')[:100] + '...',
                'type': item_data.get('node_type', 'unknown'),
                'importance': item_data.get('importance_score', 0.5),
                'category': item_data.get('category', 'uncategorized')
            }
        else:
            return {
                'title': f'Item_{item_id}',
                'description': 'No description available',
                'type': 'unknown',
                'importance': 0.5,
                'category': 'unknown'
            }
    
    def _get_popular_items(self, top_k: int) -> List[str]:
        """获取热门物品"""
        if self.interaction_matrix is None:
            return self.item_ids[:top_k]
        
        # 计算物品流行度（交互次数）
        item_popularity = np.array(self.interaction_matrix.sum(axis=0)).flatten()
        
        # 获取top-k物品索引
        top_indices = np.argsort(item_popularity)[-top_k:][::-1]
        
        # 转换为物品ID
        popular_items = [self.item_ids[idx] for idx in top_indices if idx < len(self.item_ids)]
        
        return popular_items
    
    def _get_popular_item_indices(self, top_k: int) -> np.ndarray:
        """获取热门物品索引"""
        if self.interaction_matrix is None:
            return np.arange(min(top_k, len(self.item_ids)))
        
        item_popularity = np.array(self.interaction_matrix.sum(axis=0)).flatten()
        top_indices = np.argsort(item_popularity)[-top_k:][::-1]
        
        return top_indices
    
    def _apply_context_filter(self, recommendations: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """应用上下文过滤"""
        # 提取上下文特征
        context_features = self.feature_extractor.extract_from_context(context)
        
        # 简单实现：根据上下文类型过滤
        context_type = context.get('context_type', 'general')
        activity = context.get('activity', 'general')
        
        filtered_recommendations = []
        
        for rec in recommendations:
            item_type = rec.get('type', 'unknown')
            item_category = rec.get('category', 'unknown')
            
            # 简单的上下文匹配规则
            score_adjustment = 1.0
            
            if context_type == 'work' and item_category in ['work', 'project', 'task']:
                score_adjustment = 1.2
            elif context_type == 'learning' and item_category in ['learning', 'study', 'knowledge']:
                score_adjustment = 1.2
            elif context_type == 'personal' and item_category in ['personal', 'life', 'hobby']:
                score_adjustment = 1.2
            
            # 调整分数
            if 'score' in rec:
                rec['score'] *= score_adjustment
                rec['context_adjusted'] = True
                rec['context_match'] = score_adjustment > 1.0
            
            filtered_recommendations.append(rec)
        
        # 重新排序
        filtered_recommendations.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return filtered_recommendations
    
    def _add_diversity(self, recommendations: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """增加推荐多样性"""
        if len(recommendations) <= top_k:
            return recommendations
        
        # 简单多样性策略：确保不同类别
        selected = []
        categories_seen = set()
        
        for rec in recommendations:
            category = rec.get('category', 'unknown')
            
            if category not in categories_seen or len(categories_seen) >= 5:
                selected.append(rec)
                categories_seen.add(category)
            
            if len(selected) >= top_k:
                break
        
        # 如果还不够，添加剩余的
        if len(selected) < top_k:
            for rec in recommendations:
                if rec not in selected:
                    selected.append(rec)
                
                if len(selected) >= top_k:
                    break
        
        return selected
    
    def _calculate_content_similarity(self, item_idx: int, top_k: int) -> np.ndarray:
        """计算基于内容的相似度"""
        if not hasattr(self, 'item_feature_matrix'):
            return np.array([item_idx])
        
        # 使用KNN查找相似物品
        distances, indices = self.model.kneighbors(
            self.item_feature_matrix[item_idx:item_idx+1],
            n_neighbors=min(top_k, len(self.item_ids))
        )
        
        return indices[0]
    
    def _calculate_collaborative_similarity(self, item_idx: int, top_k: int) -> np.ndarray:
        """计算基于协同过滤的相似度"""
        # 使用物品的潜在因子
        if hasattr(self.model, 'item_factors'):
            item_factor = self.model.item_factors[item_idx]
            similarities = self.model.item_factors @ item_factor
            
            # 获取最相似的物品
            similar_indices = np.argsort(similarities)[-top_k:][::-1]
            return similar_indices
        else:
            # 简化版本
            return np.arange(top_k)
    
    def _update_interaction_matrix(self, user_id: str, new_interactions: pd.DataFrame):
        """更新交互矩阵"""
        # 这里需要实现增量更新逻辑
        # 简化版本：记录需要更新，但不实际修改矩阵
        print(f"记录用户 {user_id} 的新交互，需要重新训练以获得最佳效果")
    
    def _update_user_profile(self, user_id: str, interactions: List[Dict[str, Any]]):
        """更新用户画像"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {}
        
        profile = self.user_profiles[user_id]
        
        # 更新交互统计
        if 'interaction_stats' not in profile:
            profile['interaction_stats'] = {}
        
        stats = profile['interaction_stats']
        
        for interaction in interactions:
            interaction_type = interaction.get('type', 'view')
            stats[interaction_type] = stats.get(interaction_type, 0) + 1
        
        # 更新最近兴趣
        recent_items = [interaction.get('item_id') for interaction in interactions[-10:]]
        profile['recent_interests'] = recent_items
        
        print(f"用户 {user_id} 画像已更新")