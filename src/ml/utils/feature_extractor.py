"""
特征提取器
用于从记忆节点和上下文中提取机器学习特征
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import hashlib
import json


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, config=None):
        """
        初始化特征提取器
        
        Args:
            config: ML配置对象
        """
        from ..config import ml_config
        self.config = config or ml_config
        
        # 初始化文本向量化器
        self.text_vectorizer = TfidfVectorizer(
            max_features=self.config.text_embedding_dim,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # 初始化标准化器
        self.scaler = StandardScaler()
        
        # 初始化标签编码器
        self.label_encoders = {}
        
        # 特征缓存
        self.feature_cache = {}
        
    def extract_from_memory_node(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从记忆节点提取特征
        
        Args:
            node_data: 记忆节点数据
            
        Returns:
            特征字典
        """
        features = {}
        
        # 基础数值特征
        features['access_count'] = node_data.get('access_count', 0)
        features['importance_score'] = node_data.get('importance_score', 0.5)
        features['recency_days'] = self._calculate_recency(node_data.get('last_accessed'))
        
        # 节点类型特征
        node_type = node_data.get('node_type', 'unknown')
        features['node_type'] = node_type
        
        # 分类特征
        category = node_data.get('category', 'uncategorized')
        features['category'] = category
        
        # 文本特征
        content = node_data.get('content', '')
        tags = node_data.get('tags', [])
        
        # 文本统计特征
        features['content_length'] = len(content)
        features['word_count'] = len(content.split())
        features['tag_count'] = len(tags)
        features['has_links'] = 1 if 'http' in content else 0
        features['has_images'] = 1 if any(img in content.lower() for img in ['.jpg', '.png', '.gif']) else 0
        
        # 情感特征（简化版）
        features['positive_words'] = self._count_sentiment_words(content, 'positive')
        features['negative_words'] = self._count_sentiment_words(content, 'negative')
        
        # 时间特征
        created_at = node_data.get('created_at')
        if created_at:
            features['age_days'] = self._calculate_age(created_at)
            features['created_hour'] = self._extract_hour(created_at)
            features['created_weekday'] = self._extract_weekday(created_at)
        else:
            features['age_days'] = 0
            features['created_hour'] = 12
            features['created_weekday'] = 0
        
        # 关联特征
        associations = node_data.get('associations', [])
        features['association_count'] = len(associations)
        features['strong_associations'] = sum(1 for a in associations if a.get('strength', 0) > 0.7)
        
        # 上下文特征
        contexts = node_data.get('contexts', [])
        features['context_count'] = len(contexts)
        
        # 向量特征（如果可用）
        if 'embedding' in node_data:
            embedding = node_data['embedding']
            if isinstance(embedding, list) and len(embedding) > 0:
                # 取前几个维度作为特征
                for i in range(min(5, len(embedding))):
                    features[f'embedding_{i}'] = embedding[i]
        
        return features
    
    def extract_from_context(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从上下文提取特征
        
        Args:
            context_data: 上下文数据
            
        Returns:
            特征字典
        """
        features = {}
        
        # 基础特征
        features['context_type'] = context_data.get('context_type', 'unknown')
        features['relevance_score'] = context_data.get('relevance_score', 0.5)
        
        # 时间特征
        timestamp = context_data.get('timestamp')
        if timestamp:
            features['timestamp_hour'] = self._extract_hour(timestamp)
            features['timestamp_weekday'] = self._extract_weekday(timestamp)
        
        # 位置特征
        location = context_data.get('location', {})
        features['has_location'] = 1 if location else 0
        
        # 设备特征
        device = context_data.get('device', {})
        features['device_type'] = device.get('type', 'unknown')
        
        # 活动特征
        activity = context_data.get('activity', 'unknown')
        features['activity'] = activity
        
        # 情感特征
        sentiment = context_data.get('sentiment', 0)
        features['sentiment'] = sentiment
        
        # 注意力特征
        attention_level = context_data.get('attention_level', 0.5)
        features['attention_level'] = attention_level
        
        return features
    
    def extract_combined_features(self, 
                                 node_data: Dict[str, Any], 
                                 context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取节点和上下文的组合特征
        
        Args:
            node_data: 记忆节点数据
            context_data: 上下文数据
            
        Returns:
            组合特征字典
        """
        node_features = self.extract_from_memory_node(node_data)
        context_features = self.extract_from_context(context_data)
        
        # 合并特征
        combined_features = {**node_features, **context_features}
        
        # 添加交互特征
        combined_features['node_context_similarity'] = self._calculate_similarity(
            node_features, context_features
        )
        
        # 添加时间对齐特征
        node_time = node_data.get('last_accessed')
        context_time = context_data.get('timestamp')
        if node_time and context_time:
            time_diff = abs(self._parse_datetime(node_time) - self._parse_datetime(context_time))
            combined_features['time_alignment_hours'] = time_diff.total_seconds() / 3600
        
        return combined_features
    
    def batch_extract(self, data_list: List[Dict[str, Any]], 
                     feature_type: str = 'node') -> pd.DataFrame:
        """
        批量提取特征
        
        Args:
            data_list: 数据列表
            feature_type: 特征类型 ('node', 'context', 'combined')
            
        Returns:
            特征DataFrame
        """
        features_list = []
        
        for data in data_list:
            if feature_type == 'node':
                features = self.extract_from_memory_node(data)
            elif feature_type == 'context':
                features = self.extract_from_context(data)
            elif feature_type == 'combined':
                # 假设数据是(node, context)元组
                if isinstance(data, tuple) and len(data) == 2:
                    node_data, context_data = data
                    features = self.extract_combined_features(node_data, context_data)
                else:
                    continue
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def prepare_training_data(self, 
                             X_raw: List[Dict[str, Any]], 
                             y: Optional[List] = None,
                             feature_type: str = 'node') -> tuple:
        """
        准备训练数据
        
        Args:
            X_raw: 原始特征数据
            y: 标签数据（可选）
            feature_type: 特征类型
            
        Returns:
            (X_processed, y_processed) 或 X_processed
        """
        # 提取特征
        X_df = self.batch_extract(X_raw, feature_type)
        
        # 处理分类特征
        X_processed = self._encode_categorical_features(X_df)
        
        # 标准化数值特征
        X_processed = self._scale_numerical_features(X_processed)
        
        if y is not None:
            return X_processed, np.array(y)
        else:
            return X_processed
    
    def _calculate_recency(self, timestamp: Optional[str]) -> float:
        """计算最近性（天数）"""
        if not timestamp:
            return 365.0  # 默认一年前
        
        try:
            dt = self._parse_datetime(timestamp)
            now = datetime.now()
            delta = now - dt
            return delta.days
        except:
            return 365.0
    
    def _calculate_age(self, timestamp: str) -> float:
        """计算年龄（天数）"""
        try:
            dt = self._parse_datetime(timestamp)
            now = datetime.now()
            delta = now - dt
            return delta.days
        except:
            return 0.0
    
    def _extract_hour(self, timestamp: str) -> int:
        """提取小时"""
        try:
            dt = self._parse_datetime(timestamp)
            return dt.hour
        except:
            return 12
    
    def _extract_weekday(self, timestamp: str) -> int:
        """提取星期几（0=周一，6=周日）"""
        try:
            dt = self._parse_datetime(timestamp)
            return dt.weekday()
        except:
            return 0
    
    def _parse_datetime(self, timestamp: str) -> datetime:
        """解析时间戳"""
        if isinstance(timestamp, datetime):
            return timestamp
        
        # 尝试常见格式
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp, fmt)
            except:
                continue
        
        # 如果所有格式都失败，返回当前时间
        return datetime.now()
    
    def _count_sentiment_words(self, text: str, sentiment: str) -> int:
        """统计情感词数量（简化版）"""
        positive_words = {'good', 'great', 'excellent', 'happy', 'love', 'like', 'best'}
        negative_words = {'bad', 'poor', 'terrible', 'sad', 'hate', 'dislike', 'worst'}
        
        words = set(text.lower().split())
        
        if sentiment == 'positive':
            return len(words.intersection(positive_words))
        else:  # negative
            return len(words.intersection(negative_words))
    
    def _calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        """计算特征相似度（简化版）"""
        common_keys = set(features1.keys()).intersection(set(features2.keys()))
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1 = features1[key]
            val2 = features2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # 数值特征：使用相对差异
                if val1 == 0 and val2 == 0:
                    sim = 1.0
                else:
                    diff = abs(val1 - val2)
                    max_val = max(abs(val1), abs(val2))
                    sim = 1.0 - (diff / (max_val + 1e-10))
                similarities.append(sim)
            elif isinstance(val1, str) and isinstance(val2, str):
                # 文本特征：使用精确匹配
                sim = 1.0 if val1 == val2 else 0.0
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """编码分类特征"""
        df_encoded = df.copy()
        
        for col in self.config.categorical_features:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    # 处理NaN值
                    non_null_values = df_encoded[col].fillna('unknown').astype(str)
                    self.label_encoders[col].fit(non_null_values)
                
                # 编码
                encoded = df_encoded[col].fillna('unknown').astype(str)
                df_encoded[col] = self.label_encoders[col].transform(encoded)
        
        return df_encoded
    
    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化数值特征"""
        df_scaled = df.copy()
        
        numerical_cols = [col for col in self.config.numerical_features 
                         if col in df_scaled.columns]
        
        if numerical_cols:
            # 只拟合一次
            if not hasattr(self.scaler, 'fitted_'):
                self.scaler.fit(df_scaled[numerical_cols])
                self.scaler.fitted_ = True
            
            df_scaled[numerical_cols] = self.scaler.transform(df_scaled[numerical_cols])
        
        return df_scaled
    
    def get_feature_names(self, feature_type: str = 'node') -> List[str]:
        """
        获取特征名称
        
        Args:
            feature_type: 特征类型
            
        Returns:
            特征名称列表
        """
        # 这里可以根据实际特征生成
        if feature_type == 'node':
            return [
                'access_count', 'importance_score', 'recency_days',
                'node_type', 'category', 'content_length', 'word_count',
                'tag_count', 'has_links', 'has_images', 'positive_words',
                'negative_words', 'age_days', 'created_hour', 'created_weekday',
                'association_count', 'strong_associations', 'context_count'
            ]
        elif feature_type == 'context':
            return [
                'context_type', 'relevance_score', 'timestamp_hour',
                'timestamp_weekday', 'has_location', 'device_type',
                'activity', 'sentiment', 'attention_level'
            ]
        else:  # combined
            node_features = self.get_feature_names('node')
            context_features = self.get_feature_names('context')
            return node_features + context_features + ['node_context_similarity', 'time_alignment_hours']