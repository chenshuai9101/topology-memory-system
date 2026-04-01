"""
记忆优化器 - 提升记忆质量和清晰度
负责记忆压缩、关联优化、质量评估
"""

import re
import time
import math
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
import logging
from collections import defaultdict, Counter
import heapq

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)


@dataclass
class MemoryQualityMetrics:
    """记忆质量指标"""
    clarity_score: float  # 清晰度 (0-1)
    relevance_score: float  # 相关性 (0-1)
    completeness_score: float  # 完整性 (0-1)
    freshness_score: float  # 新鲜度 (0-1)
    importance_score: float  # 重要性 (0-1)
    
    @property
    def overall_score(self) -> float:
        """综合质量分数"""
        weights = {
            'clarity': 0.3,
            'relevance': 0.25,
            'completeness': 0.2,
            'freshness': 0.15,
            'importance': 0.1
        }
        return (
            self.clarity_score * weights['clarity'] +
            self.relevance_score * weights['relevance'] +
            self.completeness_score * weights['completeness'] +
            self.freshness_score * weights['freshness'] +
            self.importance_score * weights['importance']
        )


@dataclass
class CompressionConfig:
    """压缩配置"""
    min_compression_ratio: float = 0.3  # 最小压缩比例
    max_compression_ratio: float = 0.7  # 最大压缩比例
    preserve_keywords: bool = True  # 保留关键词
    preserve_named_entities: bool = True  # 保留命名实体
    preserve_numbers: bool = True  # 保留数字
    preserve_dates: bool = True  # 保留日期
    semantic_preservation_threshold: float = 0.8  # 语义保留阈值


class MemoryOptimizer:
    """记忆优化器核心类"""
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        """
        初始化记忆优化器
        
        Args:
            config: 压缩配置
        """
        self.config = config or CompressionConfig()
        
        # 初始化NLTK（如果可用）
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.warning("NLTK资源未找到，将使用简单分词")
        
        # 关键词提取器
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # 语义模型
        self.semantic_model = None
        
        logger.info("MemoryOptimizer initialized")
    
    def optimize_memory_content(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        优化记忆内容
        
        Args:
            content: 原始记忆内容
            context: 上下文信息
            
        Returns:
            Dict[str, Any]: 优化后的记忆
        """
        start_time = time.time()
        
        # 1. 分析原始内容
        analysis = self._analyze_content(content, context)
        
        # 2. 智能压缩
        compressed_content = self._compress_content(content, analysis)
        
        # 3. 提取关键信息
        key_info = self._extract_key_information(content, analysis)
        
        # 4. 构建结构化记忆
        structured_memory = self._structure_memory(
            original_content=content,
            compressed_content=compressed_content,
            key_info=key_info,
            analysis=analysis,
            context=context
        )
        
        # 5. 计算质量指标
        quality_metrics = self._calculate_quality_metrics(
            original_content=content,
            compressed_content=compressed_content,
            key_info=key_info,
            analysis=analysis
        )
        
        # 6. 生成优化报告
        optimization_report = {
            'compression_ratio': len(compressed_content) / max(len(content), 1),
            'key_info_count': len(key_info),
            'quality_score': quality_metrics.overall_score,
            'processing_time': time.time() - start_time
        }
        
        result = {
            'content': compressed_content,
            'original_length': len(content),
            'compressed_length': len(compressed_content),
            'key_information': key_info,
            'structured_data': structured_memory,
            'quality_metrics': quality_metrics.__dict__,
            'optimization_report': optimization_report,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Memory optimized: {len(content)} -> {len(compressed_content)} chars, "
                   f"quality: {quality_metrics.overall_score:.3f}")
        
        return result
    
    def _analyze_content(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析内容特征"""
        analysis = {
            'length': len(content),
            'word_count': len(content.split()),
            'sentence_count': len(sent_tokenize(content)) if len(content) > 50 else 1,
            'contains_numbers': bool(re.search(r'\d+', content)),
            'contains_dates': bool(re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}', content)),
            'contains_entities': self._detect_entities(content),
            'key_topics': self._extract_topics(content),
            'readability_score': self._calculate_readability(content),
            'context_relevance': self._calculate_context_relevance(content, context)
        }
        
        return analysis
    
    def _compress_content(self, content: str, analysis: Dict[str, Any]) -> str:
        """智能压缩内容"""
        if len(content) < 200:
            # 短内容不压缩
            return content
        
        # 提取关键句子
        sentences = sent_tokenize(content)
        if len(sentences) <= 3:
            return content
        
        # 计算句子重要性
        sentence_scores = self._score_sentences(sentences, analysis)
        
        # 选择重要句子
        target_sentences = max(3, int(len(sentences) * self.config.max_compression_ratio))
        selected_indices = heapq.nlargest(
            target_sentences,
            range(len(sentences)),
            key=lambda i: sentence_scores[i]
        )
        selected_indices.sort()  # 保持原始顺序
        
        # 构建压缩内容
        compressed_sentences = [sentences[i] for i in selected_indices]
        compressed_content = ' '.join(compressed_sentences)
        
        # 确保语义保留
        if self._calculate_semantic_similarity(content, compressed_content) < self.config.semantic_preservation_threshold:
            # 语义损失太大，减少压缩
            return content[:int(len(content) * 0.7)]
        
        return compressed_content
    
    def _extract_key_information(self, content: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """提取关键信息"""
        key_info = []
        
        # 1. 提取关键词
        keywords = self._extract_keywords(content)
        for keyword, score in keywords[:10]:
            key_info.append({
                'type': 'keyword',
                'value': keyword,
                'score': score,
                'source': 'tfidf'
            })
        
        # 2. 提取命名实体
        entities = analysis['contains_entities']
        for entity_type, entities_list in entities.items():
            for entity in entities_list[:5]:
                key_info.append({
                    'type': 'entity',
                    'value': entity,
                    'entity_type': entity_type,
                    'source': 'ner'
                })
        
        # 3. 提取数字和日期
        numbers = re.findall(r'\b\d+\b', content)
        for number in set(numbers)[:5]:
            key_info.append({
                'type': 'number',
                'value': number,
                'source': 'regex'
            })
        
        dates = re.findall(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}', content)
        for date in set(dates)[:3]:
            key_info.append({
                'type': 'date',
                'value': date,
                'source': 'regex'
            })
        
        # 4. 提取主题
        for topic, score in analysis['key_topics'][:5]:
            key_info.append({
                'type': 'topic',
                'value': topic,
                'score': score,
                'source': 'topic_modeling'
            })
        
        return key_info
    
    def _structure_memory(self, original_content: str, compressed_content: str,
                         key_info: List[Dict[str, Any]], analysis: Dict[str, Any],
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """构建结构化记忆"""
        structured = {
            'summary': compressed_content,
            'categories': self._categorize_content(original_content),
            'entities': self._extract_structured_entities(original_content),
            'timeline': self._extract_timeline_info(original_content),
            'relationships': self._extract_relationships(key_info),
            'metadata': {
                'source': context.get('source', 'unknown'),
                'author': context.get('author', 'unknown'),
                'created_at': context.get('created_at', datetime.now().isoformat()),
                'updated_at': datetime.now().isoformat(),
                'version': '1.0'
            }
        }
        
        return structured
    
    def _calculate_quality_metrics(self, original_content: str, compressed_content: str,
                                  key_info: List[Dict[str, Any]], analysis: Dict[str, Any]) -> MemoryQualityMetrics:
        """计算质量指标"""
        # 清晰度：基于可读性和结构
        clarity = analysis['readability_score']
        
        # 相关性：基于上下文关联
        relevance = analysis['context_relevance']
        
        # 完整性：基于关键信息保留
        completeness = min(1.0, len(key_info) / 10.0)
        
        # 新鲜度：基于时间（假设新内容质量更高）
        freshness = 0.8  # 默认值，实际应根据时间计算
        
        # 重要性：基于内容特征
        importance = self._calculate_importance(original_content, analysis)
        
        return MemoryQualityMetrics(
            clarity_score=clarity,
            relevance_score=relevance,
            completeness_score=completeness,
            freshness_score=freshness,
            importance_score=importance
        )
    
    def _score_sentences(self, sentences: List[str], analysis: Dict[str, Any]) -> List[float]:
        """为句子评分"""
        scores = []
        
        for i, sentence in enumerate(sentences):
            score = 0.0
            
            # 1. 位置权重（开头和结尾的句子通常更重要）
            if i == 0:
                score += 0.3
            elif i == len(sentences) - 1:
                score += 0.2
            
            # 2. 长度权重（中等长度的句子通常包含更多信息）
            sentence_length = len(sentence.split())
            if 10 <= sentence_length <= 30:
                score += 0.2
            elif sentence_length > 30:
                score += 0.1
            
            # 3. 关键词权重
            keywords = analysis.get('key_topics', [])
            for keyword, _ in keywords:
                if keyword.lower() in sentence.lower():
                    score += 0.1
            
            # 4. 实体权重
            entities = analysis['contains_entities']
            for entity_list in entities.values():
                for entity in entity_list:
                    if entity in sentence:
                        score += 0.15
            
            # 5. 数字/日期权重
            if re.search(r'\d+', sentence):
                score += 0.05
            
            scores.append(min(1.0, score))
        
        return scores
    
    def _extract_keywords(self, content: str) -> List[Tuple[str, float]]:
        """提取关键词"""
        try:
            # 使用TF-IDF提取关键词
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([content])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            keywords = []
            for col in tfidf_matrix.nonzero()[1]:
                keywords.append((feature_names[col], tfidf_matrix[0, col]))
            
            return sorted(keywords, key=lambda x: x[1], reverse=True)
        except:
            # 回退到简单方法
            words = word_tokenize(content.lower())
            words = [w for w in words if w.isalpha() and len(w) > 2]
            word_freq = Counter(words)
            return [(word, freq/len(words)) for word, freq in word_freq.most_common(20)]
    
    def _detect_entities(self, content: str) -> Dict[str, List[str]]:
        """检测命名实体（简化版）"""
        entities = {
            'person': [],
            'organization': [],
            'location': [],
            'date': [],
            'number': []
        }
        
        # 简单规则匹配（实际应使用NER模型）
        # 人名（大写单词）
        words = content.split()
        for i, word in enumerate(words):
            if word.istitle() and len(word) > 1:
                if i > 0 and words[i-1].istitle():
                    # 可能是名字的一部分
                    entities['person'].append(f"{words[i-1]} {word}")
                else:
                    entities['person'].append(word)
        
        # 组织（包含Inc., Corp., Ltd.等）
        org_patterns = ['inc', 'corp', 'ltd', 'co', 'llc']
        for pattern in org_patterns:
            if pattern in content.lower():
                # 提取组织名
                pass
        
        # 位置（常见城市/国家名）
        locations = ['beijing', 'shanghai', 'new york', 'london', 'tokyo']
        for loc in locations:
            if loc in content.lower():
                entities['location'].append(loc.title())
        
        return entities
    
    def _extract_topics(self, content: str) -> List[Tuple[str, float]]:
        """提取主题"""
        # 简化版：使用高频词作为主题
        words = word_tokenize(content.lower())
        words = [w for w in words if w.isalpha() and len(w) > 3]
        word_freq = Counter(words)
        
        topics = []
        for word, freq in word_freq.most_common(10):
            score = freq / len(words)
            topics.append((word, score))
        
        return topics
    
    def _calculate_readability(self, content: str) -> float:
        """计算可读性分数"""
        sentences = sent_tokenize(content)
        if not sentences:
            return 0.5
        
        words = word_tokenize(content)
        
        # 简单可读性指标
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(w) for w in words) /