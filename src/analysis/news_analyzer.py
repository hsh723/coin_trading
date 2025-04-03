"""
뉴스 분석기 모듈
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from textblob import TextBlob
import pandas as pd
import numpy as np
from src.data.news_collector import news_collector
from src.utils.database import db_manager

logger = logging.getLogger(__name__)

class NewsAnalyzer:
    def __init__(self):
        """
        뉴스 분석기 초기화
        """
        self.sentiment_threshold = 0.3
        self.importance_threshold = 0.5
        self.keywords = {
            'bitcoin': ['bitcoin', 'btc', 'crypto', 'blockchain'],
            'ethereum': ['ethereum', 'eth', 'defi', 'smart contract'],
            'regulation': ['regulation', 'law', 'government', 'sec', 'ban'],
            'technology': ['technology', 'upgrade', 'fork', 'network'],
            'market': ['market', 'price', 'trading', 'volume']
        }
    
    async def analyze_market_sentiment(self) -> float:
        """
        시장 감성 분석
        
        Returns:
            float: 시장 감성 점수 (-1.0 ~ 1.0)
        """
        try:
            # 최근 24시간 뉴스 수집
            news_list = await news_collector.fetch_news(hours=24)
            
            if not news_list:
                logger.warning("수집된 뉴스가 없습니다.")
                return 0.0
            
            # 감성 분석
            sentiment_scores = []
            importance_weights = []
            
            for news in news_list:
                # 감성 점수 계산
                sentiment = self._analyze_sentiment(news['title'] + ' ' + news['content'])
                
                # 중요도 계산
                importance = self._calculate_importance(news)
                
                sentiment_scores.append(sentiment)
                importance_weights.append(importance)
            
            # 가중 평균 계산
            if importance_weights:
                total_weight = sum(importance_weights)
                weighted_sentiment = sum(
                    s * w for s, w in zip(sentiment_scores, importance_weights)
                ) / total_weight
                
                # 결과 저장
                await self._save_sentiment_analysis(weighted_sentiment, news_list)
                
                return weighted_sentiment
            else:
                return 0.0
            
        except Exception as e:
            logger.error(f"시장 감성 분석 중 오류 발생: {str(e)}")
            return 0.0
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        텍스트 감성 분석
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            float: 감성 점수 (-1.0 ~ 1.0)
        """
        try:
            # TextBlob을 사용한 감성 분석
            blob = TextBlob(text)
            return blob.sentiment.polarity
            
        except Exception as e:
            logger.error(f"감성 분석 중 오류 발생: {str(e)}")
            return 0.0
    
    def _calculate_importance(self, news: Dict[str, Any]) -> float:
        """
        뉴스 중요도 계산
        
        Args:
            news (Dict[str, Any]): 뉴스 데이터
            
        Returns:
            float: 중요도 점수 (0.0 ~ 1.0)
        """
        try:
            importance_scores = []
            
            # 키워드 기반 중요도
            text = (news['title'] + ' ' + news['content']).lower()
            for category, keywords in self.keywords.items():
                keyword_count = sum(1 for keyword in keywords if keyword in text)
                if keyword_count > 0:
                    importance_scores.append(min(keyword_count / len(keywords), 1.0))
            
            # 소스 신뢰도
            source_trust = self._get_source_trust(news['source'])
            importance_scores.append(source_trust)
            
            # 최근성
            recency = self._calculate_recency(news['published_at'])
            importance_scores.append(recency)
            
            # 평균 중요도 계산
            if importance_scores:
                return sum(importance_scores) / len(importance_scores)
            else:
                return 0.0
            
        except Exception as e:
            logger.error(f"중요도 계산 중 오류 발생: {str(e)}")
            return 0.0
    
    def _get_source_trust(self, source: str) -> float:
        """
        뉴스 소스 신뢰도 계산
        
        Args:
            source (str): 뉴스 소스
            
        Returns:
            float: 신뢰도 점수 (0.0 ~ 1.0)
        """
        # 신뢰도가 높은 소스 목록
        trusted_sources = {
            'reuters': 1.0,
            'bloomberg': 1.0,
            'coindesk': 0.9,
            'cointelegraph': 0.9,
            'cryptopanic': 0.8,
            'cryptocompare': 0.8
        }
        
        source_lower = source.lower()
        for trusted_source, trust_score in trusted_sources.items():
            if trusted_source in source_lower:
                return trust_score
        
        return 0.5  # 기본 신뢰도
    
    def _calculate_recency(self, published_at: datetime) -> float:
        """
        뉴스 최근성 계산
        
        Args:
            published_at (datetime): 발행 시간
            
        Returns:
            float: 최근성 점수 (0.0 ~ 1.0)
        """
        try:
            hours_old = (datetime.now() - published_at).total_seconds() / 3600
            return max(0.0, 1.0 - (hours_old / 24))  # 24시간 기준
            
        except Exception as e:
            logger.error(f"최근성 계산 중 오류 발생: {str(e)}")
            return 0.5
    
    async def _save_sentiment_analysis(self, sentiment: float, news_list: List[Dict[str, Any]]) -> None:
        """
        감성 분석 결과 저장
        
        Args:
            sentiment (float): 감성 점수
            news_list (List[Dict[str, Any]]): 뉴스 목록
        """
        try:
            # 분석 결과 데이터 준비
            analysis_data = {
                'timestamp': datetime.now(),
                'sentiment_score': sentiment,
                'news_count': len(news_list),
                'average_importance': sum(
                    self._calculate_importance(news) for news in news_list
                ) / len(news_list) if news_list else 0.0
            }
            
            # 데이터베이스에 저장
            await db_manager.save_sentiment_analysis(analysis_data)
            
            logger.info(f"감성 분석 결과 저장 완료: {sentiment:.2f}")
            
        except Exception as e:
            logger.error(f"감성 분석 결과 저장 중 오류 발생: {str(e)}")
    
    async def get_recent_sentiment(self, hours: int = 24) -> float:
        """
        최근 감성 점수 조회
        
        Args:
            hours (int): 조회할 시간 범위
            
        Returns:
            float: 평균 감성 점수
        """
        try:
            # 데이터베이스에서 최근 감성 분석 결과 조회
            sentiment_data = await db_manager.get_recent_sentiment(hours)
            
            if sentiment_data:
                # 시간 가중 평균 계산
                total_weight = 0
                weighted_sum = 0
                
                for data in sentiment_data:
                    weight = self._calculate_recency(data['timestamp'])
                    weighted_sum += data['sentiment_score'] * weight
                    total_weight += weight
                
                if total_weight > 0:
                    return weighted_sum / total_weight
            
            return 0.0
            
        except Exception as e:
            logger.error(f"최근 감성 점수 조회 중 오류 발생: {str(e)}")
            return 0.0

# 전역 뉴스 분석기 인스턴스
news_analyzer = NewsAnalyzer() 