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
import os
from newsapi import NewsApiClient
from newspaper import Article
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import feedparser

# 내부 모듈 import
from ..data.news_collector import news_collector
from ..utils.database import DatabaseManager
from ..utils.logger import setup_logger

logger = logging.getLogger(__name__)

# NLTK 데이터 다운로드
nltk.download('vader_lexicon')
nltk.download('punkt')

class NewsAnalyzer:
    """뉴스 분석기 클래스"""
    
    def __init__(self, db: DatabaseManager):
        """
        뉴스 분석기 초기화
        
        Args:
            db (DatabaseManager): 데이터베이스 매니저 인스턴스
        """
        self.logger = setup_logger('news_analyzer')
        self.db = db
        self.sia = SentimentIntensityAnalyzer()
        
        # RSS 피드 URL
        self.rss_feeds = {
            # 코인 관련 뉴스
            'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'cointelegraph': 'https://cointelegraph.com/rss',
            'cryptonews': 'https://cryptonews.com/news/feed/',
            
            # 미국 경제 뉴스
            'bloomberg': 'https://www.bloomberg.com/markets/rss',
            'reuters': 'https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best',
            'wsj': 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml',
            'cnbc': 'https://www.cnbc.com/id/10000664/device/rss/rss.html',
            'marketwatch': 'https://www.marketwatch.com/rss/topstories'
        }
        
        # CryptoCompare API 설정
        self.cryptocompare_config = {
            'base_url': 'https://min-api.cryptocompare.com/data/v2',
            'api_key': '8e01e9886a4b576e5fb0c4134f44fa42edec2581546fff5477c0fd9c936ee707'
        }
        
        # NewsAPI 설정 (미국 경제 뉴스용)
        self.newsapi_config = {
            'base_url': 'https://newsapi.org/v2',
            'api_key': None  # NewsAPI 키는 환경 변수에서 가져오도록 설정
        }
        
    def fetch_news_from_rss(self) -> List[Dict]:
        """RSS 피드에서 뉴스 수집"""
        all_news = []
        
        for source, url in self.rss_feeds.items():
            try:
                feed = feedparser.parse(url)
                
                for entry in feed.entries:
                    news_item = {
                        'title': entry.title,
                        'link': entry.link,
                        'timestamp': datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %z'),
                        'source': source,
                        'summary': entry.get('summary', ''),
                        'category': 'crypto' if source in ['coindesk', 'cointelegraph', 'cryptonews'] else 'economy'
                    }
                    all_news.append(news_item)
                    
            except Exception as e:
                self.logger.error(f"RSS 피드 수집 중 오류 발생 ({source}): {str(e)}")
                
        return all_news
        
    def fetch_news_from_newsapi(self) -> List[Dict]:
        """NewsAPI에서 코인 가격에 영향을 미치는 경제 뉴스 수집"""
        if not self.newsapi_config['api_key']:
            return []
            
        try:
            # 코인 가격에 영향을 미치는 주요 키워드 카테고리
            crypto_impact_keywords = {
                'regulation': {
                    'keywords': [
                        'government regulation', 'regulatory', 'SEC', 'CFTC', 'legal changes',
                        'regulatory clarity', 'crypto regulation', 'crypto ban', 'crypto restrictions',
                        'legal tender', 'crypto law', 'crypto policy', 'government ban', 'regulatory framework'
                    ],
                    'weight': 1.5  # 가장 높은 가중치
                },
                'institutional': {
                    'keywords': [
                        'institutional investment', 'corporate adoption', 'Tesla', 'MicroStrategy',
                        'institutional investor', 'hedge fund', 'asset management', 'pension fund',
                        'investment fund', 'corporate treasury', 'company investment'
                    ],
                    'weight': 1.3
                },
                'security': {
                    'keywords': [
                        'hack', 'security breach', 'vulnerability', 'exploit',
                        'exchange security', 'wallet security', 'smart contract bug',
                        'cyber attack', 'security incident', 'funds stolen', 'security flaw'
                    ],
                    'weight': 1.4
                },
                'technology': {
                    'keywords': [
                        'blockchain upgrade', 'protocol improvement', 'security enhancement',
                        'scalability', 'layer 2', 'mainnet', 'hard fork', 'soft fork',
                        'technical upgrade', 'network upgrade', 'protocol update'
                    ],
                    'weight': 1.2
                },
                'market_sentiment': {
                    'keywords': [
                        'whale movement', 'large transaction', 'investor sentiment',
                        'FOMO', 'FUD', 'market manipulation', 'pump and dump',
                        'market psychology', 'investor confidence', 'market trend'
                    ],
                    'weight': 1.1
                },
                'economic_indicators': {
                    'keywords': [
                        'inflation rate', 'interest rates', 'Federal Reserve', 'Fed',
                        'global economy', 'recession', 'economic growth', 'GDP',
                        'monetary policy', 'quantitative easing', 'QE', 'tapering',
                        'economic uncertainty', 'market volatility'
                    ],
                    'weight': 1.3
                },
                'adoption': {
                    'keywords': [
                        'corporate adoption', 'institutional investment', 'use case',
                        'partnership', 'integration', 'merchant adoption',
                        'payment integration', 'business adoption', 'commercial use'
                    ],
                    'weight': 1.2
                },
                'influencer': {
                    'keywords': [
                        'Elon Musk', 'Michael Saylor', 'Jack Dorsey', 'CZ',
                        'Vitalik Buterin', 'crypto influencer', 'crypto expert',
                        'market analyst', 'industry leader'
                    ],
                    'weight': 1.1
                }
            }
            
            # 모든 키워드를 하나의 쿼리로 결합
            all_keywords = []
            for category in crypto_impact_keywords.values():
                all_keywords.extend(category['keywords'])
            query = ' OR '.join(all_keywords)
            
            url = f"{self.newsapi_config['base_url']}/everything"
            params = {
                'apiKey': self.newsapi_config['api_key'],
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 100
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            news_data = response.json()
            all_news = []
            
            for article in news_data['articles']:
                # 제목과 내용에서 키워드 매칭 확인
                text = f"{article['title']} {article.get('description', '')}".lower()
                matched_keywords = {}
                total_impact_score = 0
                
                # 각 카테고리별로 키워드 매칭 확인
                for category, config in crypto_impact_keywords.items():
                    matched = [kw for kw in config['keywords'] if kw.lower() in text]
                    if matched:
                        matched_keywords[category] = matched
                        # 카테고리별 가중치를 고려한 영향력 점수 계산
                        total_impact_score += len(matched) * config['weight']
                
                if matched_keywords:  # 관련 키워드가 있는 뉴스만 포함
                    news_item = {
                        'title': article['title'],
                        'link': article['url'],
                        'timestamp': datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'),
                        'source': article['source']['name'],
                        'summary': article.get('description', ''),
                        'category': 'economy',
                        'impact_categories': list(matched_keywords.keys()),
                        'impact_keywords': matched_keywords,
                        'impact_score': total_impact_score,
                        'primary_impact': max(matched_keywords.items(), key=lambda x: len(x[1]))[0] if matched_keywords else None
                    }
                    all_news.append(news_item)
                
            return all_news
            
        except Exception as e:
            self.logger.error(f"NewsAPI 뉴스 수집 중 오류 발생: {str(e)}")
            return []
            
    def fetch_news_from_cryptocompare(self) -> List[Dict]:
        """CryptoCompare API에서 뉴스 수집"""
        try:
            url = f"{self.cryptocompare_config['base_url']}/news/"
            params = {
                'api_key': self.cryptocompare_config['api_key']
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            news_data = response.json()
            all_news = []
            
            for item in news_data['Data']:
                news_item = {
                    'title': item['title'],
                    'link': item['url'],
                    'timestamp': datetime.fromtimestamp(item['published_on']),
                    'source': 'cryptocompare',
                    'summary': item.get('body', ''),
                    'category': 'crypto'
                }
                all_news.append(news_item)
                
            return all_news
            
        except Exception as e:
            self.logger.error(f"CryptoCompare API 뉴스 수집 중 오류 발생: {str(e)}")
            return []
            
    def fetch_news(self) -> List[Dict]:
        """모든 소스에서 뉴스 수집"""
        all_news = []
        
        # RSS 피드에서 뉴스 수집
        rss_news = self.fetch_news_from_rss()
        all_news.extend(rss_news)
        
        # NewsAPI에서 미국 경제 뉴스 수집
        if self.newsapi_config['api_key']:
            newsapi_news = self.fetch_news_from_newsapi()
            all_news.extend(newsapi_news)
        
        # CryptoCompare API에서 뉴스 수집
        crypto_news = self.fetch_news_from_cryptocompare()
        all_news.extend(crypto_news)
            
        # 중복 제거
        seen = set()
        unique_news = []
        for news in all_news:
            key = (news['title'], news['link'])
            if key not in seen:
                seen.add(key)
                unique_news.append(news)
                
        # 타임스탬프로 정렬
        unique_news.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return unique_news
        
    def analyze_sentiment(self, text: str) -> Dict:
        """텍스트 감정 분석"""
        sentiment_scores = self.sia.polarity_scores(text)
        return {
            'positive': sentiment_scores['pos'],
            'negative': sentiment_scores['neg'],
            'neutral': sentiment_scores['neu'],
            'compound': sentiment_scores['compound']
        }
        
    def get_market_sentiment(self, category: str = 'all') -> Dict:
        """시장 전체 감정 분석"""
        news = self.fetch_news()
        
        # 카테고리 필터링
        if category != 'all':
            news = [item for item in news if item['category'] == category]
        
        if not news:
            return {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'news_count': 0,
                'positive_news': 0,
                'negative_news': 0,
                'neutral_news': 0,
                'category': category
            }
            
        sentiment_scores = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for item in news:
            sentiment = self.analyze_sentiment(item['title'] + ' ' + item.get('summary', ''))
            sentiment_scores.append(sentiment['compound'])
            
            if sentiment['compound'] > 0.2:
                positive_count += 1
            elif sentiment['compound'] < -0.2:
                negative_count += 1
            else:
                neutral_count += 1
                
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        if avg_sentiment > 0.2:
            overall_sentiment = 'positive'
        elif avg_sentiment < -0.2:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
            
        result = {
            'overall_sentiment': overall_sentiment,
            'sentiment_score': avg_sentiment,
            'news_count': len(news),
            'positive_news': positive_count,
            'negative_news': negative_count,
            'neutral_news': neutral_count,
            'category': category
        }
        
        # 데이터베이스에 저장
        self.db.save_market_sentiment(result)
        
        return result
        
    def analyze_news_impact(self, symbol: str, timeframe: str = '1h', include_economy: bool = True) -> Dict:
        """특정 심볼에 대한 뉴스 영향력 분석"""
        news = self.fetch_news()
        
        # 경제 뉴스 포함 여부
        if not include_economy:
            news = [item for item in news if item['category'] == 'crypto']
            
        price_data = self.db.get_price_data(symbol, timeframe)
        
        if not news or not price_data:
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'news_count': 0,
                'price_correlation': 0.0,
                'significant_events': []
            }
            
        # 뉴스와 가격 데이터 정렬
        news_df = pd.DataFrame(news)
        price_df = pd.DataFrame(price_data)
        
        # 시간대 맞추기
        news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
        
        # 뉴스 발생 후 가격 변동 분석
        significant_events = []
        for _, news_item in news_df.iterrows():
            news_time = news_item['timestamp']
            
            # 뉴스 발생 후 1시간 동안의 가격 변동
            price_after = price_df[price_df['timestamp'] > news_time].head(4)  # 1시간 (15분 간격)
            
            if not price_after.empty:
                price_change = (price_after['close'].iloc[-1] - price_after['close'].iloc[0]) / price_after['close'].iloc[0] * 100
                
                # 경제 뉴스의 경우 영향력 점수를 고려하여 임계값 조정
                threshold = 1.0
                if news_item['category'] == 'economy':
                    threshold = 1.0 * (news_item.get('impact_score', 1) / 2)  # 영향력 점수에 따라 임계값 조정
                
                if abs(price_change) > threshold:
                    significant_events.append({
                        'title': news_item['title'],
                        'timestamp': news_time,
                        'price_change': price_change,
                        'sentiment': self.analyze_sentiment(news_item['title']),
                        'category': news_item['category'],
                        'impact_keywords': news_item.get('impact_keywords', []),
                        'impact_score': news_item.get('impact_score', 1)
                    })
                    
        # 가격과 뉴스 감정의 상관관계 계산
        sentiment_scores = [self.analyze_sentiment(title)['compound'] for title in news_df['title']]
        price_changes = price_df['close'].pct_change().dropna()
        
        if len(sentiment_scores) > 1 and len(price_changes) > 1:
            correlation = pd.Series(sentiment_scores).corr(pd.Series(price_changes))
        else:
            correlation = 0.0
            
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'news_count': len(news),
            'price_correlation': correlation,
            'significant_events': significant_events,
            'include_economy': include_economy
        }
        
        # 데이터베이스에 저장
        self.db.save_news_impact(symbol, result)
        
        return result
        
    def get_news_summary(self, timeframe: str = '24h', category: str = 'all') -> Dict:
        """특정 기간 동안의 뉴스 요약"""
        news = self.fetch_news()
        
        # 카테고리 필터링
        if category != 'all':
            news = [item for item in news if item['category'] == category]
        
        if not news:
            return {
                'timeframe': timeframe,
                'category': category,
                'total_news': 0,
                'sentiment_distribution': {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                },
                'top_keywords': [],
                'significant_events': []
            }
            
        # 시간대 필터링
        if timeframe == '24h':
            cutoff = datetime.now() - timedelta(hours=24)
        elif timeframe == '7d':
            cutoff = datetime.now() - timedelta(days=7)
        else:
            cutoff = datetime.now() - timedelta(hours=1)
            
        filtered_news = [item for item in news if item['timestamp'] > cutoff]
        
        # 감정 분포 계산
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for item in filtered_news:
            sentiment = self.analyze_sentiment(item['title'])
            if sentiment['compound'] > 0.2:
                sentiment_counts['positive'] += 1
            elif sentiment['compound'] < -0.2:
                sentiment_counts['negative'] += 1
            else:
                sentiment_counts['neutral'] += 1
                
        # 키워드 추출 (간단한 구현)
        all_text = ' '.join([item['title'] + ' ' + item.get('summary', '') for item in filtered_news])
        words = all_text.lower().split()
        word_counts = {}
        for word in words:
            if len(word) > 3:  # 3글자 이상의 단어만 고려
                word_counts[word] = word_counts.get(word, 0) + 1
                
        top_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # 중요한 이벤트 식별
        significant_events = []
        for item in filtered_news:
            sentiment = self.analyze_sentiment(item['title'])
            if abs(sentiment['compound']) > 0.5:  # 강한 감정
                significant_events.append({
                    'title': item['title'],
                    'timestamp': item['timestamp'],
                    'sentiment': sentiment,
                    'source': item['source'],
                    'category': item['category']
                })
                
        return {
            'timeframe': timeframe,
            'category': category,
            'total_news': len(filtered_news),
            'sentiment_distribution': sentiment_counts,
            'top_keywords': [word for word, _ in top_keywords],
            'significant_events': significant_events
        }

# 전역 뉴스 분석기 인스턴스
db_manager = DatabaseManager()  # DatabaseManager 인스턴스 생성
news_analyzer = NewsAnalyzer(db=db_manager)  # db 매개변수 전달 