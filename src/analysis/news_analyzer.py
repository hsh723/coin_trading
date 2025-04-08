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
import pytz
import aiohttp
import re

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
            'cnbc': 'https://www.cnbc.com/id/19746125/device/rss/rss.xml',
            'marketwatch': 'https://www.marketwatch.com/rss/topstories',
            'bitcoinmagazine': 'https://bitcoinmagazine.com/'
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
        
    def _parse_date(self, date_str: str) -> datetime:
        """날짜 문자열 파싱"""
        try:
            # 다양한 날짜 형식 처리
            formats = [
                '%a, %d %b %Y %H:%M:%S %Z',
                '%a, %d %b %Y %H:%M:%S %z',
                '%Y-%m-%dT%H:%M:%S%z',
                '%Y-%m-%d %H:%M:%S'
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    if dt.tzinfo is None:
                        dt = pytz.UTC.localize(dt)
                    return dt
                except ValueError:
                    continue
                    
            raise ValueError(f"지원하지 않는 날짜 형식: {date_str}")
            
        except Exception as e:
            self.logger.error(f"날짜 파싱 실패: {str(e)}")
            return datetime.now(pytz.UTC)
        
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
                        'timestamp': self._parse_date(entry.published),
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
                        'timestamp': self._parse_date(article['publishedAt']),
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
                    'timestamp': self._parse_date(item['published_on']),
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
        
    async def analyze_news_impact(self, symbol: str, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        뉴스 영향 분석
        
        Args:
            symbol (str): 심볼
            news_data (List[Dict[str, Any]]): 뉴스 데이터
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        try:
            # 가격 데이터 가져오기
            price_data = await self._get_price_data(symbol)
            
            # 뉴스 영향 분석
            impact_analysis = []
            for news in news_data:
                impact = await self._analyze_single_news(news, price_data)
                impact_analysis.append(impact)
                
            # 종합 분석
            summary = self._summarize_impact(impact_analysis)
            
            return {
                'individual_impacts': impact_analysis,
                'summary': summary
            }
            
        except Exception as e:
            self.logger.error(f"뉴스 영향 분석 실패: {str(e)}")
            return {}
            
    async def _get_price_data(self, symbol: str) -> pd.DataFrame:
        """
        가격 데이터 조회
        
        Args:
            symbol (str): 심볼
            
        Returns:
            pd.DataFrame: 가격 데이터
        """
        try:
            # 데이터베이스에서 가격 데이터 조회
            price_data = await self.db.get_market_data(symbol)
            return pd.DataFrame(price_data)
            
        except Exception as e:
            self.logger.error(f"가격 데이터 조회 실패: {str(e)}")
            return pd.DataFrame()
            
    async def _analyze_single_news(self, news: Dict[str, Any], price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        개별 뉴스 영향 분석
        
        Args:
            news (Dict[str, Any]): 뉴스 데이터
            price_data (pd.DataFrame): 가격 데이터
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        try:
            news_time = datetime.fromisoformat(news['timestamp'])
            
            # 뉴스 전후 가격 변화 분석
            pre_period = price_data[price_data['timestamp'] < news_time]
            post_period = price_data[price_data['timestamp'] >= news_time]
            
            if len(pre_period) == 0 or len(post_period) == 0:
                return {
                    'news_id': news['id'],
                    'impact': 'insufficient_data'
                }
                
            # 가격 변화 계산
            pre_price = pre_period['close'].iloc[-1]
            post_price = post_period['close'].iloc[0]
            price_change = (post_price - pre_price) / pre_price * 100
            
            # 변동성 변화 계산
            pre_volatility = pre_period['close'].pct_change().std()
            post_volatility = post_period['close'].pct_change().std()
            volatility_change = (post_volatility - pre_volatility) / pre_volatility * 100
            
            # 거래량 변화 계산
            pre_volume = pre_period['volume'].mean()
            post_volume = post_period['volume'].mean()
            volume_change = (post_volume - pre_volume) / pre_volume * 100
            
            return {
                'news_id': news['id'],
                'title': news['title'],
                'timestamp': news['timestamp'],
                'price_change': price_change,
                'volatility_change': volatility_change,
                'volume_change': volume_change,
                'impact': self._classify_impact(price_change, volatility_change, volume_change)
            }
            
        except Exception as e:
            self.logger.error(f"개별 뉴스 분석 실패: {str(e)}")
            return {}
            
    def _classify_impact(self, price_change: float, volatility_change: float, volume_change: float) -> str:
        """
        영향도 분류
        
        Args:
            price_change (float): 가격 변화율
            volatility_change (float): 변동성 변화율
            volume_change (float): 거래량 변화율
            
        Returns:
            str: 영향도 등급
        """
        # 가격 변화 기준
        if abs(price_change) > 5:
            impact = 'high'
        elif abs(price_change) > 2:
            impact = 'medium'
        else:
            impact = 'low'
            
        # 변동성과 거래량 변화 고려
        if volatility_change > 50 or volume_change > 100:
            impact = 'high'
        elif volatility_change > 20 or volume_change > 50:
            impact = max(impact, 'medium')
            
        return impact
        
    def _summarize_impact(self, impact_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        영향 분석 요약
        
        Args:
            impact_analysis (List[Dict[str, Any]]): 개별 영향 분석 결과
            
        Returns:
            Dict[str, Any]: 요약 결과
        """
        try:
            # 영향도별 카운트
            impact_counts = {
                'high': 0,
                'medium': 0,
                'low': 0
            }
            
            # 평균 변화율
            price_changes = []
            volatility_changes = []
            volume_changes = []
            
            for analysis in impact_analysis:
                if 'impact' in analysis:
                    impact_counts[analysis['impact']] += 1
                    
                if 'price_change' in analysis:
                    price_changes.append(analysis['price_change'])
                if 'volatility_change' in analysis:
                    volatility_changes.append(analysis['volatility_change'])
                if 'volume_change' in analysis:
                    volume_changes.append(analysis['volume_change'])
                    
            return {
                'impact_distribution': impact_counts,
                'average_price_change': np.mean(price_changes) if price_changes else 0,
                'average_volatility_change': np.mean(volatility_changes) if volatility_changes else 0,
                'average_volume_change': np.mean(volume_changes) if volume_changes else 0,
                'total_news': len(impact_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"영향 분석 요약 실패: {str(e)}")
            return {}
        
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

    async def fetch_news(self, source: str) -> List[Dict[str, Any]]:
        """
        뉴스 수집
        
        Args:
            source (str): 뉴스 소스
            
        Returns:
            List[Dict[str, Any]]: 뉴스 목록
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.rss_feeds[source]) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        news_items = []
                        if source == 'coindesk':
                            articles = soup.find_all('article')
                            for article in articles:
                                title = article.find('h3')
                                if title:
                                    news_items.append({
                                        'title': title.text.strip(),
                                        'url': article.find('a')['href'],
                                        'source': source,
                                        'timestamp': datetime.now()
                                    })
                        elif source == 'cointelegraph':
                            articles = soup.find_all('article')
                            for article in articles:
                                title = article.find('h3')
                                if title:
                                    news_items.append({
                                        'title': title.text.strip(),
                                        'url': article.find('a')['href'],
                                        'source': source,
                                        'timestamp': datetime.now()
                                    })
                        
                        return news_items
                    else:
                        self.logger.error(f"뉴스 수집 실패: {response.status}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"뉴스 수집 중 오류 발생: {str(e)}")
            return []
            
    async def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        감성 분석
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            Dict[str, float]: 감성 분석 결과
        """
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment
            
            return {
                'polarity': sentiment.polarity,  # -1 ~ 1
                'subjectivity': sentiment.subjectivity  # 0 ~ 1
            }
            
        except Exception as e:
            self.logger.error(f"감성 분석 중 오류 발생: {str(e)}")
            return {'polarity': 0, 'subjectivity': 0}
            
    def calculate_impact_score(self, news: Dict[str, Any]) -> float:
        """
        뉴스 영향도 점수 계산
        
        Args:
            news (Dict[str, Any]): 뉴스 정보
            
        Returns:
            float: 영향도 점수
        """
        try:
            # 감성 점수 (0 ~ 1)
            sentiment_score = (news['sentiment']['polarity'] + 1) / 2
            
            # 키워드 중요도
            keywords = {
                'bitcoin': 1.0,
                'ethereum': 0.8,
                'regulation': 0.9,
                'exchange': 0.7,
                'price': 0.6
            }
            
            keyword_score = 0
            for keyword, weight in keywords.items():
                if keyword in news['title'].lower():
                    keyword_score += weight
            
            # 소스 신뢰도
            source_credibility = {
                'coindesk': 1.0,
                'cointelegraph': 0.9,
                'cryptonews': 0.8,
                'bitcoinmagazine': 0.9
            }
            
            # 최종 점수 계산
            impact_score = (
                sentiment_score * 0.4 +
                keyword_score * 0.4 +
                source_credibility[news['source']] * 0.2
            )
            
            return min(max(impact_score, 0), 1)
            
        except Exception as e:
            self.logger.error(f"영향도 점수 계산 중 오류 발생: {str(e)}")
            return 0
            
    async def analyze_news(self, symbol: str) -> List[Dict[str, Any]]:
        """
        뉴스 분석
        
        Args:
            symbol (str): 심볼
            
        Returns:
            List[Dict[str, Any]]: 분석 결과
        """
        try:
            all_news = []
            
            # 각 소스에서 뉴스 수집
            for source in self.rss_feeds:
                news_items = await self.fetch_news(source)
                all_news.extend(news_items)
            
            # 뉴스 분석
            analyzed_news = []
            for news in all_news:
                # 감성 분석
                sentiment = await self.analyze_sentiment(news['title'])
                news['sentiment'] = sentiment
                
                # 영향도 점수 계산
                impact_score = self.calculate_impact_score(news)
                news['impact_score'] = impact_score
                
                # 감성 분류
                if sentiment['polarity'] > 0.2:
                    news['sentiment_label'] = '긍정'
                elif sentiment['polarity'] < -0.2:
                    news['sentiment_label'] = '부정'
                else:
                    news['sentiment_label'] = '중립'
                
                # 영향도 분류
                if impact_score > 0.7:
                    news['impact_label'] = '높음'
                elif impact_score > 0.4:
                    news['impact_label'] = '중간'
                else:
                    news['impact_label'] = '낮음'
                
                analyzed_news.append(news)
            
            # 데이터베이스에 저장
            await self.db.save_news(analyzed_news)
            
            return analyzed_news
            
        except Exception as e:
            self.logger.error(f"뉴스 분석 중 오류 발생: {str(e)}")
            return []
            
    async def get_recent_news(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        최근 뉴스 조회
        
        Args:
            symbol (str): 심볼
            limit (int): 조회할 뉴스 수
            
        Returns:
            List[Dict[str, Any]]: 뉴스 목록
        """
        try:
            return await self.db.get_recent_news(symbol, limit)
            
        except Exception as e:
            self.logger.error(f"최근 뉴스 조회 중 오류 발생: {str(e)}")
            return []
            
    async def get_news_summary(self, symbol: str) -> Dict[str, Any]:
        """
        뉴스 요약
        
        Args:
            symbol (str): 심볼
            
        Returns:
            Dict[str, Any]: 요약 정보
        """
        try:
            recent_news = await self.get_recent_news(symbol)
            
            if not recent_news:
                return {
                    'total_news': 0,
                    'positive_news': 0,
                    'negative_news': 0,
                    'average_impact': 0
                }
            
            total_news = len(recent_news)
            positive_news = sum(1 for news in recent_news if news['sentiment_label'] == '긍정')
            negative_news = sum(1 for news in recent_news if news['sentiment_label'] == '부정')
            average_impact = sum(news['impact_score'] for news in recent_news) / total_news
            
            return {
                'total_news': total_news,
                'positive_news': positive_news,
                'negative_news': negative_news,
                'average_impact': average_impact
            }
            
        except Exception as e:
            self.logger.error(f"뉴스 요약 생성 중 오류 발생: {str(e)}")
            return {
                'total_news': 0,
                'positive_news': 0,
                'negative_news': 0,
                'average_impact': 0
            }

# 전역 뉴스 분석기 인스턴스
db_manager = DatabaseManager()  # DatabaseManager 인스턴스 생성
news_analyzer = NewsAnalyzer(db=db_manager)  # db 매개변수 전달 