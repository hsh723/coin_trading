"""
뉴스 수집기 모듈
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import aiohttp
import json
from bs4 import BeautifulSoup
import os
from src.utils.encryption import encryption_manager

logger = logging.getLogger(__name__)

class NewsCollector:
    def __init__(self):
        """
        뉴스 수집기 초기화
        """
        self.api_keys = {
            'cryptocompare': os.getenv('CRYPTOCOMPARE_API_KEY'),
            'newsapi': os.getenv('NEWSAPI_API_KEY'),
            'cryptopanic': os.getenv('CRYPTOPANIC_API_KEY')
        }
        
        self.news_sources = [
            {
                'name': 'CryptoCompare',
                'url': 'https://min-api.cryptocompare.com/data/v2/news/',
                'params': {
                    'categories': 'BTC,ETH,Blockchain',
                    'excludeCategories': 'Sponsored',
                    'lang': 'EN'
                }
            },
            {
                'name': 'NewsAPI',
                'url': 'https://newsapi.org/v2/everything',
                'params': {
                    'q': 'bitcoin OR ethereum OR cryptocurrency',
                    'language': 'en',
                    'sortBy': 'publishedAt'
                }
            },
            {
                'name': 'CryptoPanic',
                'url': 'https://cryptopanic.com/api/v1/posts/',
                'params': {
                    'auth_token': self.api_keys['cryptopanic'],
                    'kind': 'news',
                    'filter': 'hot'
                }
            }
        ]
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    async def fetch_news(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        뉴스 데이터 수집
        
        Args:
            hours (int): 수집할 시간 범위 (기본값: 24시간)
            
        Returns:
            List[Dict[str, Any]]: 수집된 뉴스 목록
        """
        try:
            all_news = []
            since_time = datetime.now() - timedelta(hours=hours)
            
            async with aiohttp.ClientSession(headers=self.headers) as session:
                for source in self.news_sources:
                    try:
                        # API 요청
                        async with session.get(
                            source['url'],
                            params=source['params']
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                news_items = self._parse_news(data, source['name'])
                                
                                # 시간 필터링
                                filtered_news = [
                                    news for news in news_items
                                    if news['published_at'] >= since_time
                                ]
                                
                                all_news.extend(filtered_news)
                            else:
                                logger.warning(f"{source['name']} API 요청 실패: {response.status}")
                                
                    except Exception as e:
                        logger.error(f"{source['name']} 뉴스 수집 중 오류 발생: {str(e)}")
                        continue
            
            # 중복 제거 및 시간순 정렬
            unique_news = self._remove_duplicates(all_news)
            sorted_news = sorted(unique_news, key=lambda x: x['published_at'], reverse=True)
            
            # 뉴스 데이터 저장
            await self._save_news(sorted_news)
            
            return sorted_news
            
        except Exception as e:
            logger.error(f"뉴스 수집 중 오류 발생: {str(e)}")
            return []
    
    def _parse_news(self, data: Dict[str, Any], source: str) -> List[Dict[str, Any]]:
        """
        뉴스 데이터 파싱
        
        Args:
            data (Dict[str, Any]): API 응답 데이터
            source (str): 뉴스 소스
            
        Returns:
            List[Dict[str, Any]]: 파싱된 뉴스 목록
        """
        news_items = []
        
        try:
            if source == 'CryptoCompare':
                for item in data.get('Data', []):
                    news_items.append({
                        'title': item.get('title', ''),
                        'content': item.get('body', ''),
                        'url': item.get('url', ''),
                        'source': item.get('source', ''),
                        'published_at': datetime.fromtimestamp(item.get('published_on', 0)),
                        'sentiment': self._analyze_sentiment(item.get('title', '') + ' ' + item.get('body', ''))
                    })
            
            elif source == 'NewsAPI':
                for item in data.get('articles', []):
                    news_items.append({
                        'title': item.get('title', ''),
                        'content': item.get('description', ''),
                        'url': item.get('url', ''),
                        'source': item.get('source', {}).get('name', ''),
                        'published_at': datetime.strptime(item.get('publishedAt', ''), '%Y-%m-%dT%H:%M:%SZ'),
                        'sentiment': self._analyze_sentiment(item.get('title', '') + ' ' + item.get('description', ''))
                    })
            
            elif source == 'CryptoPanic':
                for item in data.get('results', []):
                    news_items.append({
                        'title': item.get('title', ''),
                        'content': item.get('text', ''),
                        'url': item.get('url', ''),
                        'source': item.get('source', {}).get('title', ''),
                        'published_at': datetime.strptime(item.get('published_at', ''), '%Y-%m-%dT%H:%M:%SZ'),
                        'sentiment': self._analyze_sentiment(item.get('title', '') + ' ' + item.get('text', ''))
                    })
            
        except Exception as e:
            logger.error(f"뉴스 파싱 중 오류 발생 ({source}): {str(e)}")
        
        return news_items
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        텍스트 감성 분석
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            float: 감성 점수 (-1.0 ~ 1.0)
        """
        try:
            # 긍정/부정 키워드 분석
            positive_words = {'bullish', 'surge', 'rise', 'gain', 'up', 'high', 'positive', 'good', 'great', 'excellent'}
            negative_words = {'bearish', 'drop', 'fall', 'down', 'low', 'negative', 'bad', 'poor', 'terrible', 'crash'}
            
            words = text.lower().split()
            pos_count = sum(1 for word in words if word in positive_words)
            neg_count = sum(1 for word in words if word in negative_words)
            
            total = pos_count + neg_count
            if total == 0:
                return 0.0
            
            return (pos_count - neg_count) / total
            
        except Exception as e:
            logger.error(f"감성 분석 중 오류 발생: {str(e)}")
            return 0.0
    
    def _remove_duplicates(self, news_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        중복 뉴스 제거
        
        Args:
            news_list (List[Dict[str, Any]]): 뉴스 목록
            
        Returns:
            List[Dict[str, Any]]: 중복 제거된 뉴스 목록
        """
        seen_urls = set()
        unique_news = []
        
        for news in news_list:
            if news['url'] not in seen_urls:
                seen_urls.add(news['url'])
                unique_news.append(news)
        
        return unique_news
    
    async def _save_news(self, news_list: List[Dict[str, Any]]) -> None:
        """
        뉴스 데이터 저장
        
        Args:
            news_list (List[Dict[str, Any]]): 저장할 뉴스 목록
        """
        try:
            # 뉴스 데이터를 JSON 형식으로 변환
            news_data = []
            for news in news_list:
                news_data.append({
                    'title': news['title'],
                    'content': news['content'],
                    'url': news['url'],
                    'source': news['source'],
                    'published_at': news['published_at'].isoformat(),
                    'sentiment': news['sentiment']
                })
            
            # 데이터 암호화
            encrypted_data = encryption_manager.encrypt(json.dumps(news_data))
            
            # 파일로 저장
            filename = f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(f"data/news/{filename}", 'wb') as f:
                f.write(encrypted_data)
            
            logger.info(f"뉴스 데이터 저장 완료: {filename}")
            
        except Exception as e:
            logger.error(f"뉴스 데이터 저장 중 오류 발생: {str(e)}")

# 전역 뉴스 수집기 인스턴스
news_collector = NewsCollector() 