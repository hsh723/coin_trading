"""
API 관리 모듈
"""

import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import asyncio
import aiohttp
import json
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ExchangeType(Enum):
    """거래소 타입"""
    BINANCE = "binance"
    BYBIT = "bybit"
    KUCOIN = "kucoin"
    OKX = "okx"
    GATEIO = "gateio"

@dataclass
class MarketData:
    """시장 데이터"""
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: float
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    liquidation: Optional[float] = None

@dataclass
class OrderBook:
    """호가 데이터"""
    symbol: str
    timestamp: datetime
    bids: List[List[float]]
    asks: List[List[float]]
    spread: float
    depth: int

@dataclass
class Trade:
    """거래 데이터"""
    symbol: str
    timestamp: datetime
    side: str
    price: float
    amount: float
    cost: float
    taker: bool

class APIManager:
    """API 관리 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        초기화
        
        Args:
            config: API 설정
        """
        self.config = config
        self.exchanges = {}
        self.session = None
        self.news_api_key = config.get('news_api_key')
        self.sentiment_api_key = config.get('sentiment_api_key')
        
        # 거래소 초기화
        self._init_exchanges()
        
        # 세션 초기화
        self._init_session()
    
    def _init_exchanges(self):
        """거래소 초기화"""
        try:
            for exchange_id in self.config.get('exchanges', []):
                exchange_class = getattr(ccxt, exchange_id)
                exchange = exchange_class({
                    'apiKey': self.config.get(f'{exchange_id}_api_key'),
                    'secret': self.config.get(f'{exchange_id}_api_secret'),
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future',
                        'adjustForTimeDifference': True
                    }
                })
                self.exchanges[exchange_id] = exchange
                
        except Exception as e:
            logger.error(f"거래소 초기화 중 오류 발생: {str(e)}")
    
    def _init_session(self):
        """세션 초기화"""
        self.session = aiohttp.ClientSession()
    
    async def close(self):
        """리소스 정리"""
        if self.session:
            await self.session.close()
    
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 100,
        exchange_id: str = None
    ) -> List[MarketData]:
        """
        시장 데이터 조회
        
        Args:
            symbol: 심볼
            timeframe: 시간대
            limit: 데이터 개수
            exchange_id: 거래소 ID
            
        Returns:
            List[MarketData]: 시장 데이터 목록
        """
        try:
            if exchange_id:
                exchange = self.exchanges[exchange_id]
            else:
                exchange = next(iter(self.exchanges.values()))
            
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            market_data = []
            for data in ohlcv:
                market_data.append(MarketData(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=datetime.fromtimestamp(data[0] / 1000),
                    open=data[1],
                    high=data[2],
                    low=data[3],
                    close=data[4],
                    volume=data[5],
                    spread=0.0  # 기본값
                ))
            
            return market_data
            
        except Exception as e:
            logger.error(f"시장 데이터 조회 중 오류 발생: {str(e)}")
            return []
    
    async def get_order_book(
        self,
        symbol: str,
        limit: int = 20,
        exchange_id: str = None
    ) -> Optional[OrderBook]:
        """
        호가 데이터 조회
        
        Args:
            symbol: 심볼
            limit: 호가 깊이
            exchange_id: 거래소 ID
            
        Returns:
            Optional[OrderBook]: 호가 데이터
        """
        try:
            if exchange_id:
                exchange = self.exchanges[exchange_id]
            else:
                exchange = next(iter(self.exchanges.values()))
            
            orderbook = await exchange.fetch_order_book(symbol, limit)
            
            return OrderBook(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(orderbook['timestamp'] / 1000),
                bids=orderbook['bids'],
                asks=orderbook['asks'],
                spread=orderbook['asks'][0][0] - orderbook['bids'][0][0],
                depth=limit
            )
            
        except Exception as e:
            logger.error(f"호가 데이터 조회 중 오류 발생: {str(e)}")
            return None
    
    async def get_trades(
        self,
        symbol: str,
        limit: int = 100,
        exchange_id: str = None
    ) -> List[Trade]:
        """
        거래 내역 조회
        
        Args:
            symbol: 심볼
            limit: 데이터 개수
            exchange_id: 거래소 ID
            
        Returns:
            List[Trade]: 거래 내역
        """
        try:
            if exchange_id:
                exchange = self.exchanges[exchange_id]
            else:
                exchange = next(iter(self.exchanges.values()))
            
            trades = await exchange.fetch_trades(symbol, limit=limit)
            
            trade_list = []
            for trade in trades:
                trade_list.append(Trade(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(trade['timestamp'] / 1000),
                    side=trade['side'],
                    price=trade['price'],
                    amount=trade['amount'],
                    cost=trade['cost'],
                    taker=trade['taker']
                ))
            
            return trade_list
            
        except Exception as e:
            logger.error(f"거래 내역 조회 중 오류 발생: {str(e)}")
            return []
    
    async def get_funding_rate(
        self,
        symbol: str,
        exchange_id: str = None
    ) -> Optional[float]:
        """
        자금 조달 비율 조회
        
        Args:
            symbol: 심볼
            exchange_id: 거래소 ID
            
        Returns:
            Optional[float]: 자금 조달 비율
        """
        try:
            if exchange_id:
                exchange = self.exchanges[exchange_id]
            else:
                exchange = next(iter(self.exchanges.values()))
            
            funding_rate = await exchange.fetch_funding_rate(symbol)
            return funding_rate['fundingRate']
            
        except Exception as e:
            logger.error(f"자금 조달 비율 조회 중 오류 발생: {str(e)}")
            return None
    
    async def get_open_interest(
        self,
        symbol: str,
        exchange_id: str = None
    ) -> Optional[float]:
        """
        미체결약정 조회
        
        Args:
            symbol: 심볼
            exchange_id: 거래소 ID
            
        Returns:
            Optional[float]: 미체결약정
        """
        try:
            if exchange_id:
                exchange = self.exchanges[exchange_id]
            else:
                exchange = next(iter(self.exchanges.values()))
            
            oi = await exchange.fetch_open_interest(symbol)
            return oi['openInterestAmount']
            
        except Exception as e:
            logger.error(f"미체결약정 조회 중 오류 발생: {str(e)}")
            return None
    
    async def get_liquidation(
        self,
        symbol: str,
        exchange_id: str = None
    ) -> Optional[float]:
        """
        청산 데이터 조회
        
        Args:
            symbol: 심볼
            exchange_id: 거래소 ID
            
        Returns:
            Optional[float]: 청산 금액
        """
        try:
            if exchange_id:
                exchange = self.exchanges[exchange_id]
            else:
                exchange = next(iter(self.exchanges.values()))
            
            liquidations = await exchange.fetch_liquidations(symbol)
            return sum(liq['amount'] for liq in liquidations)
            
        except Exception as e:
            logger.error(f"청산 데이터 조회 중 오류 발생: {str(e)}")
            return None
    
    async def get_news(
        self,
        symbol: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        뉴스 데이터 조회
        
        Args:
            symbol: 심볼
            limit: 데이터 개수
            
        Returns:
            List[Dict[str, Any]]: 뉴스 데이터
        """
        try:
            if not self.news_api_key:
                logger.warning("뉴스 API 키가 설정되지 않았습니다.")
                return []
            
            url = f"https://newsapi.org/v2/everything"
            params = {
                'q': symbol,
                'apiKey': self.news_api_key,
                'pageSize': limit,
                'sortBy': 'publishedAt'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('articles', [])
                else:
                    logger.error(f"뉴스 API 요청 실패: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"뉴스 데이터 조회 중 오류 발생: {str(e)}")
            return []
    
    async def get_sentiment(
        self,
        symbol: str,
        text: str
    ) -> Optional[Dict[str, float]]:
        """
        감성 분석
        
        Args:
            symbol: 심볼
            text: 분석할 텍스트
            
        Returns:
            Optional[Dict[str, float]]: 감성 분석 결과
        """
        try:
            if not self.sentiment_api_key:
                logger.warning("감성 분석 API 키가 설정되지 않았습니다.")
                return None
            
            url = "https://api.sentiment.com/v1/analyze"
            headers = {
                'Authorization': f'Bearer {self.sentiment_api_key}',
                'Content-Type': 'application/json'
            }
            data = {
                'text': text,
                'symbol': symbol
            }
            
            async with self.session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"감성 분석 API 요청 실패: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"감성 분석 중 오류 발생: {str(e)}")
            return None
    
    async def get_market_sentiment(
        self,
        symbol: str,
        limit: int = 100
    ) -> Optional[Dict[str, float]]:
        """
        시장 감성 분석
        
        Args:
            symbol: 심볼
            limit: 분석할 뉴스 개수
            
        Returns:
            Optional[Dict[str, float]]: 시장 감성 분석 결과
        """
        try:
            # 뉴스 데이터 조회
            news = await self.get_news(symbol, limit)
            
            if not news:
                return None
            
            # 각 뉴스에 대한 감성 분석
            sentiments = []
            for article in news:
                sentiment = await self.get_sentiment(symbol, article['title'] + ' ' + article['description'])
                if sentiment:
                    sentiments.append(sentiment)
            
            if not sentiments:
                return None
            
            # 평균 감성 점수 계산
            avg_sentiment = {
                'positive': np.mean([s['positive'] for s in sentiments]),
                'negative': np.mean([s['negative'] for s in sentiments]),
                'neutral': np.mean([s['neutral'] for s in sentiments])
            }
            
            return avg_sentiment
            
        except Exception as e:
            logger.error(f"시장 감성 분석 중 오류 발생: {str(e)}")
            return None 