"""
자산 캐시 관리자 - 실시간 거래에 필요한 자산 데이터 캐싱 및 관리
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict

# 로거 설정
logger = logging.getLogger(__name__)

class AssetCacheManager:
    """자산 데이터 캐시 관리자"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        자산 캐시 관리자 초기화
        
        Args:
            config (Dict[str, Any]): 설정 정보
        """
        self.config = config
        
        # 캐시 설정
        self.cache_ttl = config.get('cache_ttl', 60)  # 캐시 유효 시간(초)
        self.refresh_interval = config.get('refresh_interval', 5)  # 갱신 주기(초)
        self.max_cache_size = config.get('max_cache_size', 10000)  # 최대 캐시 크기
        
        # 자산 데이터 캐시
        self.cache = {}  # 데이터 저장
        self.subscribed_symbols = set()  # 구독 중인 심볼
        
        # 상태 변수
        self.is_running = False
        self.tasks = []
        
    async def initialize(self):
        """자산 캐시 관리자 초기화"""
        try:
            # 초기 데이터 로드
            await self._load_initial_data()
            
            # 캐시 갱신 작업 시작
            self.is_running = True
            self.tasks.append(asyncio.create_task(self._refresh_cache_loop()))
            self.tasks.append(asyncio.create_task(self._clean_cache_loop()))
            
            logger.info("자산 캐시 관리자 초기화 완료")
            
        except Exception as e:
            logger.error(f"자산 캐시 관리자 초기화 실패: {str(e)}")
            raise
            
    async def close(self):
        """리소스 정리"""
        try:
            self.is_running = False
            
            # 실행 중인 작업 취소
            for task in self.tasks:
                if not task.done():
                    task.cancel()
                    
            # 작업 완료 대기
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
                
            # 캐시 정리
            self._clear_cache()
            
            logger.info("자산 캐시 관리자 종료")
            
        except Exception as e:
            logger.error(f"자산 캐시 관리자 종료 실패: {str(e)}")
            
    async def _load_initial_data(self):
        """초기 데이터 로드"""
        try:
            # 기본 심볼 목록
            default_symbols = self.config.get('default_symbols', [])
            
            # 구독 등록
            for symbol in default_symbols:
                await self.subscribe_symbol(symbol)
                
            logger.info(f"초기 데이터 로드 완료: {len(default_symbols)}개 심볼")
            
        except Exception as e:
            logger.error(f"초기 데이터 로드 실패: {str(e)}")
            
    async def _refresh_cache_loop(self):
        """캐시 갱신 루프"""
        try:
            while self.is_running:
                # 구독 중인 심볼의 데이터 갱신
                for symbol in self.subscribed_symbols:
                    await self._refresh_symbol_data(symbol)
                    
                # 대기
                await asyncio.sleep(self.refresh_interval)
                
        except asyncio.CancelledError:
            logger.info("캐시 갱신 작업 취소됨")
        except Exception as e:
            logger.error(f"캐시 갱신 루프 실패: {str(e)}")
            
    async def _clean_cache_loop(self):
        """캐시 정리 루프"""
        try:
            while self.is_running:
                # 캐시 사이즈 확인
                current_size = self._get_cache_size()
                
                # 최대 크기 초과 시 정리
                if current_size > self.max_cache_size:
                    await self._prune_cache()
                    
                # 만료된 캐시 정리
                self._remove_expired_cache()
                
                # 대기 (1분 간격)
                await asyncio.sleep(60)
                
        except asyncio.CancelledError:
            logger.info("캐시 정리 작업 취소됨")
        except Exception as e:
            logger.error(f"캐시 정리 루프 실패: {str(e)}")
            
    async def _refresh_symbol_data(self, symbol: str):
        """
        심볼 데이터 갱신
        
        Args:
            symbol (str): 거래 심볼
        """
        try:
            # 가격 데이터 갱신
            await self._update_price(symbol)
            
            # 오더북 데이터 갱신
            await self._update_orderbook(symbol)
            
            # 갱신 시간 기록
            self.cache[symbol]['last_update_time'] = datetime.now()
            
        except Exception as e:
            logger.error(f"{symbol} 데이터 갱신 실패: {str(e)}")
            
    async def _update_price(self, symbol: str):
        """
        가격 데이터 갱신
        
        Args:
            symbol (str): 거래 심볼
        """
        try:
            # TODO: 거래소 API에서 데이터 가져오기
            # 예제 데이터 (실제로는 거래소 API 호출 필요)
            price_data = {
                'price': 50000.0,
                'timestamp': datetime.now()
            }
            
            # 캐시 업데이트
            self.cache[symbol]['price'] = price_data
            
        except Exception as e:
            logger.error(f"{symbol} 가격 갱신 실패: {str(e)}")
            
    async def _update_orderbook(self, symbol: str):
        """
        오더북 데이터 갱신
        
        Args:
            symbol (str): 거래 심볼
        """
        try:
            # TODO: 거래소 API에서 데이터 가져오기
            # 예제 데이터 (실제로는 거래소 API 호출 필요)
            orderbook_data = {
                'bids': [(49900.0, 1.5), (49800.0, 2.3)],
                'asks': [(50100.0, 1.2), (50200.0, 3.1)],
                'timestamp': datetime.now()
            }
            
            # 캐시 업데이트
            self.cache[symbol]['orderbook'] = orderbook_data
            
        except Exception as e:
            logger.error(f"{symbol} 오더북 갱신 실패: {str(e)}")
            
    def _get_cache_size(self) -> int:
        """
        현재 캐시 크기 계산
        
        Returns:
            int: 캐시 크기
        """
        size = len(self.cache)
        
        for symbol, data in self.cache.items():
            size += len(data)
            
        return size
        
    async def _prune_cache(self):
        """캐시 정리"""
        try:
            # 정리 대상 확인
            now = datetime.now()
            inactive_symbols = []
            
            # 일정 시간 동안 사용되지 않은 심볼 찾기
            for symbol, data in self.cache.items():
                if 'last_update_time' in data:
                    last_time = data['last_update_time']
                    if (now - last_time).total_seconds() > (self.cache_ttl * 2):
                        if symbol not in self.subscribed_symbols:
                            inactive_symbols.append(symbol)
                            
            # 사용하지 않는 심볼 데이터 제거
            for symbol in inactive_symbols:
                self._remove_symbol_data(symbol)
                
            logger.info(f"캐시 정리 완료: {len(inactive_symbols)}개 심볼 제거")
            
        except Exception as e:
            logger.error(f"캐시 정리 실패: {str(e)}")
            
    def _remove_expired_cache(self):
        """만료된 캐시 제거"""
        try:
            # 현재 시간
            now = datetime.now()
            expired_time = now - timedelta(seconds=self.cache_ttl)
            
            # 만료된 데이터 제거
            for symbol in list(self.cache.keys()):
                data = self.cache[symbol]
                if 'last_update_time' in data:
                    last_time = data['last_update_time']
                    if last_time < expired_time:
                        self._remove_symbol_data(symbol)
                
        except Exception as e:
            logger.error(f"만료된 캐시 제거 실패: {str(e)}")
            
    def _remove_symbol_data(self, symbol: str):
        """
        심볼 데이터 제거
        
        Args:
            symbol (str): 거래 심볼
        """
        try:
            # 캐시에서 제거
            if symbol in self.cache:
                del self.cache[symbol]
                
        except Exception as e:
            logger.error(f"{symbol} 데이터 제거 실패: {str(e)}")
            
    def _clear_cache(self):
        """캐시 정리"""
        self.cache.clear()
        
    async def subscribe_symbol(self, symbol: str):
        """
        심볼 구독
        
        Args:
            symbol (str): 거래 심볼
        """
        try:
            # 이미 구독 중인지 확인
            if symbol in self.subscribed_symbols:
                logger.debug(f"{symbol} 이미 구독 중")
                return
                
            # 구독 추가
            self.subscribed_symbols.add(symbol)
            
            # 초기 데이터 로드
            await self._refresh_symbol_data(symbol)
            
            logger.info(f"{symbol} 구독 완료")
            
        except Exception as e:
            logger.error(f"{symbol} 구독 실패: {str(e)}")
            
    async def unsubscribe_symbol(self, symbol: str):
        """
        심볼 구독 해제
        
        Args:
            symbol (str): 거래 심볼
        """
        try:
            # 구독 중인지 확인
            if symbol not in self.subscribed_symbols:
                logger.debug(f"{symbol} 구독 중이 아님")
                return
                
            # 구독 제거
            self.subscribed_symbols.remove(symbol)
            
            logger.info(f"{symbol} 구독 해제 완료")
            
        except Exception as e:
            logger.error(f"{symbol} 구독 해제 실패: {str(e)}")
            
    def get_price(self, symbol: str) -> Optional[float]:
        """
        현재가 조회
        
        Args:
            symbol (str): 거래 심볼
            
        Returns:
            Optional[float]: 현재가
        """
        try:
            price_data = self.cache.get(symbol, {}).get('price')
            if price_data:
                return price_data.get('price')
            return None
            
        except Exception as e:
            logger.error(f"{symbol} 현재가 조회 실패: {str(e)}")
            return None
            
    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        """
        오더북 조회
        
        Args:
            symbol (str): 거래 심볼
            
        Returns:
            Optional[Dict]: 오더북 데이터
        """
        try:
            return self.cache.get(symbol, {}).get('orderbook')
            
        except Exception as e:
            logger.error(f"{symbol} 오더북 조회 실패: {str(e)}")
            return None
            
    def get_recent_trades(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        최근 체결 내역 조회
        
        Args:
            symbol (str): 거래 심볼
            limit (int): 제한 개수
            
        Returns:
            List[Dict]: 체결 내역
        """
        try:
            trades = self.cache.get(symbol, {}).get('trades', [])
            return trades[:limit]
            
        except Exception as e:
            logger.error(f"{symbol} 체결 내역 조회 실패: {str(e)}")
            return []
            
    def get_asset_info(self, symbol: str) -> Optional[Dict]:
        """
        자산 정보 조회
        
        Args:
            symbol (str): 거래 심볼
            
        Returns:
            Optional[Dict]: 자산 정보
        """
        try:
            return self.cache.get(symbol, {}).get('asset_info')
            
        except Exception as e:
            logger.error(f"{symbol} 자산 정보 조회 실패: {str(e)}")
            return None
            
    def get_all_prices(self) -> Dict[str, float]:
        """
        모든 현재가 조회
        
        Returns:
            Dict[str, float]: 심볼별 현재가
        """
        try:
            return {
                symbol: data.get('price')
                for symbol, data in self.cache.items()
            }
            
        except Exception as e:
            logger.error(f"모든 현재가 조회 실패: {str(e)}")
            return {}
            
    def update_market_data(self, symbol: str, data: Dict[str, Any]):
        """
        시장 데이터 업데이트
        
        Args:
            symbol (str): 거래 심볼
            data (Dict[str, Any]): 시장 데이터
        """
        try:
            self.cache[symbol]['market_data'] = {
                **data,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"{symbol} 시장 데이터 업데이트 실패: {str(e)}")
            
    def update_asset_info(self, symbol: str, info: Dict[str, Any]):
        """
        자산 정보 업데이트
        
        Args:
            symbol (str): 거래 심볼
            info (Dict[str, Any]): 자산 정보
        """
        try:
            self.cache[symbol]['asset_info'] = {
                **info,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"{symbol} 자산 정보 업데이트 실패: {str(e)}")
            
    def add_trade(self, symbol: str, trade: Dict[str, Any]):
        """
        체결 내역 추가
        
        Args:
            symbol (str): 거래 심볼
            trade (Dict[str, Any]): 체결 내역
        """
        try:
            # 타임스탬프 추가
            if 'timestamp' not in trade:
                trade['timestamp'] = datetime.now()
                
            # 체결 내역 추가
            self.cache[symbol]['trades'] = self.cache[symbol].get('trades', []) + [trade]
            
            # 최대 개수 제한
            max_trades = self.config.get('max_trades_per_symbol', 1000)
            if len(self.cache[symbol]['trades']) > max_trades:
                self.cache[symbol]['trades'] = self.cache[symbol]['trades'][-max_trades:]
                
        except Exception as e:
            logger.error(f"{symbol} 체결 내역 추가 실패: {str(e)}")
            
    def get_subscribed_symbols(self) -> Set[str]:
        """
        구독 중인 심볼 목록 조회
        
        Returns:
            Set[str]: 구독 중인 심볼 목록
        """
        return self.subscribed_symbols.copy()
        
    def is_price_valid(self, symbol: str) -> bool:
        """
        가격 데이터 유효성 확인
        
        Args:
            symbol (str): 거래 심볼
            
        Returns:
            bool: 유효성 여부
        """
        try:
            # 데이터 존재 여부 확인
            if symbol not in self.cache:
                return False
                
            # 타임스탬프 확인
            data = self.cache[symbol]
            timestamp = data.get('last_update_time')
            
            if not timestamp:
                return False
                
            # 유효 시간 확인
            now = datetime.now()
            return (now - timestamp).total_seconds() <= self.cache_ttl
            
        except Exception as e:
            logger.error(f"{symbol} 가격 유효성 확인 실패: {str(e)}")
            return False 