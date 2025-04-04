"""
트레이딩 봇 모듈
"""

import os
import sys
from pathlib import Path
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import time
from abc import ABC, abstractmethod
from src.exchange.binance_exchange import BinanceExchange
from ..analysis.news_analyzer import NewsAnalyzer
from ..analysis.technical_analyzer import TechnicalAnalyzer
from ..strategy.integrated_strategy import IntegratedStrategy
from ..risk.risk_manager import RiskManager
from ..database.database import Database
from ..utils.logger import setup_logger
from ..utils.database import DatabaseManager
from ..utils.telegram import TelegramNotifier

# 프로젝트 루트 경로를 시스템 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 지연 임포트 적용
def get_exchange():
    return BinanceExchange()

def get_logger():
    return setup_logger('trading_bot')

class TradingBot(ABC):
    """트레이딩 봇 기본 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        트레이딩 봇 초기화
        
        Args:
            config (Dict[str, Any]): 설정 정보
        """
        self.config = config
        self.logger = get_logger()
        self.database = DatabaseManager()
        self.exchange = get_exchange()
        self.strategy = IntegratedStrategy(db=self.database)
        self.risk_manager = RiskManager(
            initial_capital=config.get('initial_capital', 10000.0)
        )
        
        # 텔레그램 알림 설정
        bot_token = config.get('telegram_bot_token')
        chat_id = config.get('telegram_chat_id')
        if bot_token and chat_id:
            self.telegram = TelegramNotifier(bot_token=bot_token, chat_id=chat_id)
        else:
            self.telegram = None
            self.logger.warning("텔레그램 설정이 없습니다. 알림이 비활성화됩니다.")
        
        self.market_data = None
        self.current_position = None
        self._is_running = False
        self.last_update = None
        self.timeframes = ['5m', '15m', '1h', '4h']  # 멀티 타임프레임
        self.strategy_weights = {
            '5m': 0.2,
            '15m': 0.3,
            '1h': 0.3,
            '4h': 0.2
        }
        
    @property
    def is_running(self) -> bool:
        """봇 실행 상태 반환"""
        return self._is_running
        
    @is_running.setter
    def is_running(self, value: bool) -> None:
        """봇 실행 상태 설정"""
        self._is_running = value
        
    async def start(self) -> None:
        """트레이딩 봇 시작"""
        if self.is_running:
            self.logger.warning("트레이딩 봇이 이미 실행 중입니다.")
            return
            
        try:
            self.is_running = True
            self.logger.info("트레이딩 봇 시작")
            self.database.save_log('INFO', '트레이딩 봇 시작', 'trading_bot')
            
            # 메인 루프를 별도의 태스크로 실행
            self._task = asyncio.create_task(self._run_loop())
            await self._task
            
        except Exception as e:
            error_msg = f"트레이딩 봇 시작 실패: {str(e)}"
            self.logger.error(error_msg)
            self.database.save_log('ERROR', error_msg, 'trading_bot')
            self.is_running = False
            
    async def stop(self) -> None:
        """트레이딩 봇 중지"""
        if not self.is_running:
            self.logger.warning("트레이딩 봇이 이미 중지되었습니다.")
            return
            
        self.is_running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
                
        self.logger.info("트레이딩 봇 중지")
        self.database.save_log('INFO', '트레이딩 봇 중지', 'trading_bot')
        
    async def _run_loop(self) -> None:
        """트레이딩 루프 실행"""
        try:
            while self.is_running:
                try:
                    # 시장 데이터 업데이트
                    await self._update_market_data()
                    
                    # 포지션 업데이트
                    await self._update_positions()
                    
                    # 거래 신호 생성
                    signal = await self._generate_signal()
                    
                    # 리스크 관리
                    if await self._check_risk_limits():
                        # 거래 실행
                        await self._execute_trade(signal)
                    
                    # 성과 분석
                    await self._update_performance()
                    
                    # 로깅
                    self.logger.info(f"트레이딩 루프 실행 완료: {datetime.now()}")
                    
                except Exception as e:
                    self.logger.error(f"트레이딩 루프 실행 중 오류: {str(e)}")
                
                # 인터벌 대기
                await asyncio.sleep(self.config.get('interval', 60))
                
        except asyncio.CancelledError:
            # 정상적인 취소 처리
            self.logger.info("트레이딩 루프가 취소되었습니다.")
        except Exception as e:
            self.logger.error(f"트레이딩 루프 오류: {str(e)}")
        finally:
            self.is_running = False
            self.logger.info("트레이딩 루프가 종료되었습니다.")
            
    async def _update_market_data(self) -> None:
        """시장 데이터 업데이트"""
        try:
            for timeframe in self.timeframes:
                # 각 시간대별 데이터 조회
                data = await self.database.get_market_data(timeframe)
                
                if data is not None:
                    # 기술적 지표 계산
                    indicators = await self.strategy.calculate_indicators(data)
                    
                    # 데이터베이스에 저장
                    await self.database.save_technical_indicators(timeframe, indicators)
                    
            # 시장 데이터 업데이트
            self.market_data = {
                'current_price': await self.exchange.fetch_current_price(self.config['symbol']),
                'volatility': self.strategy.calculate_volatility(self.market_data['ohlcv']) if self.market_data else None,
                'trend_strength': self.strategy.calculate_trend_strength(self.market_data['ohlcv']) if self.market_data else None
            }
            
            # 리스크 파라미터 조정
            self.risk_manager.adjust_risk_parameters(self.market_data['volatility'])
            
            self.logger.info("시장 데이터 업데이트 완료")
            
        except Exception as e:
            error_msg = f"시장 데이터 업데이트 실패: {str(e)}"
            self.logger.error(error_msg)
            self.database.save_log('ERROR', error_msg, 'trading_bot')
            
    async def _update_positions(self) -> None:
        """포지션 업데이트"""
        try:
            # 현재 포지션 조회
            positions = await self.database.get_positions()
            
            for position in positions:
                symbol = position['symbol']
                self.current_position = position
                
        except Exception as e:
            error_msg = f"포지션 업데이트 실패: {str(e)}"
            self.logger.error(error_msg)
            self.database.save_log('ERROR', error_msg, 'trading_bot')
            
    async def _generate_signal(self) -> Dict[str, Any]:
        """
        거래 신호 생성
        
        Returns:
            Dict[str, Any]: 거래 신호
        """
        try:
            signals = {}
            total_score = 0
            
            for timeframe in self.timeframes:
                # 각 시간대별 데이터 조회
                data = await self.database.get_market_data(timeframe)
                
                if data is not None:
                    # 기술적 지표 조회
                    indicators = await self.database.get_technical_indicators(timeframe)
                    
                    if indicators:
                        # 볼린저 밴드 + RSI 전략
                        signal = self._bollinger_rsi_strategy(data, indicators)
                        
                        # 가중치 적용
                        weighted_signal = signal * self.strategy_weights[timeframe]
                        signals[timeframe] = weighted_signal
                        total_score += weighted_signal
                        
            # 최종 신호 결정
            if total_score >= 0.5:
                return {'action': 'buy', 'score': total_score}
            elif total_score <= -0.5:
                return {'action': 'sell', 'score': total_score}
            else:
                return {'action': 'hold', 'score': total_score}
                
        except Exception as e:
            self.logger.error(f"거래 신호 생성 실패: {str(e)}")
            return {'action': 'hold', 'score': 0}
            
    def _bollinger_rsi_strategy(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, Any]
    ) -> float:
        """
        볼린저 밴드 + RSI 전략
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            indicators (Dict[str, Any]): 기술적 지표
            
        Returns:
            float: 거래 신호 (-1: 매도, 0: 홀딩, 1: 매수)
        """
        try:
            # 현재 가격
            current_price = data['close'].iloc[-1]
            
            # 볼린저 밴드
            bb = indicators['bollinger']
            upper_band = bb['upper'].iloc[-1]
            lower_band = bb['lower'].iloc[-1]
            
            # RSI
            rsi = indicators['rsi'].iloc[-1]
            
            # 매수 조건
            if current_price <= lower_band and rsi < 30:
                return 1.0
                
            # 매도 조건
            elif current_price >= upper_band and rsi > 70:
                return -1.0
                
            return 0.0
            
        except Exception as e:
            self.logger.error(f"볼린저 밴드 + RSI 전략 실패: {str(e)}")
            return 0.0
            
    async def _check_risk_limits(self) -> bool:
        """
        리스크 한도 확인
        
        Returns:
            bool: 거래 가능 여부
        """
        try:
            # 초기 자본 조회
            initial_capital = await self.database.get_initial_capital()
            
            # 현재 자본 조회
            current_capital = await self.database.get_current_capital()
            
            # 일일 손실 한도 확인
            daily_loss_limit = initial_capital * 0.05  # 5%
            daily_pnl = await self.database.get_daily_pnl()
            
            if daily_pnl <= -daily_loss_limit:
                self.logger.warning("일일 손실 한도 도달")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"리스크 한도 확인 실패: {str(e)}")
            return False
            
    async def _execute_trade(self, signal: Dict[str, Any]):
        """
        거래 실행
        
        Args:
            signal (Dict[str, Any]): 거래 신호
        """
        try:
            if signal['action'] == 'hold':
                return
                
            # 포지션 크기 계산
            position_size = await self._calculate_position_size()
            
            if position_size <= 0:
                return
                
            # 거래 실행
            trade = {
                'timestamp': datetime.now(),
                'action': signal['action'],
                'size': position_size,
                'price': await self.exchange.fetch_current_price(self.config['symbol']),
                'score': signal['score']
            }
            
            # 거래 기록 저장
            await self.database.save_trade(trade)
            
            # 포지션 업데이트
            await self._update_positions()
            
        except Exception as e:
            self.logger.error(f"거래 실행 실패: {str(e)}")
            
    async def _calculate_position_size(self) -> float:
        """
        포지션 크기 계산
        
        Returns:
            float: 포지션 크기
        """
        try:
            # 초기 자본 조회
            initial_capital = await self.database.get_initial_capital()
            
            # 포지션 크기 제한 (초기 자본의 5%)
            max_position_size = initial_capital * 0.05
            
            return max_position_size
            
        except Exception as e:
            self.logger.error(f"포지션 크기 계산 실패: {str(e)}")
            return 0.0
            
    async def _update_performance(self):
        """성과 업데이트"""
        try:
            # 현재 자본 조회
            current_capital = await self.database.get_current_capital()
            
            # 초기 자본 조회
            initial_capital = await self.database.get_initial_capital()
            
            # 수익률 계산
            returns = (current_capital - initial_capital) / initial_capital
            
            # 성과 기록 저장
            performance = {
                'timestamp': datetime.now(),
                'capital': current_capital,
                'returns': returns
            }
            
            await self.database.save_performance(performance)
            
        except Exception as e:
            self.logger.error(f"성과 업데이트 실패: {str(e)}")
            
    def get_status(self) -> Dict[str, Any]:
        """
        봇 상태 조회
        
        Returns:
            Dict[str, Any]: 봇 상태
        """
        try:
            return {
                'is_running': self.is_running,
                'current_position': self.current_position,
                'last_signal': self.last_signal,
                'market_data': {
                    'current_price': self.market_data['current_price'] if self.market_data else None,
                    'volatility': self.market_data['volatility'] if self.market_data else None,
                    'trend_strength': self.market_data['trend_strength'] if self.market_data else None
                },
                'risk_metrics': self.risk_manager.get_risk_metrics()
            }
            
        except Exception as e:
            error_msg = f"상태 조회 실패: {str(e)}"
            self.logger.error(error_msg)
            self.database.save_log('ERROR', error_msg, 'trading_bot')
            return {}
            
    def get_market_data(self) -> Dict[str, Any]:
        """시장 데이터 조회"""
        try:
            if not self.market_data:
                return {}
                
            # OHLCV 데이터를 DataFrame으로 변환
            df = pd.DataFrame(self.market_data['ohlcv'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return {
                'ohlcv': df,
                'current_price': self.market_data['current_price'],
                'volatility': self.market_data['volatility'],
                'trend_strength': self.market_data['trend_strength'],
                'indicators': self.strategy.calculate_indicators(df)
            }
            
        except Exception as e:
            self.logger.error(f"시장 데이터 조회 실패: {str(e)}")
            return {}
            
    def get_positions(self) -> List[Dict[str, Any]]:
        """포지션 정보 조회"""
        try:
            if not self.current_position:
                return []
                
            return [{
                'symbol': self.config['symbol'],
                'entry_price': self.current_position['entry_price'],
                'size': self.current_position['size'],
                'stop_loss': self.current_position['stop_loss'],
                'take_profit': self.current_position['take_profit'],
                'pnl': (self.market_data['current_price'] - self.current_position['entry_price']) * self.current_position['size']
            }]
            
        except Exception as e:
            self.logger.error(f"포지션 정보 조회 실패: {str(e)}")
            return []
            
    def get_trades(self) -> List[Dict[str, Any]]:
        """거래 내역 조회"""
        try:
            trades = self.database.get_trades(self.config['symbol'])
            return [{
                'timestamp': trade['timestamp'],
                'symbol': trade['symbol'],
                'side': trade['side'],
                'price': trade['price'],
                'size': trade['size'],
                'pnl': trade.get('pnl', 0.0)
            } for trade in trades]
            
        except Exception as e:
            self.logger.error(f"거래 내역 조회 실패: {str(e)}")
            return [] 