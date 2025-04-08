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
import logging
import numpy as np

# 프로젝트 루트 경로를 시스템 경로에 추가
root_path = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(root_path))

# 지연 임포트 적용
def get_exchange():
    from src.exchange.binance_exchange import BinanceExchange
    return BinanceExchange()

def get_logger():
    from src.utils.logger import setup_logger
    return setup_logger('trading_bot')

def get_news_analyzer():
    from src.analysis.news_analyzer import NewsAnalyzer
    return NewsAnalyzer()

def get_technical_analyzer():
    from src.analysis.technical_analyzer import TechnicalAnalyzer
    return TechnicalAnalyzer()

def get_integrated_strategy():
    from src.strategy.integrated_strategy import IntegratedStrategy
    return IntegratedStrategy()

def get_risk_manager():
    from src.risk.risk_manager import RiskManager
    return RiskManager()

def get_database():
    from src.database.database import Database
    return Database()

def get_database_manager():
    from src.utils.database import DatabaseManager
    return DatabaseManager()

def get_telegram_notifier():
    from src.utils.telegram import TelegramNotifier
    return TelegramNotifier()

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
        self.database = get_database_manager()
        self.exchange = get_exchange()
        self.strategy = get_integrated_strategy()
        self.risk_manager = get_risk_manager()
        
        # 텔레그램 알림 설정
        bot_token = config.get('telegram_bot_token')
        chat_id = config.get('telegram_chat_id')
        if bot_token and chat_id:
            self.telegram = get_telegram_notifier()
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
        
        # API 관리자 초기화
        self.api_manager = APIManager(config)
        
        # 다중 전략 관리자 초기화
        self.strategy_manager = MultiStrategyManager(
            initial_capital=config.get('initial_capital', 10000),
            max_strategies=config.get('max_strategies', 3),
            rebalance_threshold=config.get('rebalance_threshold', 0.2),
            min_capital_per_strategy=config.get('min_capital_per_strategy', 1000)
        )
        
        # 상태 변수
        self.positions = []
        self.trades = []
        
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
            
            # 초기 데이터 로드
            await self._load_initial_data()
            
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
                    
                    # 포지션 관리
                    await self._manage_positions()
                    
                    # 리스크 체크
                    await self._check_risk()
                    
                    # 전략 실행
                    await self._execute_strategies()
                    
                    # 알림 전송
                    await self._send_notifications()
                    
                    # 데이터베이스 업데이트
                    await self._update_database()
                    
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
            
    async def _load_initial_data(self):
        """초기 데이터 로드"""
        try:
            # 시장 데이터 로드
            for symbol in self.config.get('symbols', []):
                market_data = await self.api_manager.get_market_data(symbol)
                if market_data:
                    self.market_data = market_data
            
            # 포지션 로드
            self.positions = await self.database.get_positions()
            
            # 거래 내역 로드
            self.trades = await self.database.get_trades()
            
            # 전략 초기화
            await self.strategy_manager.initialize(self.market_data)
            
        except Exception as e:
            self.logger.error(f"초기 데이터 로드 중 오류 발생: {str(e)}")
            
    async def _update_market_data(self) -> None:
        """시장 데이터 업데이트"""
        try:
            for symbol in self.config.get('symbols', []):
                # OHLCV 데이터
                market_data = await self.api_manager.get_market_data(symbol)
                if market_data:
                    self.market_data = market_data
                
                # 호가 데이터
                orderbook = await self.api_manager.get_order_book(symbol)
                if orderbook:
                    self.market_data.append(orderbook)
                
                # 자금 조달 비율
                funding_rate = await self.api_manager.get_funding_rate(symbol)
                if funding_rate:
                    self.market_data.append(funding_rate)
                
                # 미체결약정
                open_interest = await self.api_manager.get_open_interest(symbol)
                if open_interest:
                    self.market_data.append(open_interest)
                
                # 청산 데이터
                liquidation = await self.api_manager.get_liquidation(symbol)
                if liquidation:
                    self.market_data.append(liquidation)
                
                # 시장 감성 분석
                sentiment = await self.api_manager.get_market_sentiment(symbol)
                if sentiment:
                    self.market_data.append(sentiment)
            
            self.last_update = datetime.now()
            
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
            
    async def _manage_positions(self):
        """포지션 관리"""
        try:
            for position in self.positions:
                # 포지션 업데이트
                symbol = position['symbol']
                current_price = self.market_data[symbol][-1]['close']
                
                # 손익 계산
                pnl = (current_price - position['entry_price']) * position['size']
                position['unrealized_pnl'] = pnl
                position['current_price'] = current_price
                
                # 리스크 체크
                risk_metrics = self.risk_manager.calculate_risk_metrics(position)
                if self.risk_manager.check_risk_limits(risk_metrics):
                    # 리스크 한도 초과 시 포지션 청산
                    await self._close_position(position)
                
                # 스탑로스/익절 체크
                if current_price <= position['stop_loss']:
                    await self._close_position(position)
                elif current_price >= position['take_profit']:
                    await self._close_position(position)
            
        except Exception as e:
            self.logger.error(f"포지션 관리 중 오류 발생: {str(e)}")
            
    async def _check_risk(self):
        """리스크 체크"""
        try:
            # 포트폴리오 리스크 체크
            portfolio_risk = self.risk_manager.calculate_portfolio_risk(self.positions)
            if self.risk_manager.check_portfolio_risk(portfolio_risk):
                # 리스크 한도 초과 시 알림 전송
                await self.telegram.send_risk_alert(portfolio_risk)
            
            # 변동성 체크
            for symbol in self.config.get('symbols', []):
                volatility = self.risk_manager.calculate_volatility(self.market_data[symbol])
                if volatility > self.config.get('volatility_threshold', 0.05):
                    # 변동성 한도 초과 시 알림 전송
                    await self.telegram.send_volatility_alert(symbol, volatility)
            
        except Exception as e:
            self.logger.error(f"리스크 체크 중 오류 발생: {str(e)}")
            
    async def _execute_strategies(self):
        """전략 실행"""
        try:
            # 전략 신호 생성
            signals = await self.strategy_manager.generate_signals(self.market_data)
            
            for signal in signals:
                # 리스크 평가
                if not await self._evaluate_risk(signal):
                    continue
                
                # 포지션 진입
                await self._enter_position(signal)
            
        except Exception as e:
            self.logger.error(f"전략 실행 중 오류 발생: {str(e)}")
    
    async def _evaluate_risk(self, signal: Dict[str, Any]) -> bool:
        """
        리스크 평가
        
        Args:
            signal: 거래 신호
            
        Returns:
            bool: 거래 가능 여부
        """
        try:
            # 포지션 수 체크
            if len(self.positions) >= self.config.get('max_positions', 5):
                return False
            
            # 자본금 체크
            if self.risk_manager.capital < self.config.get('min_capital', 1000):
                return False
            
            # 리스크 한도 체크
            position_size = self.risk_manager.calculate_position_size(
                signal['entry_price'],
                signal['stop_loss']
            )
            if position_size > self.config.get('max_position_size', 0.1):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"리스크 평가 중 오류 발생: {str(e)}")
            return False
    
    async def _enter_position(self, signal: Dict[str, Any]):
        """
        포지션 진입
        
        Args:
            signal: 거래 신호
        """
        try:
            # 포지션 크기 계산
            position_size = self.risk_manager.calculate_position_size(
                signal['entry_price'],
                signal['stop_loss']
            )
            
            # 포지션 생성
            position = {
                'symbol': signal['symbol'],
                'side': signal['side'],
                'entry_price': signal['entry_price'],
                'size': position_size,
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'entry_time': datetime.now(),
                'strategy': signal['strategy']
            }
            
            # 포지션 추가
            self.positions.append(position)
            
            # 거래 기록
            trade = {
                'symbol': signal['symbol'],
                'side': signal['side'],
                'entry_price': signal['entry_price'],
                'size': position_size,
                'entry_time': datetime.now(),
                'strategy': signal['strategy']
            }
            self.trades.append(trade)
            
            # 알림 전송
            await self.telegram.send_trade_alert(trade)
            
        except Exception as e:
            self.logger.error(f"포지션 진입 중 오류 발생: {str(e)}")
    
    async def _close_position(self, position: Dict[str, Any]):
        """
        포지션 청산
        
        Args:
            position: 포지션 정보
        """
        try:
            # 포지션 제거
            self.positions.remove(position)
            
            # 거래 기록 업데이트
            for trade in self.trades:
                if (trade['symbol'] == position['symbol'] and
                    trade['entry_time'] == position['entry_time']):
                    trade['exit_price'] = position['current_price']
                    trade['exit_time'] = datetime.now()
                    trade['pnl'] = position['unrealized_pnl']
                    break
            
            # 알림 전송
            await self.telegram.send_position_closed_alert(position)
            
        except Exception as e:
            self.logger.error(f"포지션 청산 중 오류 발생: {str(e)}")
    
    async def _send_notifications(self):
        """알림 전송"""
        try:
            # 일일 리포트
            if datetime.now().hour == 0 and datetime.now().minute == 0:
                await self.telegram.send_daily_report({
                    'positions': self.positions,
                    'trades': self.trades,
                    'performance': self.strategy_manager.get_portfolio_metrics()
                })
            
            # 리스크 알림
            for position in self.positions:
                risk_metrics = self.risk_manager.calculate_risk_metrics(position)
                if self.risk_manager.check_risk_limits(risk_metrics):
                    await self.telegram.send_risk_alert(risk_metrics)
            
        except Exception as e:
            self.logger.error(f"알림 전송 중 오류 발생: {str(e)}")
    
    async def _update_database(self):
        """데이터베이스 업데이트"""
        try:
            # 포지션 업데이트
            await self.database.update_positions(self.positions)
            
            # 거래 내역 업데이트
            await self.database.update_trades(self.trades)
            
            # 시장 데이터 업데이트
            await self.database.update_market_data(self.market_data)
            
        except Exception as e:
            self.logger.error(f"데이터베이스 업데이트 중 오류 발생: {str(e)}")
            
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
            
    async def get_portfolio_metrics(self) -> Dict[str, Any]:
        """
        포트폴리오 지표 조회
        
        Returns:
            Dict[str, Any]: 포트폴리오 지표
        """
        return self.strategy_manager.get_portfolio_metrics() 