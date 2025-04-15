import numpy as np
import pandas as pd
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
import queue
import threading
import time

logger = logging.getLogger(__name__)

class TradingEngine:
    """
    실시간 트레이딩 엔진
    
    주요 기능:
    - 실시간 전략 실행
    - 손절/익절 관리
    - 수익률 집계
    - 리스크 관리
    """
    
    def __init__(self,
                 strategy,
                 exchange_api,
                 initial_capital: float = 10000.0,
                 max_position_size: float = 0.1,
                 stop_loss: float = 0.02,
                 take_profit: float = 0.03,
                 max_daily_loss: float = 0.05,
                 volatility_threshold: float = 0.05):
        """
        트레이딩 엔진 초기화
        
        Args:
            strategy: 트레이딩 전략 객체
            exchange_api: 거래소 API 객체
            initial_capital: 초기 자본금
            max_position_size: 최대 포지션 크기
            stop_loss: 손절 비율
            take_profit: 익절 비율
            max_daily_loss: 일일 최대 손실 비율
            volatility_threshold: 변동성 임계값
        """
        self.strategy = strategy
        self.exchange_api = exchange_api
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_daily_loss = max_daily_loss
        self.volatility_threshold = volatility_threshold
        
        # 트레이딩 상태 변수
        self.running = False
        self.current_position = 0.0
        self.entry_price = 0.0
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.trades = []
        self.risk_triggered = False
        
        # 데이터 큐
        self.data_queue = queue.Queue()
        
        # 성과 지표
        self.metrics = {
            'daily_return': 0.0,
            'total_return': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
    
    def start(self):
        """트레이딩 엔진 시작"""
        logger.info("트레이딩 엔진 시작 중...")
        self.running = True
        
        # 데이터 수집 스레드 시작
        self.data_thread = threading.Thread(target=self._collect_data)
        self.data_thread.start()
        
        # 트레이딩 스레드 시작
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.start()
    
    def stop(self):
        """트레이딩 엔진 중지"""
        logger.info("트레이딩 엔진 중지 중...")
        self.running = False
        
        # 스레드 종료 대기
        if hasattr(self, 'data_thread'):
            self.data_thread.join()
        if hasattr(self, 'trading_thread'):
            self.trading_thread.join()
    
    def _collect_data(self):
        """실시간 데이터 수집"""
        while self.running:
            try:
                # 현재 가격 및 변동성 데이터 수집
                current_price = self.exchange_api.get_current_price()
                volatility = self.exchange_api.get_volatility()
                
                # 데이터 큐에 추가
                self.data_queue.put({
                    'timestamp': datetime.now(),
                    'price': current_price,
                    'volatility': volatility
                })
                
                # 업데이트 간격 대기
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"데이터 수집 중 오류 발생: {e}")
    
    def _trading_loop(self):
        """트레이딩 실행 루프"""
        while self.running:
            try:
                # 데이터 큐에서 새로운 데이터 가져오기
                while not self.data_queue.empty():
                    data = self.data_queue.get()
                    
                    # 리스크 체크
                    if self._check_risk(data):
                        self._handle_risk()
                        continue
                    
                    # 전략 신호 확인
                    signal = self.strategy.get_signal(data)
                    
                    # 포지션 관리
                    if signal != 0 and self.current_position == 0:
                        self._open_position(signal, data['price'])
                    elif self.current_position != 0:
                        self._manage_position(data['price'])
                    
                    # 성과 지표 업데이트
                    self._update_metrics()
                
                # 업데이트 간격 대기
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"트레이딩 루프 중 오류 발생: {e}")
    
    def _check_risk(self, data: Dict[str, Any]) -> bool:
        """리스크 체크"""
        # 변동성 체크
        if data['volatility'] > self.volatility_threshold:
            logger.warning(f"변동성 임계값 초과: {data['volatility']}")
            return True
        
        # 일일 손실 체크
        if self.daily_pnl < -self.max_daily_loss * self.initial_capital:
            logger.warning(f"일일 손실 한도 초과: {self.daily_pnl}")
            return True
        
        return False
    
    def _handle_risk(self):
        """리스크 대응"""
        if self.current_position != 0:
            # 포지션 청산
            self._close_position(self.exchange_api.get_current_price())
            self.risk_triggered = True
            logger.info("리스크 트리거로 인한 포지션 청산")
    
    def _open_position(self, signal: int, price: float):
        """포지션 진입"""
        try:
            # 포지션 크기 계산
            position_size = min(
                self.max_position_size,
                self.initial_capital / price
            ) * signal
            
            # 주문 실행
            order = self.exchange_api.place_order(
                symbol='BTC/USDT',
                side='buy' if signal > 0 else 'sell',
                type='market',
                amount=abs(position_size)
            )
            
            # 포지션 상태 업데이트
            self.current_position = position_size
            self.entry_price = price
            
            # 거래 기록
            self.trades.append({
                'timestamp': datetime.now(),
                'type': 'open',
                'side': 'buy' if signal > 0 else 'sell',
                'size': abs(position_size),
                'price': price
            })
            
            logger.info(f"포지션 진입: {position_size} @ {price}")
            
        except Exception as e:
            logger.error(f"포지션 진입 중 오류 발생: {e}")
    
    def _manage_position(self, current_price: float):
        """포지션 관리"""
        # 손익 계산
        pnl = (current_price - self.entry_price) * self.current_position
        
        # 손절 체크
        if pnl < -self.stop_loss * abs(self.current_position * self.entry_price):
            self._close_position(current_price)
            logger.info(f"손절 실행: {pnl}")
            return
        
        # 익절 체크
        if pnl > self.take_profit * abs(self.current_position * self.entry_price):
            self._close_position(current_price)
            logger.info(f"익절 실행: {pnl}")
            return
    
    def _close_position(self, price: float):
        """포지션 청산"""
        try:
            # 주문 실행
            order = self.exchange_api.place_order(
                symbol='BTC/USDT',
                side='sell' if self.current_position > 0 else 'buy',
                type='market',
                amount=abs(self.current_position)
            )
            
            # 손익 계산
            pnl = (price - self.entry_price) * self.current_position
            self.daily_pnl += pnl
            self.total_pnl += pnl
            
            # 거래 기록
            self.trades.append({
                'timestamp': datetime.now(),
                'type': 'close',
                'side': 'sell' if self.current_position > 0 else 'buy',
                'size': abs(self.current_position),
                'price': price,
                'pnl': pnl
            })
            
            # 포지션 상태 초기화
            self.current_position = 0.0
            self.entry_price = 0.0
            
            logger.info(f"포지션 청산: {pnl}")
            
        except Exception as e:
            logger.error(f"포지션 청산 중 오류 발생: {e}")
    
    def _update_metrics(self):
        """성과 지표 업데이트"""
        if not self.trades:
            return
        
        # 수익률 계산
        self.metrics['daily_return'] = self.daily_pnl / self.initial_capital
        self.metrics['total_return'] = self.total_pnl / self.initial_capital
        
        # 승률 계산
        winning_trades = len([t for t in self.trades if t.get('pnl', 0) > 0])
        total_trades = len([t for t in self.trades if t.get('pnl') is not None])
        if total_trades > 0:
            self.metrics['win_rate'] = winning_trades / total_trades
        
        # 샤프 비율 계산 (단순화된 버전)
        returns = [t.get('pnl', 0) / self.initial_capital for t in self.trades if t.get('pnl') is not None]
        if returns:
            self.metrics['sharpe_ratio'] = np.mean(returns) / (np.std(returns) + 1e-6)
        
        # 최대 낙폭 계산
        cumulative_returns = np.cumsum(returns)
        self.metrics['max_drawdown'] = np.min(cumulative_returns - np.maximum.accumulate(cumulative_returns))
    
    def get_metrics(self) -> Dict[str, float]:
        """성과 지표 반환"""
        return self.metrics 