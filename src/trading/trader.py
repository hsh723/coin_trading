"""
실시간 트레이딩 시스템 모듈
"""

import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import threading
import time
import logging
from datetime import datetime, timedelta
from queue import Queue
from src.utils.logger import get_logger
from src.data.collector import DataCollector
from src.risk.risk_manager import RiskManager
from src.strategy.base_strategy import BaseStrategy

class Trader:
    """실시간 트레이딩 클래스"""
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        strategy: BaseStrategy,
        risk_manager: RiskManager,
        symbol: str = "BTC/USDT",
        exchange: str = "binance"
    ):
        """
        트레이더 초기화
        
        Args:
            api_key (str): API 키
            api_secret (str): API 시크릿
            strategy (BaseStrategy): 트레이딩 전략
            risk_manager (RiskManager): 리스크 관리자
            symbol (str): 거래 심볼
            exchange (str): 거래소
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.symbol = symbol
        self.exchange = exchange
        
        # 로거 초기화
        self.logger = get_logger(__name__)
        
        # 데이터 수집기 초기화
        self.data_collector = DataCollector(
            api_key=api_key,
            api_secret=api_secret,
            symbols=[symbol],
            exchange_id=exchange
        )
        
        # 거래소 초기화
        self.exchange = self._init_exchange()
        
        # 상태 변수
        self._state = {
            'position': 0.0,
            'entry_price': 0.0,
            'last_trade': None,
            'trades': []
        }
        
        # 큐 초기화
        self.tick_queue = Queue()
        self.order_queue = Queue()
        
        # 스레드 초기화
        self.tick_thread = None
        self.order_thread = None
        self.monitor_thread = None
        
    def _init_exchange(self) -> ccxt.Exchange:
        """
        거래소 초기화
        
        Returns:
            ccxt.Exchange: 거래소 객체
        """
        try:
            exchange_class = getattr(ccxt, self.exchange)
            exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True
            })
            return exchange
            
        except Exception as e:
            self.logger.error(f"거래소 초기화 실패: {str(e)}")
            raise
            
    def start_trading(self, mode: str = 'live', test_mode: bool = False):
        """
        트레이딩 시작
        
        Args:
            mode (str): 트레이딩 모드 ('live' 또는 'paper')
            test_mode (bool): 테스트 모드 여부
        """
        try:
            self.logger.info(f"트레이딩 시작: mode={mode}, test_mode={test_mode}")
            
            # WebSocket 스트림 시작
            self.data_collector.start_websocket_stream(self._handle_tick_data)
            
            # 스레드 시작
            self.tick_thread = threading.Thread(target=self._process_tick_data)
            self.order_thread = threading.Thread(target=self._process_orders)
            self.monitor_thread = threading.Thread(target=self._monitor_system)
            
            self.tick_thread.start()
            self.order_thread.start()
            self.monitor_thread.start()
            
        except Exception as e:
            self.logger.error(f"트레이딩 시작 실패: {str(e)}")
            raise
            
    def stop_trading(self):
        """트레이딩 중지"""
        try:
            self.logger.info("트레이딩 중지")
            
            # WebSocket 스트림 중지
            self.data_collector.stop_websocket_stream()
            
            # 스레드 중지
            if self.tick_thread:
                self.tick_thread.join()
            if self.order_thread:
                self.order_thread.join()
            if self.monitor_thread:
                self.monitor_thread.join()
                
        except Exception as e:
            self.logger.error(f"트레이딩 중지 실패: {str(e)}")
            raise
            
    def _handle_tick_data(self, tick_data: Dict[str, Any]):
        """
        틱 데이터 처리
        
        Args:
            tick_data (Dict[str, Any]): 틱 데이터
        """
        try:
            self.tick_queue.put(tick_data)
            
        except Exception as e:
            self.logger.error(f"틱 데이터 처리 실패: {str(e)}")
            
    def _process_tick_data(self):
        """틱 데이터 처리 스레드"""
        while True:
            try:
                if not self.tick_queue.empty():
                    tick_data = self.tick_queue.get()
                    self.process_tick_data(tick_data)
                    
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"틱 데이터 처리 스레드 오류: {str(e)}")
                time.sleep(1)
                
    def process_tick_data(self, tick_data: Dict[str, Any]):
        """
        틱 데이터 처리
        
        Args:
            tick_data (Dict[str, Any]): 틱 데이터
        """
        try:
            # 데이터 전처리
            df = pd.DataFrame([tick_data])
            df = self.data_collector.processor.preprocess_data(df)
            
            # 전략 분석
            signal = self.strategy.analyze(df)
            
            # 신호 처리
            if signal:
                self.execute_signal(signal, tick_data['symbol'])
                
        except Exception as e:
            self.logger.error(f"틱 데이터 처리 실패: {str(e)}")
            
    def execute_signal(self, signal: Dict[str, Any], symbol: str):
        """
        매매 신호 실행
        
        Args:
            signal (Dict[str, Any]): 매매 신호
            symbol (str): 거래 심볼
        """
        try:
            # 포지션 사이즈 계산
            position_size = self.calculate_position_size(signal, symbol)
            
            # 주문 생성
            order = self.create_order(
                order_type=signal['order_type'],
                side=signal['side'],
                symbol=symbol,
                amount=position_size,
                price=signal.get('price')
            )
            
            # 주문 큐에 추가
            self.order_queue.put(order)
            
        except Exception as e:
            self.logger.error(f"매매 신호 실행 실패: {str(e)}")
            
    def create_order(
        self,
        order_type: str,
        side: str,
        symbol: str,
        amount: float,
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        주문 생성
        
        Args:
            order_type (str): 주문 유형
            side (str): 매수/매도
            symbol (str): 거래 심볼
            amount (float): 수량
            price (Optional[float]): 가격
            
        Returns:
            Dict[str, Any]: 주문 정보
        """
        try:
            order = {
                'type': order_type,
                'side': side,
                'symbol': symbol,
                'amount': amount,
                'price': price,
                'timestamp': datetime.now()
            }
            
            return order
            
        except Exception as e:
            self.logger.error(f"주문 생성 실패: {str(e)}")
            raise
            
    def _process_orders(self):
        """주문 처리 스레드"""
        while True:
            try:
                if not self.order_queue.empty():
                    order = self.order_queue.get()
                    self._submit_order(order)
                    
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"주문 처리 스레드 오류: {str(e)}")
                time.sleep(1)
                
    def _submit_order(self, order: Dict[str, Any]):
        """
        주문 제출
        
        Args:
            order (Dict[str, Any]): 주문 정보
        """
        try:
            # 주문 제출
            response = self.exchange.create_order(
                symbol=order['symbol'],
                type=order['type'],
                side=order['side'],
                amount=order['amount'],
                price=order['price']
            )
            
            # 주문 상태 업데이트
            self._state['trades'].append(response)
            self._state['last_trade'] = response
            
            # 주문 모니터링 시작
            self._monitor_order(response['id'])
            
        except Exception as e:
            self.logger.error(f"주문 제출 실패: {str(e)}")
            
    def _monitor_order(self, order_id: str):
        """
        주문 모니터링
        
        Args:
            order_id (str): 주문 ID
        """
        try:
            while True:
                # 주문 상태 확인
                order = self.exchange.fetch_order(order_id)
                
                if order['status'] == 'closed':
                    # 주문 완료 처리
                    self.handle_filled_order(order)
                    break
                    
                elif order['status'] == 'canceled':
                    # 주문 취소 처리
                    self.logger.info(f"주문 취소됨: {order_id}")
                    break
                    
                time.sleep(1)
                
        except Exception as e:
            self.logger.error(f"주문 모니터링 실패: {str(e)}")
            
    def handle_filled_order(self, order_info: Dict[str, Any]):
        """
        체결된 주문 처리
        
        Args:
            order_info (Dict[str, Any]): 주문 정보
        """
        try:
            # 포지션 업데이트
            self.update_positions(order_info)
            
            # 성과 기록
            self.log_performance(order_info)
            
        except Exception as e:
            self.logger.error(f"체결된 주문 처리 실패: {str(e)}")
            
    def update_positions(self, order_info: Dict[str, Any]):
        """
        포지션 업데이트
        
        Args:
            order_info (Dict[str, Any]): 주문 정보
        """
        try:
            symbol = order_info['symbol']
            side = order_info['side']
            amount = order_info['amount']
            
            if symbol not in self._state:
                self._state[symbol] = {
                    'amount': 0,
                    'entry_price': 0,
                    'unrealized_pnl': 0
                }
                
            position = self._state[symbol]
            
            if side == 'buy':
                position['amount'] += amount
                position['entry_price'] = (
                    (position['entry_price'] * (position['amount'] - amount) +
                     order_info['price'] * amount) / position['amount']
                )
                
            elif side == 'sell':
                position['amount'] -= amount
                
            # 미실현 손익 계산
            if position['amount'] != 0:
                current_price = self.exchange.fetch_ticker(symbol)['last']
                position['unrealized_pnl'] = (
                    (current_price - position['entry_price']) * position['amount']
                )
                
        except Exception as e:
            self.logger.error(f"포지션 업데이트 실패: {str(e)}")
            
    def calculate_position_size(
        self,
        signal: Dict[str, Any],
        symbol: str
    ) -> float:
        """
        포지션 사이즈 계산
        
        Args:
            signal (Dict[str, Any]): 매매 신호
            symbol (str): 거래 심볼
            
        Returns:
            float: 포지션 사이즈
        """
        try:
            # 리스크 관리자로부터 포지션 사이즈 계산
            position_size = self.risk_manager.calculate_position_size(
                symbol=symbol,
                signal=signal,
                account_balance=self.exchange.fetch_balance()['total']['USDT']
            )
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"포지션 사이즈 계산 실패: {str(e)}")
            raise
            
    def log_performance(self, order_info: Dict[str, Any]):
        """
        성과 기록
        
        Args:
            order_info (Dict[str, Any]): 주문 정보
        """
        try:
            # 거래 횟수 업데이트
            self._state['trades'].append(order_info)
            
            # 수익/손실 업데이트
            pnl = order_info.get('pnl', 0)
            self._state['total_pnl'] += pnl
            self._state['daily_pnl'] += pnl
            
            if pnl > 0:
                self._state['winning_trades'] += 1
                self._state['consecutive_losses'] = 0
            else:
                self._state['losing_trades'] += 1
                self._state['consecutive_losses'] += 1
                
            # 최대 손실폭 업데이트
            if pnl < self._state['max_drawdown']:
                self._state['max_drawdown'] = pnl
                
            # 성과 로깅
            self.logger.info(f"성과 업데이트: {self._state}")
            
        except Exception as e:
            self.logger.error(f"성과 기록 실패: {str(e)}")
            
    def _monitor_system(self):
        """시스템 모니터링 스레드"""
        while True:
            try:
                # 일일 손실 한도 확인
                if self._state['daily_pnl'] < -self.risk_manager.max_daily_loss:
                    self.emergency_stop("일일 손실 한도 도달")
                    
                # 연속 손실 확인
                if self._state['consecutive_losses'] >= self.risk_manager.max_consecutive_losses:
                    self.emergency_stop("연속 손실 한도 도달")
                    
                # API 통신 상태 확인
                if not self._check_api_connection():
                    self.emergency_stop("API 통신 오류")
                    
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"시스템 모니터링 스레드 오류: {str(e)}")
                time.sleep(1)
                
    def _check_api_connection(self) -> bool:
        """
        API 통신 상태 확인
        
        Returns:
            bool: 통신 상태
        """
        try:
            self.exchange.fetch_time()
            return True
            
        except Exception as e:
            self.logger.error(f"API 통신 상태 확인 실패: {str(e)}")
            return False
            
    def emergency_stop(self, reason: str):
        """
        비상 정지
        
        Args:
            reason (str): 정지 사유
        """
        try:
            self.logger.error(f"비상 정지: {reason}")
            
            # 모든 포지션 청산
            for symbol, position in self._state.items():
                if isinstance(position, dict) and 'amount' in position and position['amount'] != 0:
                    self.create_order(
                        order_type='market',
                        side='sell' if position['amount'] > 0 else 'buy',
                        symbol=symbol,
                        amount=abs(position['amount'])
                    )
                    
            # 트레이딩 중지
            self.stop_trading()
            
        except Exception as e:
            self.logger.error(f"비상 정지 실패: {str(e)}")
            raise

    def initialize(self, data: pd.DataFrame) -> None:
        """
        초기화
        
        Args:
            data (pd.DataFrame): 초기 데이터
        """
        self.strategy.initialize(data)
        self._state['position'] = 0.0
        self._state['entry_price'] = 0.0
        self._state['last_trade'] = None
        self._state['trades'] = []
        
    def execute_trade(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        거래 실행
        
        Args:
            data (pd.DataFrame): 시장 데이터
            
        Returns:
            Dict[str, Any]: 거래 결과
        """
        # 거래 가능 상태 확인
        if not self.risk_manager.check_trading_status():
            return {'action': 'hold', 'amount': 0.0}
            
        # 전략 실행
        signals = self.strategy.execute(data, self._state['position'])
        
        if signals['action'] == 'buy':
            # 매수 가능 금액 계산
            price = data['close'].iloc[-1]
            risk_per_trade = 0.01  # 1%
            amount = self.risk_manager.calculate_position_size(price, risk_per_trade)
            
            # 거래 기록
            trade = {
                'timestamp': data.index[-1],
                'action': 'buy',
                'price': price,
                'amount': amount
            }
            self._state['trades'].append(trade)
            self._state['last_trade'] = trade
            self._state['position'] = amount
            self._state['entry_price'] = price
            
            return {'action': 'buy', 'amount': amount}
            
        elif signals['action'] == 'sell':
            # 매도 기록
            price = data['close'].iloc[-1]
            amount = self._state['position']
            
            trade = {
                'timestamp': data.index[-1],
                'action': 'sell',
                'price': price,
                'amount': amount
            }
            self._state['trades'].append(trade)
            self._state['last_trade'] = trade
            
            # 손익 계산 및 리스크 지표 업데이트
            pnl = (price - self._state['entry_price']) * amount
            self.risk_manager.update_risk_metrics(pnl)
            
            # 포지션 초기화
            self._state['position'] = 0.0
            self._state['entry_price'] = 0.0
            
            return {'action': 'sell', 'amount': amount}
            
        return {'action': 'hold', 'amount': 0.0}
        
    def update_state(self, data: pd.DataFrame) -> None:
        """
        상태 업데이트
        
        Args:
            data (pd.DataFrame): 시장 데이터
        """
        self.strategy.update(data)
        
        # 트레일링 스탑 업데이트
        if self._state['position'] > 0:
            stop_price = self.risk_manager.update_trailing_stop(
                self.symbol,
                data['close'].iloc[-1]
            )
            
            if stop_price is not None and data['close'].iloc[-1] < stop_price:
                self.close_position(data)
                
    def close_position(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        포지션 청산
        
        Args:
            data (pd.DataFrame): 시장 데이터
            
        Returns:
            Dict[str, Any]: 청산 결과
        """
        if self._state['position'] == 0:
            return {'action': 'hold', 'amount': 0.0}
            
        price = data['close'].iloc[-1]
        amount = self._state['position']
        
        trade = {
            'timestamp': data.index[-1],
            'action': 'sell',
            'price': price,
            'amount': amount
        }
        self._state['trades'].append(trade)
        self._state['last_trade'] = trade
        
        # 손익 계산 및 리스크 지표 업데이트
        pnl = (price - self._state['entry_price']) * amount
        self.risk_manager.update_risk_metrics(pnl)
        
        # 포지션 초기화
        self._state['position'] = 0.0
        self._state['entry_price'] = 0.0
        
        return {'action': 'sell', 'amount': amount}
        
    def get_state(self) -> Dict[str, Any]:
        """
        상태 조회
        
        Returns:
            Dict[str, Any]: 현재 상태
        """
        return {
            'position': self._state['position'],
            'entry_price': self._state['entry_price'],
            'last_trade': self._state['last_trade'],
            'trades': self._state['trades'],
            'risk_metrics': self.risk_manager.get_risk_metrics()
        } 