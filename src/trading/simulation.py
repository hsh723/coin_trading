"""
트레이딩 시뮬레이션 모듈
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
import os
from decimal import Decimal, ROUND_DOWN
import asyncio
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from collections import deque
import gc

# 로거 설정
logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """시뮬레이션 설정"""
    initial_capital: float
    slippage: float = 0.001  # 0.1% 기본 슬리피지
    partial_fill_probability: float = 0.3  # 30% 확률로 부분 체결
    market_volatility: float = 1.0  # 시장 변동성 (1.0: 정상, >1.0: 변동성 증가)
    max_positions: int = 5  # 최대 동시 포지션 수
    position_size_limit: float = 0.1  # 최대 포지션 크기 (자본의 10%)
    max_daily_loss: float = 0.05  # 최대 일일 손실 (자본의 5%)
    memory_limit: int = 1000  # 메모리 제한 (거래 기록 최대 개수)

class TradingSimulator:
    def __init__(self, config: SimulationConfig):
        """
        트레이딩 시뮬레이터 초기화
        
        Args:
            config: 시뮬레이션 설정
        """
        self.config = config
        self.balance = Decimal(str(config.initial_capital))
        self.positions: Dict[str, Dict] = {}
        self.trade_history = deque(maxlen=config.memory_limit)  # 메모리 제한 적용
        self.daily_pnl = Decimal('0')
        self.last_reset = datetime.now().date()
        self._setup_data_storage()
        
    def _setup_data_storage(self):
        """데이터 저장소 설정"""
        self.data_dir = "data_storage"
        os.makedirs(self.data_dir, exist_ok=True)
        
    async def execute_order(self, symbol: str, side: str, amount: float, price: float) -> bool:
        """
        주문 실행 시뮬레이션
        
        Args:
            symbol: 거래 심볼
            side: 매수/매도 방향
            amount: 주문 수량
            price: 주문 가격
            
        Returns:
            bool: 주문 성공 여부
        """
        try:
            # 리스크 관리 검사
            if not self._check_risk_limits(symbol, side, amount, price):
                return False
                
            # 슬리피지 적용
            executed_price = self._apply_slippage(price)
            
            # 부분 체결 시뮬레이션
            if np.random.random() < self.config.partial_fill_probability:
                amount = amount * np.random.uniform(0.5, 0.9)
                
            # 주문 실행
            if side == "buy":
                success = await self._execute_buy(symbol, amount, executed_price)
            else:
                success = await self._execute_sell(symbol, amount, executed_price)
                
            if success:
                # 일일 손익 업데이트
                self._update_daily_pnl()
                
                # 메모리 관리
                self._manage_memory()
                
            return success
            
        except Exception as e:
            logger.error(f"주문 실행 중 오류 발생: {str(e)}")
            return False
            
    def _check_risk_limits(self, symbol: str, side: str, amount: float, price: float) -> bool:
        """리스크 한도 검사"""
        # 최대 포지션 수 제한
        if len(self.positions) >= self.config.max_positions and symbol not in self.positions:
            logger.warning("최대 포지션 수 초과")
            return False
            
        # 포지션 크기 제한
        position_value = amount * price
        if position_value > self.balance * self.config.position_size_limit:
            logger.warning("포지션 크기 제한 초과")
            return False
            
        # 일일 손실 제한
        if self.daily_pnl < -self.balance * self.config.max_daily_loss:
            logger.warning("일일 손실 제한 초과")
            return False
            
        return True
        
    def _apply_slippage(self, price: float) -> float:
        """슬리피지 적용"""
        slippage = np.random.normal(0, self.config.slippage)
        return price * (1 + slippage)
        
    def _update_daily_pnl(self):
        """일일 손익 업데이트"""
        current_date = datetime.now().date()
        if current_date > self.last_reset:
            self.daily_pnl = Decimal('0')
            self.last_reset = current_date
            
    def _manage_memory(self):
        """메모리 관리"""
        # 오래된 거래 기록 정리
        if len(self.trade_history) > self.config.memory_limit:
            self.trade_history.popleft()
            
        # 가비지 컬렉션 실행
        gc.collect()
        
    async def _execute_buy(self, symbol: str, amount: float, price: float) -> bool:
        """매수 주문 실행"""
        cost = Decimal(str(amount * price))
        fee = cost * Decimal('0.001')  # 0.1% 수수료
        
        if self.balance >= (cost + fee):
            self.balance -= (cost + fee)
            
            if symbol in self.positions:
                # 기존 포지션 업데이트
                pos = self.positions[symbol]
                total_amount = pos['amount'] + amount
                total_cost = (pos['amount'] * pos['entry_price']) + cost
                pos['amount'] = total_amount
                pos['entry_price'] = float(total_cost / total_amount)
            else:
                # 새 포지션 생성
                self.positions[symbol] = {
                    'side': 'buy',
                    'amount': amount,
                    'entry_price': price,
                    'unrealized_pnl': 0
                }
                
            await self._record_trade(symbol, 'buy', amount, price, fee)
            return True
            
        return False
        
    async def _execute_sell(self, symbol: str, amount: float, price: float) -> bool:
        """매도 주문 실행"""
        if symbol not in self.positions:
            return False
            
        pos = self.positions[symbol]
        if pos['amount'] < amount:
            return False
            
        revenue = Decimal(str(amount * price))
        fee = revenue * Decimal('0.001')  # 0.1% 수수료
        
        self.balance += (revenue - fee)
        pos['amount'] -= amount
        
        # 포지션 청산
        if pos['amount'] <= 0:
            del self.positions[symbol]
            
        await self._record_trade(symbol, 'sell', amount, price, fee)
        return True
        
    async def _record_trade(self, symbol: str, side: str, amount: float, price: float, fee: Decimal):
        """거래 기록"""
        trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'price': price,
            'fee': float(fee),
            'balance': float(self.balance)
        }
        
        self.trade_history.append(trade)
        
        # 파일에 저장
        await self._save_trade_history()
        
    async def _save_trade_history(self):
        """거래 내역 저장"""
        file_path = os.path.join(self.data_dir, 'trades.json')
        with open(file_path, 'w') as f:
            json.dump(list(self.trade_history), f)
            
    def get_account_summary(self) -> Dict:
        """계좌 요약 정보"""
        total_pnl = sum(pos.get('unrealized_pnl', 0) for pos in self.positions.values())
        total_value = float(self.balance) + total_pnl
        
        return {
            'balance': float(self.balance),
            'total_value': total_value,
            'unrealized_pnl': total_pnl,
            'daily_pnl': float(self.daily_pnl),
            'positions': len(self.positions)
        }
        
    def get_position_value(self, symbol: str) -> Optional[float]:
        """포지션 가치 계산"""
        if symbol not in self.positions:
            return None
            
        pos = self.positions[symbol]
        return pos['amount'] * pos['entry_price']
        
    def update_position_pnl(self, symbol: str, current_price: float):
        """포지션 손익 업데이트"""
        if symbol not in self.positions:
            return
            
        pos = self.positions[symbol]
        if pos['side'] == 'buy':
            pnl = (current_price - pos['entry_price']) * pos['amount']
        else:
            pnl = (pos['entry_price'] - current_price) * pos['amount']
            
        pos['unrealized_pnl'] = pnl
        
    def simulate_market_volatility(self, price: float) -> float:
        """시장 변동성 시뮬레이션"""
        volatility = np.random.normal(0, self.config.market_volatility * 0.01)
        return price * (1 + volatility)
        
    def reset(self):
        """시뮬레이터 초기화"""
        self.balance = Decimal(str(self.config.initial_capital))
        self.positions.clear()
        self.trade_history.clear()
        self.daily_pnl = Decimal('0')
        self.last_reset = datetime.now().date() 