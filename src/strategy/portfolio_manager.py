"""
포트폴리오 관리 모듈
"""

from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

class PortfolioManager:
    """포트폴리오 관리 클래스"""
    
    def __init__(self):
        """포트폴리오 관리자 초기화"""
        self.positions: Dict[str, float] = {}  # 심볼: 수량
        self.balance: float = 0.0  # USDT 잔고
        self.trade_history: List[dict] = []
        
    def update_position(self, symbol: str, quantity: float) -> None:
        """포지션 업데이트"""
        self.positions[symbol] = quantity
        
    def update_balance(self, amount: float) -> None:
        """잔고 업데이트"""
        self.balance = amount
        
    def add_trade(self, symbol: str, side: str, quantity: float, price: float) -> None:
        """거래 내역 추가"""
        self.trade_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price
        })
        
    def get_position(self, symbol: str) -> float:
        """특정 심볼의 포지션 조회"""
        return self.positions.get(symbol, 0.0)
    
    def get_all_positions(self) -> Dict[str, float]:
        """모든 포지션 조회"""
        return self.positions
    
    def get_balance(self) -> float:
        """잔고 조회"""
        return self.balance
    
    def get_trade_history(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """거래 내역 조회"""
        df = pd.DataFrame(self.trade_history)
        if symbol:
            df = df[df['symbol'] == symbol]
        return df 