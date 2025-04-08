from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd

class ExchangeBase(ABC):
    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """OHLCV 데이터 조회"""
        pass

    @abstractmethod
    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None) -> Dict:
        """주문 생성"""
        pass

    @abstractmethod
    async def get_balance(self) -> Dict[str, float]:
        """잔고 조회"""
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> Dict:
        """포지션 정보 조회"""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """주문 취소"""
        pass
