from typing import Dict, List
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class StrategyMetrics:
    position_size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    risk_ratio: float

class BaseStrategy(ABC):
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.position = None
        self.metrics = None
        self.trade_history = []
        
    @abstractmethod
    async def analyze_market(self, market_data: Dict) -> Dict:
        """시장 분석"""
        pass
        
    @abstractmethod
    async def generate_signals(self, analysis: Dict) -> Dict:
        """거래 신호 생성"""
        pass
        
    @abstractmethod
    async def manage_risk(self, position: Dict, market_data: Dict) -> Dict:
        """리스크 관리"""
        pass
