from typing import Dict, List
from dataclasses import dataclass

@dataclass
class SimulatedTrade:
    timestamp: str
    symbol: str
    side: str
    price: float
    amount: float
    fees: float
    slippage: float

class TradeSimulator:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'commission_rate': 0.001,
            'slippage_model': 'basic',
            'market_impact': True
        }
        
    async def simulate_trade(self, order: Dict, market_data: Dict) -> SimulatedTrade:
        """거래 시뮬레이션 실행"""
        execution_price = self._calculate_execution_price(order, market_data)
        slippage = self._estimate_slippage(order, market_data)
        fees = self._calculate_fees(order['amount'], execution_price)
        
        return SimulatedTrade(
            timestamp=market_data['timestamp'],
            symbol=order['symbol'],
            side=order['side'],
            price=execution_price,
            amount=order['amount'],
            fees=fees,
            slippage=slippage
        )
