from typing import Dict
from dataclasses import dataclass

@dataclass
class FeeCalculation:
    trading_fee: float
    network_fee: float
    total_fee: float
    fee_currency: str

class FeeCalculator:
    def __init__(self, fee_structure: Dict = None):
        self.fee_structure = fee_structure or {
            'maker': 0.001,  # 0.1%
            'taker': 0.002,  # 0.2%
            'network_fee_multiplier': 1.5
        }
        
    def calculate_fees(self, trade: Dict, 
                      is_maker: bool = False) -> FeeCalculation:
        """거래 수수료 계산"""
        base_fee_rate = (
            self.fee_structure['maker'] if is_maker 
            else self.fee_structure['taker']
        )
        
        trading_fee = trade['amount'] * base_fee_rate
        network_fee = self._estimate_network_fee(trade)
        
        return FeeCalculation(
            trading_fee=trading_fee,
            network_fee=network_fee,
            total_fee=trading_fee + network_fee,
            fee_currency=trade['quote_currency']
        )
