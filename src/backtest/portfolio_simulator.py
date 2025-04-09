import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class PortfolioState:
    timestamp: str
    positions: Dict[str, float]
    cash: float
    equity: float
    margin: float

class PortfolioSimulator:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_state = None
        
    def simulate_portfolio(self, trades: List[Dict], market_data: Dict) -> List[PortfolioState]:
        """포트폴리오 시뮬레이션"""
        states = []
        self._initialize_state()
        
        for trade in trades:
            state = self._process_trade(trade, market_data)
            states.append(state)
            
        return states
