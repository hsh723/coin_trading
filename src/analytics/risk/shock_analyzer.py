from typing import Dict, List
import numpy as np
import pandas as pd

class MarketShockAnalyzer:
    def __init__(self, config: Dict):
        self.shock_threshold = config.get('shock_threshold', 0.03)
        self.min_shock_duration = config.get('min_shock_duration', 5)
        
    def detect_market_shocks(self, price_data: pd.DataFrame) -> List[Dict]:
        """시장 충격 감지"""
        returns = price_data['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        
        shocks = []
        current_shock = None
        
        for timestamp, ret in returns.items():
            if abs(ret) > self.shock_threshold:
                if current_shock is None:
                    current_shock = {
                        'start_time': timestamp,
                        'max_return': ret,
                        'cumulative_return': ret
                    }
                else:
                    current_shock['cumulative_return'] += ret
                    current_shock['max_return'] = max(
                        abs(current_shock['max_return']), 
                        abs(ret)
                    )
            elif current_shock is not None:
                shocks.append(current_shock)
                current_shock = None
                
        return shocks
