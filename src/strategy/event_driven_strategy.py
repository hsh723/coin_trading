from typing import Dict, List
import pandas as pd
from .base import BaseStrategy

class EventDrivenStrategy(BaseStrategy):
    def __init__(self, config: Dict):
        super().__init__()
        self.event_thresholds = config.get('event_thresholds', {})
        self.position_sizing = config.get('position_sizing', {})
        
    def detect_events(self, market_data: pd.DataFrame) -> List[Dict]:
        """시장 이벤트 감지"""
        events = []
        
        # 가격 급등/급락 이벤트
        returns = market_data['close'].pct_change()
        if abs(returns.iloc[-1]) > self.event_thresholds.get('price_move', 0.05):
            events.append({
                'type': 'PRICE_SHOCK',
                'magnitude': returns.iloc[-1]
            })
            
        return events
