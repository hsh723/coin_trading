from typing import Dict, List
from dataclasses import dataclass

@dataclass
class MarketStatus:
    volatility_level: float
    trading_activity: str
    market_regime: str
    liquidity_status: str
    risk_level: str

class MarketStatusMonitor:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'volatility_window': 20,
            'activity_threshold': 0.5,
            'risk_levels': ['low', 'medium', 'high']
        }
        
    async def monitor_market_status(self, market_data: Dict) -> MarketStatus:
        """시장 상태 모니터링"""
        volatility = self._calculate_volatility(market_data)
        activity = self._analyze_trading_activity(market_data)
        regime = self._detect_market_regime(market_data)
        liquidity = self._assess_liquidity_status(market_data)
        
        return MarketStatus(
            volatility_level=volatility,
            trading_activity=activity,
            market_regime=regime,
            liquidity_status=liquidity,
            risk_level=self._determine_risk_level(volatility, activity)
        )
