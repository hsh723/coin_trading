from typing import Dict, List
import pandas as pd
from dataclasses import dataclass

@dataclass
class AggregatedData:
    ohlcv: pd.DataFrame
    indicators: Dict[str, pd.Series]
    metadata: Dict[str, any]

class MarketDataAggregator:
    def __init__(self, timeframes: List[str] = ['1m', '5m', '15m', '1h', '4h', '1d']):
        self.timeframes = timeframes
        
    async def aggregate_market_data(self, raw_data: pd.DataFrame) -> Dict[str, AggregatedData]:
        """시장 데이터 집계"""
        aggregated = {}
        
        for timeframe in self.timeframes:
            resampled = self._resample_data(raw_data, timeframe)
            indicators = self._calculate_indicators(resampled)
            
            aggregated[timeframe] = AggregatedData(
                ohlcv=resampled,
                indicators=indicators,
                metadata=self._generate_metadata(timeframe, resampled)
            )
            
        return aggregated
