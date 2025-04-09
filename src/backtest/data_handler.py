from typing import Dict, List
import pandas as pd
from dataclasses import dataclass

@dataclass
class BacktestData:
    ohlcv: pd.DataFrame
    indicators: Dict[str, pd.Series]
    signals: pd.Series
    metadata: Dict

class BacktestDataHandler:
    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date
        
    async def prepare_backtest_data(self, symbols: List[str]) -> Dict[str, BacktestData]:
        """백테스트용 데이터 준비"""
        backtest_data = {}
        for symbol in symbols:
            ohlcv = await self._load_market_data(symbol)
            indicators = self._calculate_indicators(ohlcv)
            signals = self._generate_signals(ohlcv, indicators)
            
            backtest_data[symbol] = BacktestData(
                ohlcv=ohlcv,
                indicators=indicators,
                signals=signals,
                metadata=self._create_metadata(symbol)
            )
        return backtest_data
