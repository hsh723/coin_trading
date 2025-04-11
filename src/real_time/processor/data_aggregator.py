import numpy as np
from typing import Dict, List
import pandas as pd

class DataAggregator:
    def __init__(self, aggregation_periods: List[str]):
        self.periods = aggregation_periods
        self.buffers = {period: [] for period in aggregation_periods}
        
    async def aggregate_data(self, data: Dict) -> Dict[str, pd.DataFrame]:
        """실시간 데이터 집계"""
        results = {}
        for period in self.periods:
            aggregated = await self._aggregate_period(data, period)
            results[period] = aggregated
        return results
