from typing import Dict, List
import pandas as pd
from dataclasses import dataclass

@dataclass
class AggregationConfig:
    timeframes: List[str]
    methods: Dict[str, str]
    custom_functions: Dict[str, callable] = None

class TimeSeriesAggregator:
    def __init__(self, config: AggregationConfig):
        self.config = config
        
    async def aggregate_data(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """시계열 데이터 집계"""
        results = {}
        
        for timeframe in self.config.timeframes:
            resampled = data.resample(timeframe)
            aggregated = {}
            
            for column, method in self.config.methods.items():
                if method == 'custom' and self.config.custom_functions:
                    aggregated[column] = resampled[column].apply(
                        self.config.custom_functions[column]
                    )
                else:
                    aggregated[column] = getattr(resampled[column], method)()
                    
            results[timeframe] = pd.DataFrame(aggregated)
            
        return results
