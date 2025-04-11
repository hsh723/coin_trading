import pandas as pd
import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class VolumeMetrics:
    relative_volume: float
    volume_trend: str
    unusual_volume: bool
    volume_profile: Dict[str, float]

class RelativeVolumeAnalyzer:
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
        
    def analyze_volume(self, market_data: pd.DataFrame) -> VolumeMetrics:
        """상대 거래량 분석"""
        avg_volume = market_data['volume'].rolling(window=self.lookback_period).mean()
        current_volume = market_data['volume'].iloc[-1]
        relative_vol = current_volume / avg_volume.iloc[-1]
        
        return VolumeMetrics(
            relative_volume=relative_vol,
            volume_trend=self._determine_volume_trend(market_data),
            unusual_volume=relative_vol > 2.0,
            volume_profile=self._calculate_volume_profile(market_data)
        )

    def _determine_volume_trend(self, market_data: pd.DataFrame) -> str:
        """거래량 추세 판단"""
        recent_volumes = market_data['volume'].tail(self.lookback_period)
        volume_sma = recent_volumes.rolling(window=5).mean()
        
        if volume_sma.iloc[-1] > volume_sma.iloc[-2]:
            return "increasing"
        elif volume_sma.iloc[-1] < volume_sma.iloc[-2]:
            return "decreasing"
        return "stable"
    
    def _calculate_volume_profile(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """거래량 프로파일 계산"""
        recent_data = market_data.tail(self.lookback_period)
        total_volume = recent_data['volume'].sum()
        
        return {
            'average_volume': recent_data['volume'].mean(),
            'volume_volatility': recent_data['volume'].std() / recent_data['volume'].mean(),
            'volume_concentration': total_volume / len(recent_data)
        }
