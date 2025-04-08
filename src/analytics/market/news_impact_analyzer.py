import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class NewsImpact:
    importance: str
    market_reaction: float
    volume_impact: float
    volatility_change: float

class NewsImpactAnalyzer:
    def __init__(self, lookback_period: int = 60):
        self.lookback_period = lookback_period
        
    async def analyze_news_impact(self, 
                                news_data: Dict,
                                market_data: pd.DataFrame) -> NewsImpact:
        """뉴스가 시장에 미치는 영향 분석"""
        pre_news_data = market_data[:news_data['timestamp']]
        post_news_data = market_data[news_data['timestamp']:]
        
        return NewsImpact(
            importance=self._classify_importance(news_data),
            market_reaction=self._calculate_price_reaction(pre_news_data, post_news_data),
            volume_impact=self._calculate_volume_impact(pre_news_data, post_news_data),
            volatility_change=self._calculate_volatility_change(pre_news_data, post_news_data)
        )
