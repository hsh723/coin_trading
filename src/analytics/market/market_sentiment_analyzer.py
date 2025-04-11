import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class MarketSentiment:
    sentiment_score: float
    fear_greed_index: float
    momentum_strength: float
    market_phase: str

class MarketSentimentAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'sentiment_window': 20,
            'momentum_threshold': 0.6
        }
        
    async def analyze_sentiment(self, market_data: Dict) -> MarketSentiment:
        """시장 심리 분석"""
        return MarketSentiment(
            sentiment_score=self._calculate_sentiment_score(market_data),
            fear_greed_index=self._calculate_fear_greed(market_data),
            momentum_strength=self._calculate_momentum_strength(market_data),
            market_phase=self._determine_market_phase(market_data)
        )
