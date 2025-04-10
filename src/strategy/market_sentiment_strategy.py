from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class SentimentSignal:
    sentiment_score: float
    market_fear: float
    market_greed: float
    signal_type: str
    confidence: float

class MarketSentimentStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'sentiment_threshold': 0.6,
            'volume_weight': 0.3,
            'price_action_weight': 0.3,
            'social_weight': 0.4
        }
        
    async def analyze_sentiment(self, market_data: pd.DataFrame, 
                              social_data: Dict) -> SentimentSignal:
        """시장 심리 분석"""
        volume_sentiment = self._analyze_volume_sentiment(market_data)
        price_sentiment = self._analyze_price_sentiment(market_data)
        social_sentiment = self._analyze_social_sentiment(social_data)
        
        total_sentiment = (
            volume_sentiment * self.config['volume_weight'] +
            price_sentiment * self.config['price_action_weight'] +
            social_sentiment * self.config['social_weight']
        )
        
        return SentimentSignal(
            sentiment_score=total_sentiment,
            market_fear=self._calculate_fear_index(market_data),
            market_greed=self._calculate_greed_index(market_data),
            signal_type=self._determine_signal(total_sentiment),
            confidence=abs(total_sentiment)
        )
