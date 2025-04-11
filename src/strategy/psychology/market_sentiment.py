from typing import Dict, List
from dataclasses import dataclass

@dataclass
class SentimentAnalysis:
    fear_greed_index: float
    market_mood: str
    sentiment_signals: Dict[str, float]
    confidence_score: float

class MarketSentimentAnalyzer:
    def __init__(self):
        self.indicators = {
            'price_momentum': 0.3,
            'volume_force': 0.2,
            'volatility': 0.2,
            'trend_strength': 0.3
        }
