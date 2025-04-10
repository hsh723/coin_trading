from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class SentimentResult:
    market_sentiment: float  # -1.0 to 1.0
    news_impact: Dict[str, float]
    social_metrics: Dict[str, float]
    trading_signal: str

class SentimentAnalysisStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'sentiment_threshold': 0.3,
            'data_sources': ['twitter', 'reddit', 'news'],
            'impact_window': 24  # hours
        }
        
    async def analyze_sentiment(self, sentiment_data: Dict) -> SentimentResult:
        """감성 분석 실행"""
        sentiment_score = self._calculate_weighted_sentiment(sentiment_data)
        news_impact = self._analyze_news_impact(sentiment_data.get('news', {}))
        social_metrics = self._analyze_social_metrics(sentiment_data)
        
        return SentimentResult(
            market_sentiment=sentiment_score,
            news_impact=news_impact,
            social_metrics=social_metrics,
            trading_signal=self._generate_signal(sentiment_score, news_impact)
        )
