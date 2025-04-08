from typing import Dict, List
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class SentimentScore:
    overall_score: float
    news_score: float
    social_score: float
    on_chain_score: float
    market_score: float

class CombinedSentimentAnalyzer:
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'news': 0.3,
            'social': 0.2,
            'on_chain': 0.3,
            'market': 0.2
        }
        
    async def analyze_sentiment(self, data: Dict) -> SentimentScore:
        """다중 소스 감성 분석 통합"""
        news_sentiment = await self._analyze_news_sentiment(data.get('news', []))
        social_sentiment = await self._analyze_social_sentiment(data.get('social', []))
        on_chain_sentiment = self._analyze_on_chain_metrics(data.get('on_chain', {}))
        market_sentiment = self._analyze_market_sentiment(data.get('market', {}))
        
        overall = self._combine_sentiments({
            'news': news_sentiment,
            'social': social_sentiment,
            'on_chain': on_chain_sentiment,
            'market': market_sentiment
        })
        
        return SentimentScore(
            overall_score=overall,
            news_score=news_sentiment,
            social_score=social_sentiment,
            on_chain_score=on_chain_sentiment,
            market_score=market_sentiment
        )
