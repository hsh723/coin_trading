from textblob import TextBlob
import pandas as pd
from typing import Dict, List
import asyncio
import aiohttp

class SocialMediaAnalyzer:
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.sentiment_cache = {}
        
    async def analyze_social_sentiment(self, symbol: str) -> Dict[str, float]:
        """소셜 미디어 감성 분석"""
        twitter_sentiment = await self._analyze_twitter(symbol)
        reddit_sentiment = await self._analyze_reddit(symbol)
        
        return {
            'twitter': twitter_sentiment,
            'reddit': reddit_sentiment,
            'combined': (twitter_sentiment + reddit_sentiment) / 2
        }
