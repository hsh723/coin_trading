import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class QuoteRevision:
    revision_frequency: float
    quote_intensity: float
    quote_stability: float
    revision_impact: Dict[str, float]

class QuoteRevisionAnalyzer:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
    async def analyze_revisions(self, quote_data: List[Dict]) -> QuoteRevision:
        """호가 변경 분석"""
        recent_quotes = quote_data[-self.window_size:]
        frequency = self._calculate_revision_frequency(recent_quotes)
        intensity = self._calculate_quote_intensity(recent_quotes)
        
        return QuoteRevision(
            revision_frequency=frequency,
            quote_intensity=intensity,
            quote_stability=self._calculate_stability(recent_quotes),
            revision_impact=self._calculate_impact(recent_quotes)
        )
