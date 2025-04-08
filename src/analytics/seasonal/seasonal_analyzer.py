from typing import Dict, List
import pandas as pd
import numpy as np

class SeasonalAnalyzer:
    def __init__(self, min_history_days: int = 365):
        self.min_history_days = min_history_days
        
    def analyze_seasonality(self, price_data: pd.DataFrame) -> Dict:
        """계절성 패턴 분석"""
        daily_patterns = self._analyze_daily_patterns(price_data)
        weekly_patterns = self._analyze_weekly_patterns(price_data)
        monthly_patterns = self._analyze_monthly_patterns(price_data)
        
        return {
            'daily_patterns': daily_patterns,
            'weekly_patterns': weekly_patterns,
            'monthly_patterns': monthly_patterns,
            'strength': self._calculate_seasonal_strength(price_data),
            'current_effect': self._calculate_current_seasonal_effect(price_data)
        }
