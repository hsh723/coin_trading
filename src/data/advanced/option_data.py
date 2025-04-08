from typing import Dict, List
import pandas as pd
import numpy as np
from scipy.stats import norm

class OptionDataAnalyzer:
    def __init__(self):
        self.volatility_surface = {}
        self.greeks = {}
        
    def calculate_implied_volatility(self, 
                                   option_prices: pd.DataFrame, 
                                   underlying_price: float,
                                   risk_free_rate: float = 0.02) -> pd.DataFrame:
        """내재변동성 계산"""
        def _black_scholes(S, K, T, r, sigma, option_type='call'):
            d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            if option_type == 'call':
                return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            else:
                return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
