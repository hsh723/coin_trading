from typing import Dict, List
import numpy as np

class HybridIndicatorSystem:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'indicators': ['rsi', 'macd', 'bollinger', 'atr'],
            'fusion_method': 'weighted',
            'weights': {'rsi': 0.3, 'macd': 0.3, 'bollinger': 0.2, 'atr': 0.2}
        }
        
    async def generate_hybrid_signal(self, market_data: pd.DataFrame) -> Dict:
        """여러 지표를 결합한 하이브리드 신호 생성"""
        signals = {}
        for indicator in self.config['indicators']:
            signals[indicator] = await self._calculate_indicator(market_data, indicator)
            
        return {
            'composite_signal': self._fuse_signals(signals),
            'confidence_score': self._calculate_confidence(signals),
            'individual_signals': signals,
            'signal_correlation': self._calculate_signal_correlation(signals)
        }
