from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class AdaptiveSignal:
    signal_type: str       # 'entry', 'exit', 'adjust'
    confidence: float      # 0.0 to 1.0
    primary_factors: List[str]
    adaptive_parameters: Dict[str, float]

class AdaptiveSignalStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'min_confidence': 0.6,
            'adaptation_speed': 0.1,
            'signal_timeout': 30
        }
        
    async def generate_adaptive_signal(self, market_data: pd.DataFrame) -> AdaptiveSignal:
        """적응형 매매 신호 생성"""
        factors = self._analyze_market_factors(market_data)
        confidence = self._calculate_signal_confidence(factors)
        
        return AdaptiveSignal(
            signal_type=self._determine_signal_type(factors, confidence),
            confidence=confidence,
            primary_factors=self._identify_primary_factors(factors),
            adaptive_parameters=self._adjust_parameters(factors)
        )
