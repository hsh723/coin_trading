from typing import Dict, List
from dataclasses import dataclass

@dataclass
class SignalMetrics:
    signal_strength: float
    signal_confidence: float
    confirmation_count: int
    signal_type: str

class SignalProcessor:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'min_strength': 0.5,
            'confirmation_threshold': 2,
            'signal_timeout': 300  # 5분
        }
        
    async def process_signal(self, signal_data: Dict) -> SignalMetrics:
        """거래 신호 처리"""
        strength = self._calculate_signal_strength(signal_data)
        confidence = self._assess_signal_confidence(signal_data)
        confirmations = self._count_confirmations(signal_data)
        
        return SignalMetrics(
            signal_strength=strength,
            signal_confidence=confidence,
            confirmation_count=confirmations,
            signal_type=self._determine_signal_type(strength, confidence)
        )
