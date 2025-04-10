from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ProcessedSignal:
    original_signal: Dict
    filtered: bool
    confirmed: bool
    execution_params: Dict

class SignalProcessor:
    def __init__(self, processor_config: Dict = None):
        self.config = processor_config or {
            'confirmation_threshold': 0.7,
            'min_confidence': 0.5
        }
        
    async def process_signal(self, signal: Dict, market_data: Dict) -> ProcessedSignal:
        """신호 처리 및 검증"""
        if not self._validate_signal(signal):
            return None
            
        confirmed = await self._confirm_signal(signal, market_data)
        execution_params = self._generate_execution_params(signal, market_data)
        
        return ProcessedSignal(
            original_signal=signal,
            filtered=self._should_filter(signal),
            confirmed=confirmed,
            execution_params=execution_params
        )
