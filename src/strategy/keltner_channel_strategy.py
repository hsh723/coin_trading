from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class KeltnerSignal:
    signal_type: str
    channel_position: float
    channel_width: float
    atr_value: float
    breakout_detected: bool

class KeltnerChannelStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'ema_period': 20,
            'atr_period': 10,
            'channel_multiplier': 2.0
        }
        
    async def generate_signals(self, market_data: pd.DataFrame) -> KeltnerSignal:
        """켈트너 채널 신호 생성"""
        channels = self._calculate_keltner_channels(market_data)
        current_price = market_data['close'].iloc[-1]
        
        return KeltnerSignal(
            signal_type=self._determine_signal(current_price, channels),
            channel_position=self._calculate_channel_position(current_price, channels),
            channel_width=channels['upper'][-1] - channels['lower'][-1],
            atr_value=self._calculate_atr(market_data),
            breakout_detected=self._detect_breakout(current_price, channels)
        )
