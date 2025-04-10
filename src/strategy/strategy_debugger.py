from typing import Dict, List
from dataclasses import dataclass

@dataclass
class DebugInfo:
    strategy_id: str
    execution_trace: List[Dict]
    signal_history: List[Dict]
    performance_snapshot: Dict

class StrategyDebugger:
    def __init__(self, debug_config: Dict = None):
        self.config = debug_config or {
            'trace_depth': 10,
            'log_signals': True,
            'performance_tracking': True
        }
        self.debug_history = []
        
    async def debug_strategy(self, strategy_id: str, 
                           execution_data: Dict) -> DebugInfo:
        """전략 디버깅 정보 수집"""
        trace = self._collect_execution_trace(execution_data)
        signals = self._collect_signal_history(strategy_id)
        performance = self._create_performance_snapshot(strategy_id)
        
        debug_info = DebugInfo(
            strategy_id=strategy_id,
            execution_trace=trace,
            signal_history=signals,
            performance_snapshot=performance
        )
        
        self.debug_history.append(debug_info)
        return debug_info
