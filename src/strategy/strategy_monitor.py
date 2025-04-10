from typing import Dict, List
from dataclasses import dataclass

@dataclass
class StrategyMonitorMetrics:
    strategy_id: str
    current_positions: Dict[str, float]
    realized_pnl: float
    drawdown: float
    health_status: str

class StrategyMonitor:
    def __init__(self, monitor_config: Dict = None):
        self.config = monitor_config or {
            'health_check_interval': 60,
            'alert_thresholds': {
                'drawdown': 0.1,
                'loss_streak': 3
            }
        }
        self.strategy_states = {}
        
    async def monitor_strategy(self, strategy_id: str, 
                             market_data: Dict) -> StrategyMonitorMetrics:
        """전략 상태 모니터링"""
        current_state = self.strategy_states.get(strategy_id, {})
        positions = self._analyze_positions(current_state)
        pnl = self._calculate_pnl(positions, market_data)
        
        metrics = StrategyMonitorMetrics(
            strategy_id=strategy_id,
            current_positions=positions,
            realized_pnl=pnl['realized'],
            drawdown=self._calculate_drawdown(pnl['history']),
            health_status=self._check_strategy_health(current_state)
        )
        
        await self._handle_alerts(metrics)
        return metrics
