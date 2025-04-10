from typing import Dict, List
from dataclasses import dataclass

@dataclass
class StrategyStatus:
    active_strategies: Dict[str, bool]
    performance_metrics: Dict[str, float]
    risk_exposure: Dict[str, float]

class StrategyManager:
    def __init__(self, strategy_config: Dict = None):
        self.config = strategy_config or {}
        self.strategies = {}
        self.active_strategies = {}
        
    async def deploy_strategy(self, strategy_id: str, 
                            strategy_config: Dict) -> bool:
        """전략 배포"""
        try:
            strategy = await self._initialize_strategy(strategy_id, strategy_config)
            validation = await self._validate_strategy(strategy)
            
            if validation.is_valid:
                self.strategies[strategy_id] = strategy
                self.active_strategies[strategy_id] = True
                return True
                
            return False
            
        except Exception as e:
            await self._handle_deployment_error(e, strategy_id)
            return False
