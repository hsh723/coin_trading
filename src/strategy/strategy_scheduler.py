from typing import Dict, List
from dataclasses import dataclass
import asyncio

@dataclass
class ScheduleConfig:
    strategy_id: str
    interval: int
    active_hours: List[Dict]
    max_runs: Optional[int]

class StrategyScheduler:
    def __init__(self, scheduler_config: Dict = None):
        self.config = scheduler_config or {
            'default_interval': 60,  # 1분
            'max_concurrent': 5
        }
        self.scheduled_tasks = {}
        
    async def schedule_strategy(self, strategy_id: str, 
                              schedule: ScheduleConfig) -> bool:
        """전략 스케줄링"""
        if len(self.scheduled_tasks) >= self.config['max_concurrent']:
            return False
            
        task = asyncio.create_task(
            self._run_strategy_loop(strategy_id, schedule)
        )
        
        self.scheduled_tasks[strategy_id] = {
            'task': task,
            'config': schedule,
            'runs': 0
        }
        
        return True
