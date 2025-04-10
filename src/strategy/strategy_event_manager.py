from typing import Dict, Callable
from dataclasses import dataclass

@dataclass
class StrategyEvent:
    event_type: str
    strategy_id: str
    data: Dict
    timestamp: float

class StrategyEventManager:
    def __init__(self):
        self.event_handlers = {}
        self.event_history = []
        
    async def handle_event(self, event: StrategyEvent):
        """전략 이벤트 처리"""
        if event.event_type in self.event_handlers:
            for handler in self.event_handlers[event.event_type]:
                await handler(event)
                
        self.event_history.append(event)
