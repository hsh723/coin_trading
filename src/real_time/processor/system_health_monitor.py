import asyncio
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class SystemHealth:
    system_load: float
    memory_usage: float
    io_stats: Dict[str, float]
    health_status: str

class SystemHealthMonitor:
    def __init__(self, check_interval: float = 1.0):
        self.check_interval = check_interval
        self.health_history = []
        
    async def monitor_health(self) -> SystemHealth:
        """시스템 상태 모니터링"""
        system_load = await self._get_system_load()
        memory_usage = await self._get_memory_usage()
        io_stats = await self._get_io_stats()
        
        health = SystemHealth(
            system_load=system_load,
            memory_usage=memory_usage,
            io_stats=io_stats,
            health_status=self._determine_health_status(system_load, memory_usage)
        )
        
        self.health_history.append(health)
        return health
