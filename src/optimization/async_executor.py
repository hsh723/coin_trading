import asyncio
from typing import List, Callable, Any
import aiohttp
import logging

class AsyncExecutor:
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def execute(self, tasks: List[Callable]) -> List[Any]:
        async with self:
            return await asyncio.gather(
                *(self._run_task(task) for task in tasks)
            )

    async def _run_task(self, task: Callable) -> Any:
        async with self.semaphore:
            try:
                return await task()
            except Exception as e:
                logging.error(f"Task execution failed: {str(e)}")
                return None
