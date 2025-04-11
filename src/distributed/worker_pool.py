import asyncio
from typing import Dict, List
import multiprocessing as mp

class WorkerPool:
    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or mp.cpu_count()
        self.task_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.workers = []
        
    async def start_workers(self):
        """워커 프로세스 시작"""
        for _ in range(self.num_workers):
            worker = Worker(self.task_queue, self.result_queue)
            self.workers.append(worker)
            await worker.start()
