import psutil
import threading
import time
from typing import Dict, List
import numpy as np

class PerformanceMonitor:
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.metrics: Dict[str, List[float]] = {
            'cpu_percent': [],
            'memory_percent': [],
            'disk_usage': [],
            'network_io': []
        }
        self._stop_flag = threading.Event()
        
    def start(self) -> None:
        self._stop_flag.clear()
        thread = threading.Thread(target=self._monitor)
        thread.daemon = True
        thread.start()

    def stop(self) -> None:
        self._stop_flag.set()

    def _monitor(self) -> None:
        while not self._stop_flag.is_set():
            self.metrics['cpu_percent'].append(psutil.cpu_percent())
            self.metrics['memory_percent'].append(psutil.virtual_memory().percent)
            self.metrics['disk_usage'].append(psutil.disk_usage('/').percent)
            
            net_io = psutil.net_io_counters()
            self.metrics['network_io'].append(net_io.bytes_sent + net_io.bytes_recv)
            
            time.sleep(self.sampling_interval)

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for metric, values in self.metrics.items():
            if values:
                stats[metric] = {
                    'mean': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'std': np.std(values)
                }
        return stats
