import psutil
import pandas as pd
from typing import Dict
import asyncio

class SystemMonitor:
    def __init__(self, alert_threshold: Dict = None):
        self.alert_threshold = alert_threshold or {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0
        }
        self.metrics_history = []
        
    async def monitor_resources(self) -> Dict:
        """시스템 리소스 모니터링"""
        metrics = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters()._asdict()
        }
        
        self.metrics_history.append({
            'timestamp': pd.Timestamp.now(),
            **metrics
        })
        
        return metrics

    async def check_alerts(self) -> List[str]:
        """임계값 초과 확인"""
        alerts = []
        current_metrics = await self.monitor_resources()
        
        for metric, value in current_metrics.items():
            if metric in self.alert_threshold and value > self.alert_threshold[metric]:
                alerts.append(f"Warning: {metric} usage ({value}%) exceeds threshold")
                
        return alerts
