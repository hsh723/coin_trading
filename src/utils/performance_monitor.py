"""
시스템 성능 모니터링 모듈
"""

import psutil
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """시스템 메트릭스 데이터 클래스"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    thread_count: int
    open_files: int
    swap_usage: float

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(
        self,
        interval: float = 1.0,
        max_samples: int = 1000,
        log_dir: str = "logs/performance"
    ):
        self.interval = interval
        self.max_samples = max_samples
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history: List[SystemMetrics] = []
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """모니터링 시작"""
        if not self.is_running:
            self.is_running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            self.logger.info("성능 모니터링 시작")
    
    def stop(self):
        """모니터링 중지"""
        if self.is_running:
            self.is_running = False
            if self.monitor_thread:
                self.monitor_thread.join()
            self.logger.info("성능 모니터링 중지")
    
    def _monitor_loop(self):
        """모니터링 루프"""
        while self.is_running:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # 최대 샘플 수 제한
                if len(self.metrics_history) > self.max_samples:
                    self.metrics_history = self.metrics_history[-self.max_samples:]
                
                time.sleep(self.interval)
                
            except Exception as e:
                self.logger.error(f"메트릭스 수집 중 오류 발생: {str(e)}")
                time.sleep(self.interval)
    
    def _collect_metrics(self) -> SystemMetrics:
        """시스템 메트릭스 수집"""
        try:
            # CPU 사용량
            cpu_usage = psutil.cpu_percent(interval=None)
            
            # 메모리 사용량
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # 디스크 사용량
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # 네트워크 I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # 프로세스 및 스레드 수
            process_count = len(psutil.pids())
            thread_count = threading.active_count()
            
            # 열린 파일 수
            open_files = len(psutil.Process().open_files())
            
            # 스왑 사용량
            swap = psutil.swap_memory()
            swap_usage = swap.percent
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                process_count=process_count,
                thread_count=thread_count,
                open_files=open_files,
                swap_usage=swap_usage
            )
            
        except Exception as e:
            self.logger.error(f"메트릭스 수집 중 오류 발생: {str(e)}")
            raise
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """현재 메트릭스 조회"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_history(self) -> List[SystemMetrics]:
        """메트릭스 히스토리 조회"""
        return self.metrics_history
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """메트릭스 요약 통계"""
        if not self.metrics_history:
            return {}
        
        df = pd.DataFrame([vars(m) for m in self.metrics_history])
        
        return {
            'cpu_usage': {
                'mean': df['cpu_usage'].mean(),
                'max': df['cpu_usage'].max(),
                'min': df['cpu_usage'].min(),
                'std': df['cpu_usage'].std()
            },
            'memory_usage': {
                'mean': df['memory_usage'].mean(),
                'max': df['memory_usage'].max(),
                'min': df['memory_usage'].min(),
                'std': df['memory_usage'].std()
            },
            'disk_usage': {
                'mean': df['disk_usage'].mean(),
                'max': df['disk_usage'].max(),
                'min': df['disk_usage'].min(),
                'std': df['disk_usage'].std()
            },
            'process_count': {
                'mean': df['process_count'].mean(),
                'max': df['process_count'].max(),
                'min': df['process_count'].min(),
                'std': df['process_count'].std()
            },
            'thread_count': {
                'mean': df['thread_count'].mean(),
                'max': df['thread_count'].max(),
                'min': df['thread_count'].min(),
                'std': df['thread_count'].std()
            },
            'open_files': {
                'mean': df['open_files'].mean(),
                'max': df['open_files'].max(),
                'min': df['open_files'].min(),
                'std': df['open_files'].std()
            },
            'swap_usage': {
                'mean': df['swap_usage'].mean(),
                'max': df['swap_usage'].max(),
                'min': df['swap_usage'].min(),
                'std': df['swap_usage'].std()
            }
        }
    
    def save_metrics(self, filename: Optional[str] = None):
        """메트릭스 저장"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"metrics_{timestamp}.json"
            
            filepath = self.log_dir / filename
            
            metrics_data = {
                'summary': self.get_metrics_summary(),
                'history': [vars(m) for m in self.metrics_history]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"메트릭스 저장 완료: {filename}")
            
        except Exception as e:
            self.logger.error(f"메트릭스 저장 중 오류 발생: {str(e)}")
    
    def load_metrics(self, filename: str):
        """메트릭스 로드"""
        try:
            filepath = self.log_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"메트릭스 파일을 찾을 수 없습니다: {filename}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            
            self.metrics_history = [
                SystemMetrics(**m) for m in metrics_data['history']
            ]
            
            self.logger.info(f"메트릭스 로드 완료: {filename}")
            
        except Exception as e:
            self.logger.error(f"메트릭스 로드 중 오류 발생: {str(e)}")
            raise
    
    def clear_metrics(self):
        """메트릭스 초기화"""
        self.metrics_history = []
        self.logger.info("메트릭스 초기화 완료")
    
    def check_alerts(self) -> List[str]:
        """경고 체크"""
        alerts = []
        current_metrics = self.get_current_metrics()
        
        if current_metrics:
            # CPU 사용량 경고
            if current_metrics.cpu_usage > 90:
                alerts.append(f"CPU 사용량이 높습니다: {current_metrics.cpu_usage}%")
            
            # 메모리 사용량 경고
            if current_metrics.memory_usage > 90:
                alerts.append(f"메모리 사용량이 높습니다: {current_metrics.memory_usage}%")
            
            # 디스크 사용량 경고
            if current_metrics.disk_usage > 90:
                alerts.append(f"디스크 사용량이 높습니다: {current_metrics.disk_usage}%")
            
            # 스왑 사용량 경고
            if current_metrics.swap_usage > 80:
                alerts.append(f"스왑 사용량이 높습니다: {current_metrics.swap_usage}%")
            
            # 프로세스 수 경고
            if current_metrics.process_count > 1000:
                alerts.append(f"프로세스 수가 많습니다: {current_metrics.process_count}")
            
            # 열린 파일 수 경고
            if current_metrics.open_files > 1000:
                alerts.append(f"열린 파일 수가 많습니다: {current_metrics.open_files}")
        
        return alerts 