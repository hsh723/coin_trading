import psutil
import time
from typing import Dict, List, Optional
from datetime import datetime
from collections import deque
from ..utils.logger import setup_logger

logger = setup_logger()

class ResourceMonitor:
    """
    시스템 리소스 사용량을 모니터링하는 도구
    
    Attributes:
        history_size (int): 저장할 히스토리 크기
        check_interval (int): 모니터링 간격 (초)
        thresholds (Dict): 리소스 임계값
    """
    
    def __init__(
        self,
        history_size: int = 1000,
        check_interval: int = 1,
        thresholds: Optional[Dict] = None
    ):
        self.history_size = history_size
        self.check_interval = check_interval
        self.thresholds = thresholds or {
            'cpu': 80,
            'memory': 80,
            'disk': 80
        }
        
        # 리소스 사용량 히스토리
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.disk_history = deque(maxlen=history_size)
        self.network_history = deque(maxlen=history_size)
        
        # 프로세스 모니터링
        self.process_history = {}
        
        # 마지막 체크 시간
        self.last_check = None
    
    def start_monitoring(self):
        """리소스 모니터링을 시작합니다."""
        try:
            while True:
                self.check_resources()
                time.sleep(self.check_interval)
        except KeyboardInterrupt:
            logger.info("Resource monitoring stopped")
        except Exception as e:
            logger.error(f"Resource monitoring error: {str(e)}")
    
    def check_resources(self) -> Dict:
        """
        시스템 리소스 사용량을 확인합니다.
        
        Returns:
            Dict: 리소스 사용량 정보
        """
        try:
            resource_data = {
                'timestamp': datetime.now().isoformat(),
                'cpu': self._check_cpu(),
                'memory': self._check_memory(),
                'disk': self._check_disk(),
                'network': self._check_network(),
                'processes': self._check_processes()
            }
            
            # 히스토리 업데이트
            self._update_history(resource_data)
            
            # 임계값 체크
            self._check_thresholds(resource_data)
            
            self.last_check = datetime.now()
            return resource_data
            
        except Exception as e:
            logger.error(f"Resource check failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _check_cpu(self) -> Dict:
        """CPU 사용량을 확인합니다."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            return {
                'percent': cpu_percent,
                'count': cpu_count,
                'frequency': {
                    'current': cpu_freq.current,
                    'min': cpu_freq.min,
                    'max': cpu_freq.max
                }
            }
        except Exception as e:
            logger.error(f"CPU check failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _check_memory(self) -> Dict:
        """메모리 사용량을 확인합니다."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used,
                'free': memory.free,
                'swap': {
                    'total': swap.total,
                    'used': swap.used,
                    'free': swap.free,
                    'percent': swap.percent
                }
            }
        except Exception as e:
            logger.error(f"Memory check failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _check_disk(self) -> Dict:
        """디스크 사용량을 확인합니다."""
        try:
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            return {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent,
                'io': {
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes,
                    'read_count': disk_io.read_count,
                    'write_count': disk_io.write_count
                }
            }
        except Exception as e:
            logger.error(f"Disk check failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _check_network(self) -> Dict:
        """네트워크 사용량을 확인합니다."""
        try:
            net_io = psutil.net_io_counters()
            net_if = psutil.net_if_stats()
            
            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'interfaces': {
                    iface: {
                        'isup': stats.isup,
                        'speed': stats.speed,
                        'mtu': stats.mtu
                    }
                    for iface, stats in net_if.items()
                }
            }
        except Exception as e:
            logger.error(f"Network check failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _check_processes(self) -> Dict:
        """프로세스 사용량을 확인합니다."""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # 프로세스별 리소스 사용량 추적
            for proc in processes:
                pid = proc['pid']
                if pid not in self.process_history:
                    self.process_history[pid] = {
                        'cpu_history': deque(maxlen=self.history_size),
                        'memory_history': deque(maxlen=self.history_size)
                    }
                
                self.process_history[pid]['cpu_history'].append(proc['cpu_percent'])
                self.process_history[pid]['memory_history'].append(proc['memory_percent'])
            
            return {
                'total': len(processes),
                'top_cpu': sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)[:5],
                'top_memory': sorted(processes, key=lambda x: x['memory_percent'], reverse=True)[:5]
            }
        except Exception as e:
            logger.error(f"Process check failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _update_history(self, resource_data: Dict):
        """리소스 사용량 히스토리를 업데이트합니다."""
        self.cpu_history.append(resource_data['cpu']['percent'])
        self.memory_history.append(resource_data['memory']['percent'])
        self.disk_history.append(resource_data['disk']['percent'])
        self.network_history.append({
            'timestamp': resource_data['timestamp'],
            'bytes_sent': resource_data['network']['bytes_sent'],
            'bytes_recv': resource_data['network']['bytes_recv']
        })
    
    def _check_thresholds(self, resource_data: Dict):
        """리소스 임계값을 체크합니다."""
        # CPU 임계값 체크
        if resource_data['cpu']['percent'] > self.thresholds['cpu']:
            logger.warning(f"CPU usage above threshold: {resource_data['cpu']['percent']}%")
        
        # 메모리 임계값 체크
        if resource_data['memory']['percent'] > self.thresholds['memory']:
            logger.warning(f"Memory usage above threshold: {resource_data['memory']['percent']}%")
        
        # 디스크 임계값 체크
        if resource_data['disk']['percent'] > self.thresholds['disk']:
            logger.warning(f"Disk usage above threshold: {resource_data['disk']['percent']}%")
    
    def get_resource_report(self) -> Dict:
        """
        리소스 사용량 리포트를 생성합니다.
        
        Returns:
            Dict: 리소스 사용량 리포트
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'current': {
                'cpu': list(self.cpu_history)[-1] if self.cpu_history else None,
                'memory': list(self.memory_history)[-1] if self.memory_history else None,
                'disk': list(self.disk_history)[-1] if self.disk_history else None,
                'network': list(self.network_history)[-1] if self.network_history else None
            },
            'history': {
                'cpu': list(self.cpu_history),
                'memory': list(self.memory_history),
                'disk': list(self.disk_history),
                'network': list(self.network_history)
            },
            'processes': self.process_history
        }
    
    def get_resource_stats(self) -> Dict:
        """
        리소스 사용량 통계를 계산합니다.
        
        Returns:
            Dict: 리소스 사용량 통계
        """
        return {
            'cpu': {
                'mean': sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0,
                'max': max(self.cpu_history) if self.cpu_history else 0,
                'min': min(self.cpu_history) if self.cpu_history else 0
            },
            'memory': {
                'mean': sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0,
                'max': max(self.memory_history) if self.memory_history else 0,
                'min': min(self.memory_history) if self.memory_history else 0
            },
            'disk': {
                'mean': sum(self.disk_history) / len(self.disk_history) if self.disk_history else 0,
                'max': max(self.disk_history) if self.disk_history else 0,
                'min': min(self.disk_history) if self.disk_history else 0
            }
        } 