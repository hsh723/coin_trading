import psutil
import requests
import time
from typing import Dict, List, Optional
from datetime import datetime
from ..utils.logger import setup_logger

logger = setup_logger()

class HealthChecker:
    """
    시스템의 전반적인 상태를 확인하는 도구
    
    Attributes:
        check_interval (int): 상태 확인 간격 (초)
        max_retries (int): 최대 재시도 횟수
        timeout (int): 요청 타임아웃 (초)
    """
    
    def __init__(
        self,
        check_interval: int = 60,
        max_retries: int = 3,
        timeout: int = 5
    ):
        self.check_interval = check_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.last_check = None
        self.health_status = {}
    
    def check_system_health(self) -> Dict:
        """
        시스템의 전반적인 상태를 확인합니다.
        
        Returns:
            Dict: 시스템 상태 정보
        """
        try:
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'system': self._check_system_status(),
                'network': self._check_network_status(),
                'processes': self._check_process_status(),
                'services': self._check_service_status()
            }
            
            self.health_status = health_status
            self.last_check = datetime.now()
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _check_system_status(self) -> Dict:
        """시스템 기본 상태를 확인합니다."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                'uptime': time.time() - psutil.boot_time()
            }
        except Exception as e:
            logger.error(f"System status check failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _check_network_status(self) -> Dict:
        """네트워크 상태를 확인합니다."""
        try:
            net_io = psutil.net_io_counters()
            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'connections': len(psutil.net_connections())
            }
        except Exception as e:
            logger.error(f"Network status check failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _check_process_status(self) -> Dict:
        """주요 프로세스 상태를 확인합니다."""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return {
                'total_processes': len(processes),
                'top_cpu_processes': sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)[:5],
                'top_memory_processes': sorted(processes, key=lambda x: x['memory_percent'], reverse=True)[:5]
            }
        except Exception as e:
            logger.error(f"Process status check failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _check_service_status(self) -> Dict:
        """주요 서비스 상태를 확인합니다."""
        try:
            services = {
                'trading_system': self._check_trading_system(),
                'database': self._check_database(),
                'api': self._check_api()
            }
            return services
        except Exception as e:
            logger.error(f"Service status check failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _check_trading_system(self) -> Dict:
        """거래 시스템 상태를 확인합니다."""
        try:
            # 거래 시스템 프로세스 확인
            trading_process = None
            for proc in psutil.process_iter(['pid', 'name']):
                if 'trading' in proc.info['name'].lower():
                    trading_process = proc
                    break
            
            if trading_process:
                return {
                    'status': 'running',
                    'pid': trading_process.info['pid'],
                    'uptime': time.time() - trading_process.create_time()
                }
            else:
                return {'status': 'not_running'}
                
        except Exception as e:
            logger.error(f"Trading system check failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _check_database(self) -> Dict:
        """데이터베이스 상태를 확인합니다."""
        try:
            # 데이터베이스 연결 테스트
            # 실제 구현에서는 데이터베이스 연결 테스트 코드 추가
            return {'status': 'connected'}
        except Exception as e:
            logger.error(f"Database check failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _check_api(self) -> Dict:
        """API 상태를 확인합니다."""
        try:
            # API 엔드포인트 테스트
            response = requests.get('http://localhost:8501', timeout=self.timeout)
            return {
                'status': 'available',
                'response_time': response.elapsed.total_seconds(),
                'status_code': response.status_code
            }
        except Exception as e:
            logger.error(f"API check failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def is_healthy(self) -> bool:
        """
        시스템이 정상 상태인지 확인합니다.
        
        Returns:
            bool: 시스템 상태
        """
        if not self.health_status:
            return False
            
        # 시스템 리소스 체크
        system_status = self.health_status.get('system', {})
        if (system_status.get('cpu_percent', 100) > 90 or
            system_status.get('memory_percent', 100) > 90 or
            system_status.get('disk_usage', 100) > 90):
            return False
            
        # 서비스 상태 체크
        services = self.health_status.get('services', {})
        for service, status in services.items():
            if status.get('status') != 'running' and status.get('status') != 'connected':
                return False
        
        return True
    
    def get_health_report(self) -> Dict:
        """
        상세한 건강 상태 리포트를 생성합니다.
        
        Returns:
            Dict: 건강 상태 리포트
        """
        if not self.health_status:
            self.check_system_health()
            
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy' if self.is_healthy() else 'unhealthy',
            'details': self.health_status,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """시스템 상태에 따른 권장사항을 생성합니다."""
        recommendations = []
        
        if not self.health_status:
            return recommendations
            
        system_status = self.health_status.get('system', {})
        
        # CPU 사용량 권장사항
        if system_status.get('cpu_percent', 0) > 80:
            recommendations.append("CPU 사용량이 높습니다. 프로세스를 최적화하거나 리소스를 추가하세요.")
            
        # 메모리 사용량 권장사항
        if system_status.get('memory_percent', 0) > 80:
            recommendations.append("메모리 사용량이 높습니다. 메모리 누수를 확인하거나 메모리를 추가하세요.")
            
        # 디스크 사용량 권장사항
        if system_status.get('disk_usage', 0) > 80:
            recommendations.append("디스크 공간이 부족합니다. 불필요한 파일을 정리하거나 스토리지를 추가하세요.")
        
        return recommendations 