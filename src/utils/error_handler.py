"""
에러 처리 모듈
"""

import logging
import time
from typing import Dict, Optional, Callable
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
from functools import wraps

class ErrorHandler:
    """에러 처리 클래스"""
    
    def __init__(self, db_manager):
        """
        초기화
        
        Args:
            db_manager: 데이터베이스 관리자
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.error_counts = {}
        self.last_error_time = {}
        self.recovery_attempts = {}
        
    def handle_api_error(self,
                        func: Callable,
                        max_retries: int = 3,
                        retry_delay: int = 5) -> Callable:
        """
        API 에러 처리 데코레이터
        
        Args:
            func (Callable): 대상 함수
            max_retries (int): 최대 재시도 횟수
            retry_delay (int): 재시도 간격(초)
            
        Returns:
            Callable: 데코레이터가 적용된 함수
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            last_error = None
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    last_error = e
                    retries += 1
                    self.logger.warning(
                        f"API 에러 발생: {str(e)}. "
                        f"재시도 {retries}/{max_retries}"
                    )
                    
                    if retries < max_retries:
                        time.sleep(retry_delay)
                        
            self.logger.error(f"API 호출 실패: {str(last_error)}")
            self._log_error('api_error', str(last_error))
            raise last_error
            
        return wrapper
        
    def handle_data_error(self,
                         func: Callable,
                         data_validator: Optional[Callable] = None) -> Callable:
        """
        데이터 에러 처리 데코레이터
        
        Args:
            func (Callable): 대상 함수
            data_validator (Optional[Callable]): 데이터 검증 함수
            
        Returns:
            Callable: 데코레이터가 적용된 함수
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                
                if data_validator and not data_validator(result):
                    raise ValueError("데이터 검증 실패")
                    
                return result
                
            except (ValueError, pd.errors.EmptyDataError) as e:
                self.logger.error(f"데이터 에러 발생: {str(e)}")
                self._log_error('data_error', str(e))
                raise
                
        return wrapper
        
    def handle_network_error(self,
                           func: Callable,
                           timeout: int = 30) -> Callable:
        """
        네트워크 에러 처리 데코레이터
        
        Args:
            func (Callable): 대상 함수
            timeout (int): 타임아웃 시간(초)
            
        Returns:
            Callable: 데코레이터가 적용된 함수
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (requests.exceptions.Timeout,
                   requests.exceptions.ConnectionError) as e:
                self.logger.error(f"네트워크 에러 발생: {str(e)}")
                self._log_error('network_error', str(e))
                raise
                
        return wrapper
        
    def _log_error(self, error_type: str, message: str):
        """
        에러 로깅
        
        Args:
            error_type (str): 에러 유형
            message (str): 에러 메시지
        """
        try:
            # 에러 카운트 업데이트
            if error_type not in self.error_counts:
                self.error_counts[error_type] = 0
            self.error_counts[error_type] += 1
            
            # 마지막 에러 시간 업데이트
            self.last_error_time[error_type] = datetime.now()
            
            # 에러 로그 저장
            self.db_manager.save_error_log({
                'timestamp': datetime.now(),
                'type': error_type,
                'message': message,
                'count': self.error_counts[error_type]
            })
            
        except Exception as e:
            self.logger.error(f"에러 로깅 실패: {str(e)}")
            
    def get_error_stats(self) -> Dict:
        """
        에러 통계 조회
        
        Returns:
            Dict: 에러 통계
        """
        try:
            stats = {}
            
            for error_type, count in self.error_counts.items():
                last_time = self.last_error_time.get(error_type)
                time_since_last = (
                    (datetime.now() - last_time).total_seconds()
                    if last_time else None
                )
                
                stats[error_type] = {
                    'count': count,
                    'last_occurrence': last_time,
                    'time_since_last': time_since_last
                }
                
            return stats
            
        except Exception as e:
            self.logger.error(f"에러 통계 조회 실패: {str(e)}")
            return {}
            
    def reset_error_counts(self):
        """에러 카운트 초기화"""
        try:
            self.error_counts = {}
            self.last_error_time = {}
            self.recovery_attempts = {}
            
        except Exception as e:
            self.logger.error(f"에러 카운트 초기화 실패: {str(e)}")
            
    def check_system_health(self) -> Dict:
        """
        시스템 상태 점검
        
        Returns:
            Dict: 시스템 상태
        """
        try:
            # API 응답 시간 확인
            api_response_time = self._check_api_response_time()
            
            # 데이터베이스 연결 확인
            db_connection = self._check_database_connection()
            
            # 리소스 사용량 확인
            resource_usage = self._check_resource_usage()
            
            return {
                'api_response_time': api_response_time,
                'database_connection': db_connection,
                'resource_usage': resource_usage,
                'error_stats': self.get_error_stats()
            }
            
        except Exception as e:
            self.logger.error(f"시스템 상태 점검 실패: {str(e)}")
            return {}
            
    def _check_api_response_time(self) -> float:
        """
        API 응답 시간 확인
        
        Returns:
            float: 응답 시간(초)
        """
        try:
            start_time = time.time()
            response = requests.get('https://api.example.com/health')
            response.raise_for_status()
            return time.time() - start_time
            
        except Exception as e:
            self.logger.error(f"API 응답 시간 확인 실패: {str(e)}")
            return float('inf')
            
    def _check_database_connection(self) -> bool:
        """
        데이터베이스 연결 확인
        
        Returns:
            bool: 연결 상태
        """
        try:
            self.db_manager.test_connection()
            return True
            
        except Exception as e:
            self.logger.error(f"데이터베이스 연결 확인 실패: {str(e)}")
            return False
            
    def _check_resource_usage(self) -> Dict:
        """
        리소스 사용량 확인
        
        Returns:
            Dict: 리소스 사용량
        """
        try:
            import psutil
            
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent
            
            return {
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'disk_usage': disk_usage
            }
            
        except Exception as e:
            self.logger.error(f"리소스 사용량 확인 실패: {str(e)}")
            return {} 