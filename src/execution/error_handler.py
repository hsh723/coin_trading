"""
실행 시스템 오류 처리 모듈
"""

import logging
import traceback
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from src.execution.notifier import ExecutionNotifier

logger = logging.getLogger(__name__)

class ExecutionError(Exception):
    """실행 시스템 오류"""
    
    def __init__(
        self,
        message: str,
        error_type: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        오류 초기화
        
        Args:
            message (str): 오류 메시지
            error_type (str): 오류 타입
            details (Optional[Dict[str, Any]]): 상세 정보
        """
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}
        self.timestamp = datetime.now()

class ErrorHandler:
    """오류 처리기"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        오류 처리기 초기화
        
        Args:
            config (Dict[str, Any]): 설정
        """
        self.config = config
        self.error_stats = {
            'total_errors': 0,
            'error_types': {},
            'last_error': None
        }
        
        # 오류 처리 설정
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        
        # 오류 이력
        self.error_history = []
        self.max_history_size = config.get('max_history_size', 1000)
        
        # 오류 처리 콜백
        self.error_callbacks = {}
        
        # 알림 설정
        self.notifier = ExecutionNotifier(config)
        
    async def initialize(self):
        """오류 처리기 초기화"""
        try:
            # 알림 시스템 초기화
            await self.notifier.initialize()
            
            logger.info("오류 처리기 초기화 완료")
            
        except Exception as e:
            logger.error(f"오류 처리기 초기화 실패: {str(e)}")
            raise
            
    async def close(self):
        """리소스 정리"""
        try:
            # 알림 시스템 정리
            await self.notifier.close()
            
            logger.info("오류 처리기 종료")
            
        except Exception as e:
            logger.error(f"오류 처리기 종료 실패: {str(e)}")
            
    async def handle_execution_error(self, order_id: str, strategy: str, error: str):
        """
        실행 오류 처리
        
        Args:
            order_id (str): 주문 ID
            strategy (str): 전략 이름
            error (str): 오류 메시지
        """
        self.error_stats['total_errors'] += 1
        
        # 오류 유형 통계 업데이트
        error_type = error.split(':')[0]
        if error_type in self.error_stats['error_types']:
            self.error_stats['error_types'][error_type] += 1
        else:
            self.error_stats['error_types'][error_type] = 1
            
        # 마지막 오류 업데이트
        self.error_stats['last_error'] = {
            'order_id': order_id,
            'strategy': strategy,
            'error': error,
            'timestamp': datetime.now()
        }
        
    async def handle_error(self, error_type: str, details: Dict[str, Any]):
        """
        일반 오류 처리
        
        Args:
            error_type (str): 오류 유형
            details (Dict[str, Any]): 오류 상세 정보
        """
        self.error_stats['total_errors'] += 1
        
        # 오류 유형 통계 업데이트
        if error_type in self.error_stats['error_types']:
            self.error_stats['error_types'][error_type] += 1
        else:
            self.error_stats['error_types'][error_type] = 1
            
        # 마지막 오류 업데이트
        self.error_stats['last_error'] = {
            'type': error_type,
            'details': details,
            'timestamp': datetime.now()
        }
        
    async def get_error_stats(self) -> Dict[str, Any]:
        """
        오류 통계 조회
        
        Returns:
            Dict[str, Any]: 오류 통계
        """
        return self.error_stats
        
    def register_error_callback(
        self,
        error_type: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        오류 처리 콜백 등록
        
        Args:
            error_type (str): 오류 타입
            callback (Callable[[Dict[str, Any]], None]): 콜백 함수
        """
        self.error_callbacks[error_type] = callback
        
    def _create_error_info(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        오류 정보 생성
        
        Args:
            error (Exception): 오류
            context (Optional[Dict[str, Any]]): 컨텍스트
            
        Returns:
            Dict[str, Any]: 오류 정보
        """
        # 기본 오류 정보
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'message': str(error),
            'type': error.__class__.__name__,
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        # ExecutionError인 경우 추가 정보
        if isinstance(error, ExecutionError):
            error_info.update({
                'error_type': error.error_type,
                'details': error.details
            })
            
        return error_info
        
    def _record_error(self, error_info: Dict[str, Any]) -> None:
        """
        오류 이력 기록
        
        Args:
            error_info (Dict[str, Any]): 오류 정보
        """
        self.error_history.append(error_info)
        
        # 이력 크기 제한
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
            
    async def _execute_error_callbacks(
        self,
        error_info: Dict[str, Any]
    ) -> None:
        """
        오류 처리 콜백 실행
        
        Args:
            error_info (Dict[str, Any]): 오류 정보
        """
        error_type = error_info.get('error_type', error_info['type'])
        
        if error_type in self.error_callbacks:
            try:
                callback = self.error_callbacks[error_type]
                await callback(error_info)
            except Exception as e:
                logger.error(f"오류 처리 콜백 실행 실패: {str(e)}")
                
    def get_error_history(
        self,
        error_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> list:
        """
        오류 이력 조회
        
        Args:
            error_type (Optional[str]): 오류 타입
            start_time (Optional[datetime]): 시작 시간
            end_time (Optional[datetime]): 종료 시간
            
        Returns:
            list: 오류 이력
        """
        try:
            filtered_history = []
            
            for error in self.error_history:
                # 시간 필터링
                error_time = datetime.fromisoformat(error['timestamp'])
                if start_time and error_time < start_time:
                    continue
                if end_time and error_time > end_time:
                    continue
                    
                # 타입 필터링
                if error_type:
                    error_type_match = (
                        error.get('error_type', error['type']) == error_type
                    )
                    if not error_type_match:
                        continue
                        
                filtered_history.append(error)
                
            return filtered_history
            
        except Exception as e:
            logger.error(f"오류 이력 조회 실패: {str(e)}")
            return []
            
    def clear_error_history(self) -> None:
        """오류 이력 초기화"""
        self.error_history = [] 