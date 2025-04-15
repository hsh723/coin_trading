"""
실행 시스템 알림
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import aiohttp
from src.notification.telegram import TelegramNotifier

logger = logging.getLogger(__name__)

class ExecutionNotifier:
    """실행 시스템 알림"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        알림 시스템 초기화
        
        Args:
            config (Dict[str, Any]): 설정
        """
        self.config = config
        
        # 알림 설정
        notification_config = config.get('notification', {})
        self.enabled = notification_config.get('enabled', True)
        self.notification_types = notification_config.get('types', ['telegram'])
        
        # 알림 채널 설정
        self.telegram = None
        if 'telegram' in self.notification_types:
            telegram_config = notification_config.get('telegram', {})
            token = telegram_config.get('token')
            chat_id = telegram_config.get('chat_id')
            if token and chat_id:
                self.telegram = TelegramNotifier(token=token, chat_id=chat_id)
            
        # 알림 레벨 설정
        self.alert_levels = {
            'critical': notification_config.get('critical_threshold', 0.9),
            'warning': notification_config.get('warning_threshold', 0.7),
            'info': notification_config.get('info_threshold', 0.5)
        }
        
        # 알림 기록
        self.notification_history = []
        self.max_history_size = notification_config.get('max_history_size', 1000)
        
        # 알림 제한
        self.rate_limit = notification_config.get('rate_limit', 10)  # 분당 최대 알림 수
        self.rate_window = notification_config.get('rate_window', 60)  # 초
        self.notification_count = 0
        self.last_reset_time = datetime.now()
        
    async def initialize(self):
        """알림 시스템 초기화"""
        try:
            # 텔레그램 초기화
            if self.telegram:
                await self.telegram.initialize()
                
            logger.info("알림 시스템 초기화 완료")
            
        except Exception as e:
            logger.error(f"알림 시스템 초기화 실패: {str(e)}")
            
    async def close(self):
        """리소스 정리"""
        try:
            # 텔레그램 정리
            if self.telegram:
                await self.telegram.close()
                
            logger.info("알림 시스템 종료")
            
        except Exception as e:
            logger.error(f"알림 시스템 종료 실패: {str(e)}")
            
    async def notify_execution(
        self,
        order_id: str,
        execution_details: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ):
        """
        실행 알림 전송
        
        Args:
            order_id (str): 주문 ID
            execution_details (Dict[str, Any]): 실행 상세 정보
            performance_metrics (Dict[str, Any]): 성능 메트릭
        """
        try:
            if not self.enabled:
                return
                
            # 알림 레벨 결정
            level = self._determine_level(execution_details)
            
            # 알림 메시지 생성
            message = self._create_execution_message(execution_details, level)
            
            # 알림 전송
            await self._send_notification(message, level)
            
        except Exception as e:
            logger.error(f"실행 알림 실패: {str(e)}")
            
    async def notify_error(
        self,
        error_type: str,
        message: str,
        details: Dict[str, Any]
    ):
        """
        오류 알림 전송
        
        Args:
            error_type (str): 오류 유형
            message (str): 오류 메시지
            details (Dict[str, Any]): 오류 상세 정보
        """
        try:
            if not self.enabled:
                return
                
            # 알림 메시지 생성
            error_message = self._create_error_message(error_type, message, details)
            
            # 알림 전송
            await self._send_notification(error_message, 'critical')
            
        except Exception as e:
            logger.error(f"오류 알림 실패: {str(e)}")
            
    async def notify_performance(
        self,
        metrics: Dict[str, Any],
        message: str,
        level: str = 'info'
    ):
        """
        성능 알림 전송
        
        Args:
            metrics (Dict[str, Any]): 성능 메트릭
            message (str): 알림 메시지
            level (str): 알림 레벨
        """
        try:
            if not self.enabled:
                return
                
            # 알림 레벨 결정
            level = self._determine_performance_level(metrics)
            
            # 알림 메시지 생성
            performance_message = self._create_performance_message(metrics, level)
            
            # 알림 전송
            await self._send_notification(performance_message, level)
            
        except Exception as e:
            logger.error(f"성능 알림 실패: {str(e)}")
            
    async def _send_notification(
        self,
        message: str,
        level: str
    ) -> None:
        """
        알림 전송
        
        Args:
            message (str): 알림 메시지
            level (str): 알림 레벨
        """
        try:
            # 속도 제한 확인
            if not self._check_rate_limit():
                logger.warning("알림 속도 제한 초과")
                return
                
            # 텔레그램 알림
            if self.telegram:
                await self.telegram.send_message(message)
                
            # 알림 기록
            self._record_notification({
                'timestamp': datetime.now().isoformat(),
                'message': message,
                'level': level
            })
            
            logger.debug(f"알림 전송 완료: {level}")
            
        except Exception as e:
            logger.error(f"알림 전송 실패: {str(e)}")
            
    def _determine_level(
        self,
        execution_data: Dict[str, Any]
    ) -> str:
        """
        알림 레벨 결정
        
        Args:
            execution_data (Dict[str, Any]): 실행 데이터
            
        Returns:
            str: 알림 레벨
        """
        # 실행 실패
        if not execution_data.get('success', False):
            return 'critical'
            
        # 슬리피지 확인
        slippage = execution_data.get('slippage', 0)
        if slippage > self.alert_levels['critical']:
            return 'critical'
        elif slippage > self.alert_levels['warning']:
            return 'warning'
            
        return 'info'
        
    def _determine_performance_level(
        self,
        performance_data: Dict[str, Any]
    ) -> str:
        """
        성능 알림 레벨 결정
        
        Args:
            performance_data (Dict[str, Any]): 성능 데이터
            
        Returns:
            str: 알림 레벨
        """
        # 성공률 확인
        success_rate = performance_data.get('success_rate', 1.0)
        if success_rate < (1 - self.alert_levels['critical']):
            return 'critical'
        elif success_rate < (1 - self.alert_levels['warning']):
            return 'warning'
            
        # 지연 시간 확인
        latency = performance_data.get('latency', 0)
        if latency > 1000:  # 1초
            return 'critical'
        elif latency > 500:  # 500ms
            return 'warning'
            
        return 'info'
        
    def _create_execution_message(
        self,
        execution_data: Dict[str, Any],
        level: str
    ) -> str:
        """
        실행 알림 메시지 생성
        
        Args:
            execution_data (Dict[str, Any]): 실행 데이터
            level (str): 알림 레벨
            
        Returns:
            str: 알림 메시지
        """
        # 기본 정보
        message = [
            f"🔔 실행 알림 ({level.upper()})",
            f"주문 ID: {execution_data.get('order_id', 'unknown')}",
            f"상태: {'성공' if execution_data.get('success', False) else '실패'}"
        ]
        
        # 실행 정보
        if execution_data.get('success', False):
            message.extend([
                f"가격: {execution_data.get('price', 0):.2f}",
                f"수량: {execution_data.get('volume', 0):.4f}",
                f"슬리피지: {execution_data.get('slippage', 0):.2%}"
            ])
        else:
            message.extend([
                f"오류: {execution_data.get('error', 'unknown')}",
                f"상세: {execution_data.get('details', {})}"
            ])
            
        return "\n".join(message)
        
    def _create_error_message(
        self,
        error_type: str,
        message: str,
        details: Dict[str, Any]
    ) -> str:
        """
        오류 알림 메시지 생성
        
        Args:
            error_type (str): 오류 유형
            message (str): 오류 메시지
            details (Dict[str, Any]): 오류 상세 정보
            
        Returns:
            str: 알림 메시지
        """
        return f"⚠️ 오류 알림 (CRITICAL)\n메시지: {message}\n타입: {error_type}\n상세: {details}"
        
    def _create_performance_message(
        self,
        performance_data: Dict[str, Any],
        level: str
    ) -> str:
        """
        성능 알림 메시지 생성
        
        Args:
            performance_data (Dict[str, Any]): 성능 데이터
            level (str): 알림 레벨
            
        Returns:
            str: 알림 메시지
        """
        message = [
            f"📊 성능 알림 ({level.upper()})",
            f"지연시간: {performance_data.get('latency', 0):.2f}ms",
            f"성공률: {performance_data.get('success_rate', 0):.2%}",
            f"슬리피지: {performance_data.get('slippage', 0):.2%}"
        ]
        
        return "\n".join(message)
        
    def _check_rate_limit(self) -> bool:
        """
        알림 속도 제한 확인
        
        Returns:
            bool: 알림 가능 여부
        """
        now = datetime.now()
        
        # 시간 창 초기화
        if (now - self.last_reset_time).total_seconds() > self.rate_window:
            self.notification_count = 0
            self.last_reset_time = now
            
        # 속도 제한 확인
        if self.notification_count >= self.rate_limit:
            return False
            
        self.notification_count += 1
        return True
        
    def _record_notification(
        self,
        notification: Dict[str, Any]
    ) -> None:
        """
        알림 기록
        
        Args:
            notification (Dict[str, Any]): 알림 데이터
        """
        self.notification_history.append(notification)
        
        # 기록 크기 제한
        if len(self.notification_history) > self.max_history_size:
            self.notification_history = self.notification_history[-self.max_history_size:]
            
    def get_notification_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        알림 기록 조회
        
        Args:
            start_time (Optional[datetime]): 시작 시간
            end_time (Optional[datetime]): 종료 시간
            level (Optional[str]): 알림 레벨
            
        Returns:
            List[Dict[str, Any]]: 알림 기록
        """
        filtered_history = []
        
        for notification in self.notification_history:
            # 시간 필터링
            notification_time = datetime.fromisoformat(
                notification['timestamp']
            )
            if start_time and notification_time < start_time:
                continue
            if end_time and notification_time > end_time:
                continue
                
            # 레벨 필터링
            if level and notification['level'] != level:
                continue
                
            filtered_history.append(notification)
            
        return filtered_history 