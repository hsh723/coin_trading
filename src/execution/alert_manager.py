"""
실행 시스템 알림 관리 모듈
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from src.notification.telegram import TelegramNotifier
from src.utils.config_loader import get_config

logger = logging.getLogger(__name__)

class AlertManager:
    """알림 관리자"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        알림 관리자 초기화
        
        Args:
            config (Dict[str, Any]): 설정
        """
        self.config = config
        
        # 알림 설정
        self.alert_config = config['alerts']
        self.enabled = self.alert_config.get('enabled', True)
        self.rate_limit = self.alert_config.get('rate_limit', 60)  # 초당 알림 제한
        self.cooldown = self.alert_config.get('cooldown', 300)  # 동일 알림 대기 시간
        
        # 알림 채널
        self.telegram = TelegramNotifier(config)
        
        # 알림 이력
        self.alert_history = []
        self.last_alert_time = {}
        
        # 알림 레벨 임계값
        self.thresholds = {
            'latency': self.alert_config.get('latency_threshold', 1000),  # ms
            'error_rate': self.alert_config.get('error_rate_threshold', 0.05),
            'fill_rate': self.alert_config.get('fill_rate_threshold', 0.95),
            'slippage': self.alert_config.get('slippage_threshold', 0.001),
            'volume': self.alert_config.get('volume_threshold', 100.0)
        }
        
    async def initialize(self):
        """알림 관리자 초기화"""
        try:
            # 텔레그램 초기화
            await self.telegram.initialize()
            
            logger.info("알림 관리자 초기화 완료")
            
        except Exception as e:
            logger.error(f"알림 관리자 초기화 실패: {str(e)}")
            raise
            
    async def close(self):
        """리소스 정리"""
        try:
            await self.telegram.close()
            logger.info("알림 관리자 종료")
        except Exception as e:
            logger.error(f"알림 관리자 종료 실패: {str(e)}")
            
    async def process_metrics(self, metrics: Dict[str, float]):
        """
        메트릭 처리 및 알림 생성
        
        Args:
            metrics (Dict[str, float]): 성능 메트릭
        """
        try:
            if not self.enabled:
                return
                
            # 알림 레벨 결정
            alert_level = self._determine_alert_level(metrics)
            
            if alert_level != 'normal':
                # 알림 메시지 생성
                message = self._create_alert_message(metrics, alert_level)
                
                # 알림 전송
                await self._send_alert(message, alert_level)
                
        except Exception as e:
            logger.error(f"메트릭 처리 실패: {str(e)}")
            
    def _determine_alert_level(self, metrics: Dict[str, float]) -> str:
        """
        알림 레벨 결정
        
        Args:
            metrics (Dict[str, float]): 성능 메트릭
            
        Returns:
            str: 알림 레벨 (critical, warning, normal)
        """
        try:
            # 임계값 확인
            if (
                metrics['latency'] > self.thresholds['latency'] * 2 or
                metrics['error_rate'] > self.thresholds['error_rate'] * 2 or
                metrics['fill_rate'] < self.thresholds['fill_rate'] / 2
            ):
                return 'critical'
                
            elif (
                metrics['latency'] > self.thresholds['latency'] or
                metrics['error_rate'] > self.thresholds['error_rate'] or
                metrics['fill_rate'] < self.thresholds['fill_rate'] or
                metrics['slippage'] > self.thresholds['slippage']
            ):
                return 'warning'
                
            return 'normal'
            
        except Exception as e:
            logger.error(f"알림 레벨 결정 실패: {str(e)}")
            return 'normal'
            
    def _create_alert_message(
        self,
        metrics: Dict[str, float],
        level: str
    ) -> str:
        """
        알림 메시지 생성
        
        Args:
            metrics (Dict[str, float]): 성능 메트릭
            level (str): 알림 레벨
            
        Returns:
            str: 알림 메시지
        """
        try:
            # 메시지 템플릿
            template = {
                'critical': '🚨 심각: 실행 시스템 성능 저하',
                'warning': '⚠️ 경고: 실행 시스템 성능 이상'
            }
            
            # 메시지 생성
            message = f"{template[level]}\n\n"
            message += f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            message += f"지연 시간: {metrics['latency']:.2f}ms\n"
            message += f"에러율: {metrics['error_rate']*100:.2f}%\n"
            message += f"체결률: {metrics['fill_rate']*100:.2f}%\n"
            message += f"슬리피지: {metrics['slippage']*100:.4f}%\n"
            message += f"처리량: {metrics['throughput']:.2f} TPS"
            
            return message
            
        except Exception as e:
            logger.error(f"알림 메시지 생성 실패: {str(e)}")
            return "실행 시스템 성능 이상"
            
    async def _send_alert(self, message: str, level: str):
        """
        알림 전송
        
        Args:
            message (str): 알림 메시지
            level (str): 알림 레벨
        """
        try:
            # 알림 제한 확인
            if not self._check_rate_limit(level):
                return
                
            # 알림 전송
            await self.telegram.send_message(message)
            
            # 알림 이력 기록
            self.alert_history.append({
                'timestamp': datetime.now(),
                'level': level,
                'message': message
            })
            
            # 이력 크기 제한
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]
                
            # 마지막 알림 시간 업데이트
            self.last_alert_time[level] = datetime.now()
            
        except Exception as e:
            logger.error(f"알림 전송 실패: {str(e)}")
            
    def _check_rate_limit(self, level: str) -> bool:
        """
        알림 제한 확인
        
        Args:
            level (str): 알림 레벨
            
        Returns:
            bool: 알림 전송 가능 여부
        """
        try:
            now = datetime.now()
            
            # 마지막 알림 시간 확인
            if level in self.last_alert_time:
                last_time = self.last_alert_time[level]
                
                # 대기 시간 계산
                if level == 'critical':
                    wait_time = self.rate_limit
                else:
                    wait_time = self.cooldown
                    
                # 제한 확인
                if (now - last_time).total_seconds() < wait_time:
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"알림 제한 확인 실패: {str(e)}")
            return False
            
    def get_alert_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        알림 이력 조회
        
        Args:
            start_time (Optional[datetime]): 시작 시간
            end_time (Optional[datetime]): 종료 시간
            level (Optional[str]): 알림 레벨
            
        Returns:
            List[Dict[str, Any]]: 알림 이력
        """
        try:
            # 시간 범위 필터링
            filtered_history = self.alert_history
            
            if start_time:
                filtered_history = [
                    alert for alert in filtered_history
                    if alert['timestamp'] >= start_time
                ]
                
            if end_time:
                filtered_history = [
                    alert for alert in filtered_history
                    if alert['timestamp'] <= end_time
                ]
                
            # 레벨 필터링
            if level:
                filtered_history = [
                    alert for alert in filtered_history
                    if alert['level'] == level
                ]
                
            return filtered_history
            
        except Exception as e:
            logger.error(f"알림 이력 조회 실패: {str(e)}")
            return []
            
    def get_alert_stats(self) -> Dict[str, Any]:
        """
        알림 통계 조회
        
        Returns:
            Dict[str, Any]: 알림 통계
        """
        try:
            # 24시간 이내 알림
            now = datetime.now()
            start_time = now - timedelta(days=1)
            
            daily_alerts = self.get_alert_history(start_time=start_time)
            
            return {
                'total_alerts': len(self.alert_history),
                'daily_alerts': len(daily_alerts),
                'critical_alerts': len([
                    alert for alert in daily_alerts
                    if alert['level'] == 'critical'
                ]),
                'warning_alerts': len([
                    alert for alert in daily_alerts
                    if alert['level'] == 'warning'
                ])
            }
            
        except Exception as e:
            logger.error(f"알림 통계 조회 실패: {str(e)}")
            return {
                'total_alerts': 0,
                'daily_alerts': 0,
                'critical_alerts': 0,
                'warning_alerts': 0
            } 