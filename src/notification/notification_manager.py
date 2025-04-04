"""
알림 관리 모듈
"""

import asyncio
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import telegram
from telegram.ext import Application
from dataclasses import dataclass
from ..utils.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

@dataclass
class NotificationRule:
    """알림 규칙 데이터 클래스"""
    name: str
    condition: str
    message: str
    priority: int
    enabled: bool
    notification_types: List[str]

class NotificationManager:
    """알림 관리 클래스"""
    
    def __init__(
        self,
        telegram_token: str,
        chat_id: str,
        database_manager: Optional[DatabaseManager] = None
    ):
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.database_manager = database_manager
        self.logger = logging.getLogger(__name__)
        
        # 텔레그램 봇 초기화
        self.bot = Application.builder().token(telegram_token).build()
        
        # 알림 규칙 초기화
        self.rules: List[NotificationRule] = []
        
        # 알림 우선순위 설정
        self.priority_levels = {
            'critical': 1,
            'high': 2,
            'medium': 3,
            'low': 4
        }
    
    async def add_rule(
        self,
        name: str,
        condition: str,
        message: str,
        priority: str = 'medium',
        enabled: bool = True,
        notification_types: List[str] = None
    ):
        """알림 규칙 추가"""
        try:
            rule = NotificationRule(
                name=name,
                condition=condition,
                message=message,
                priority=self.priority_levels[priority],
                enabled=enabled,
                notification_types=notification_types or ['telegram']
            )
            self.rules.append(rule)
            
            if self.database_manager:
                await self.database_manager.save_notification_rule(rule)
            
            self.logger.info(f"알림 규칙 추가: {name}")
            
        except Exception as e:
            self.logger.error(f"알림 규칙 추가 중 오류 발생: {str(e)}")
            raise
    
    async def remove_rule(self, name: str):
        """알림 규칙 제거"""
        try:
            self.rules = [rule for rule in self.rules if rule.name != name]
            
            if self.database_manager:
                await self.database_manager.delete_notification_rule(name)
            
            self.logger.info(f"알림 규칙 제거: {name}")
            
        except Exception as e:
            self.logger.error(f"알림 규칙 제거 중 오류 발생: {str(e)}")
            raise
    
    async def update_rule(
        self,
        name: str,
        condition: str = None,
        message: str = None,
        priority: str = None,
        enabled: bool = None,
        notification_types: List[str] = None
    ):
        """알림 규칙 업데이트"""
        try:
            for rule in self.rules:
                if rule.name == name:
                    if condition is not None:
                        rule.condition = condition
                    if message is not None:
                        rule.message = message
                    if priority is not None:
                        rule.priority = self.priority_levels[priority]
                    if enabled is not None:
                        rule.enabled = enabled
                    if notification_types is not None:
                        rule.notification_types = notification_types
                    
                    if self.database_manager:
                        await self.database_manager.update_notification_rule(rule)
                    
                    self.logger.info(f"알림 규칙 업데이트: {name}")
                    break
            
        except Exception as e:
            self.logger.error(f"알림 규칙 업데이트 중 오류 발생: {str(e)}")
            raise
    
    async def check_rules(self, data: Dict[str, Any]):
        """알림 규칙 확인"""
        try:
            for rule in self.rules:
                if not rule.enabled:
                    continue
                
                # 조건 평가
                if eval(rule.condition, {'data': data}):
                    # 알림 전송
                    await self.send_notification(rule, data)
            
        except Exception as e:
            self.logger.error(f"알림 규칙 확인 중 오류 발생: {str(e)}")
            raise
    
    async def send_notification(self, rule: NotificationRule, data: Dict[str, Any]):
        """알림 전송"""
        try:
            # 메시지 포맷팅
            message = rule.message.format(**data)
            
            # 우선순위에 따른 지연 시간 설정
            delay = (rule.priority - 1) * 5  # 우선순위가 낮을수록 더 늦게 전송
            
            # 알림 전송
            for notification_type in rule.notification_types:
                if notification_type == 'telegram':
                    await asyncio.sleep(delay)
                    await self._send_telegram_message(message)
                # 다른 알림 타입 추가 가능
            
            self.logger.info(f"알림 전송: {rule.name}")
            
        except Exception as e:
            self.logger.error(f"알림 전송 중 오류 발생: {str(e)}")
            raise
    
    async def _send_telegram_message(self, message: str):
        """텔레그램 메시지 전송"""
        try:
            await self.bot.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
        except Exception as e:
            self.logger.error(f"텔레그램 메시지 전송 중 오류 발생: {str(e)}")
            raise
    
    async def get_notification_history(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        rule_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """알림 내역 조회"""
        try:
            if self.database_manager:
                return await self.database_manager.get_notification_history(
                    start_date=start_date,
                    end_date=end_date,
                    rule_name=rule_name
                )
            return []
            
        except Exception as e:
            self.logger.error(f"알림 내역 조회 중 오류 발생: {str(e)}")
            raise
    
    async def clear_notification_history(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """알림 내역 삭제"""
        try:
            if self.database_manager:
                await self.database_manager.clear_notification_history(
                    start_date=start_date,
                    end_date=end_date
                )
            
        except Exception as e:
            self.logger.error(f"알림 내역 삭제 중 오류 발생: {str(e)}")
            raise 