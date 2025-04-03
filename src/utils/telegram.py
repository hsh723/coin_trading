"""
텔레그램 알림 모듈
"""

import requests
import logging
from typing import Optional
from .config import config_manager
from .logger import logger

class TelegramNotifier:
    def __init__(self):
        """텔레그램 알림기 초기화"""
        self.bot_token = config_manager.get_config('TELEGRAM_BOT_TOKEN')
        self.chat_id = config_manager.get_config('TELEGRAM_CHAT_ID')
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        if not self.bot_token or not self.chat_id:
            logger.warning("텔레그램 설정이 없습니다. 알림이 비활성화됩니다.")
    
    def send_message(self, message: str) -> bool:
        """
        텔레그램 메시지 전송
        
        Args:
            message (str): 전송할 메시지
            
        Returns:
            bool: 전송 성공 여부
        """
        if not self.bot_token or not self.chat_id:
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, json=data)
            response.raise_for_status()
            
            logger.debug(f"텔레그램 메시지 전송 성공: {message}")
            return True
            
        except Exception as e:
            logger.error(f"텔레그램 메시지 전송 실패: {str(e)}")
            return False
    
    def send_error(self, error: Exception, context: Optional[str] = None) -> bool:
        """
        오류 알림 전송
        
        Args:
            error (Exception): 발생한 오류
            context (Optional[str]): 오류 컨텍스트
            
        Returns:
            bool: 전송 성공 여부
        """
        message = f"❌ <b>오류 발생</b>\n\n"
        
        if context:
            message += f"<b>컨텍스트:</b> {context}\n"
        
        message += f"<b>오류 유형:</b> {type(error).__name__}\n"
        message += f"<b>오류 메시지:</b> {str(error)}"
        
        return self.send_message(message)
    
    def send_warning(self, warning: str, context: Optional[str] = None) -> bool:
        """
        경고 알림 전송
        
        Args:
            warning (str): 경고 메시지
            context (Optional[str]): 경고 컨텍스트
            
        Returns:
            bool: 전송 성공 여부
        """
        message = f"⚠️ <b>경고</b>\n\n"
        
        if context:
            message += f"<b>컨텍스트:</b> {context}\n"
        
        message += f"<b>경고 내용:</b> {warning}"
        
        return self.send_message(message)
    
    def send_info(self, info: str, context: Optional[str] = None) -> bool:
        """
        정보 알림 전송
        
        Args:
            info (str): 정보 메시지
            context (Optional[str]): 정보 컨텍스트
            
        Returns:
            bool: 전송 성공 여부
        """
        message = f"ℹ️ <b>정보</b>\n\n"
        
        if context:
            message += f"<b>컨텍스트:</b> {context}\n"
        
        message += f"<b>내용:</b> {info}"
        
        return self.send_message(message) 