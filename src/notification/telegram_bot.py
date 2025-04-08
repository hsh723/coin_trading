"""
텔레그램 봇 알림 클래스
"""

import requests
from typing import Optional
from src.utils.logger import get_logger

class TelegramNotifier:
    """텔레그램 봇 알림 클래스"""
    
    def __init__(self, token: str, chat_id: str):
        """
        초기화
        
        Args:
            token (str): 텔레그램 봇 토큰
            chat_id (str): 채팅 ID
        """
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.logger = get_logger(__name__)
        
    def send_message(self, message: str) -> bool:
        """
        메시지 전송
        
        Args:
            message (str): 전송할 메시지
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, data=data)
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            self.logger.error(f"텔레그램 메시지 전송 실패: {str(e)}")
            return False
            
    def send_alert(self, title: str, message: str, level: str = "info") -> bool:
        """
        알림 전송
        
        Args:
            title (str): 알림 제목
            message (str): 알림 메시지
            level (str): 알림 레벨 (info, warning, error)
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            emoji = {
                "info": "ℹ️",
                "warning": "⚠️",
                "error": "❌"
            }.get(level, "ℹ️")
            
            formatted_message = f"{emoji} <b>{title}</b>\n\n{message}"
            return self.send_message(formatted_message)
            
        except Exception as e:
            self.logger.error(f"텔레그램 알림 전송 실패: {str(e)}")
            return False 