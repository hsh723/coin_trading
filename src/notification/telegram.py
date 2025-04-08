import os
import logging
from typing import Optional
from telegram import Bot
from telegram.error import TelegramError

class TelegramNotifier:
    """텔레그램을 통한 알림 전송 클래스"""
    
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        초기화
        
        Args:
            token: 텔레그램 봇 토큰
            chat_id: 텔레그램 채팅 ID
        """
        self.token = token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.bot = Bot(token=self.token) if self.token else None
        self.logger = logging.getLogger(__name__)
        
    async def send_message(self, message: str) -> bool:
        """
        메시지 전송
        
        Args:
            message: 전송할 메시지
            
        Returns:
            bool: 전송 성공 여부
        """
        if not self.bot or not self.chat_id:
            self.logger.warning("텔레그램 봇 설정이 완료되지 않았습니다.")
            return False
            
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message)
            return True
        except TelegramError as e:
            self.logger.error(f"텔레그램 메시지 전송 실패: {str(e)}")
            return False
            
    async def send_alert(self, title: str, content: str, level: str = 'INFO') -> bool:
        """
        알림 전송
        
        Args:
            title: 알림 제목
            content: 알림 내용
            level: 알림 레벨 (INFO, WARNING, ERROR)
            
        Returns:
            bool: 전송 성공 여부
        """
        emoji = {
            'INFO': '📝',
            'WARNING': '⚠️',
            'ERROR': '🚨'
        }
        
        message = f"{emoji.get(level, '📝')} {level}\n\n"
        message += f"제목: {title}\n"
        message += f"내용: {content}"
        
        return await self.send_message(message) 