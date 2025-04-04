"""
텔레그램 알림 시스템
"""

import asyncio
import logging
from typing import Optional, Dict, Any
import telegram
from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv

# 로거 설정
logger = logging.getLogger(__name__)

class TelegramNotifier:
    """텔레그램 알림 시스템 클래스"""
    
    def __init__(self):
        """초기화"""
        # 환경 변수 로드
        load_dotenv()
        
        # 텔레그램 설정
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.bot = None if not self.bot_token else telegram.Bot(self.bot_token)
        
        # 알림 설정
        self.enabled = False
        self.notification_types = set()
        self.min_interval = 5  # 기본 5분
        self.last_notification = {}
    
    def setup(self, enabled: bool, notification_types: set, min_interval: int = 5):
        """
        알림 설정
        
        Args:
            enabled (bool): 알림 활성화 여부
            notification_types (set): 알림 유형 목록
            min_interval (int): 최소 알림 간격 (분)
        """
        self.enabled = enabled
        self.notification_types = notification_types
        self.min_interval = min_interval
    
    async def send_message(self, message: str, notification_type: str = None) -> bool:
        """
        메시지 전송
        
        Args:
            message (str): 전송할 메시지
            notification_type (str): 알림 유형
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            if not self.enabled or not self.bot:
                return False
            
            # 알림 유형 확인
            if notification_type and notification_type not in self.notification_types:
                return False
            
            # 알림 간격 확인
            now = datetime.now()
            if notification_type in self.last_notification:
                time_diff = (now - self.last_notification[notification_type]).total_seconds() / 60
                if time_diff < self.min_interval:
                    return False
            
            # 메시지 전송
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
            
            # 마지막 알림 시간 업데이트
            if notification_type:
                self.last_notification[notification_type] = now
            
            return True
            
        except Exception as e:
            logger.error(f"텔레그램 메시지 전송 실패: {str(e)}")
            return False
    
    async def send_trade_signal(self, signal: Dict[str, Any]) -> bool:
        """
        거래 신호 알림
        
        Args:
            signal (Dict[str, Any]): 거래 신호 정보
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            message = (
                f"🔔 <b>거래 신호</b>\n\n"
                f"심볼: {signal['symbol']}\n"
                f"방향: {'매수 🟢' if signal['side'] == 'buy' else '매도 🔴'}\n"
                f"가격: ${signal['price']:,.2f}\n"
                f"시간: {signal['timestamp']}\n"
                f"신뢰도: {signal['confidence']:.1f}%"
            )
            
            return await self.send_message(message, 'trade_signal')
            
        except Exception as e:
            logger.error(f"거래 신호 알림 전송 실패: {str(e)}")
            return False
    
    async def send_position_update(self, position: Dict[str, Any]) -> bool:
        """
        포지션 업데이트 알림
        
        Args:
            position (Dict[str, Any]): 포지션 정보
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            message = (
                f"📊 <b>포지션 업데이트</b>\n\n"
                f"심볼: {position['symbol']}\n"
                f"상태: {position['status']}\n"
                f"수익률: {position['pnl_pct']:.2%}\n"
                f"수익금: ${position['pnl']:,.2f}"
            )
            
            return await self.send_message(message, 'position_update')
            
        except Exception as e:
            logger.error(f"포지션 업데이트 알림 전송 실패: {str(e)}")
            return False
    
    async def send_daily_report(self, report: Dict[str, Any]) -> bool:
        """
        일일 리포트 알림
        
        Args:
            report (Dict[str, Any]): 일일 성과 리포트
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            message = (
                f"📈 <b>일일 거래 리포트</b>\n\n"
                f"날짜: {report['date']}\n"
                f"총 거래: {report['total_trades']}건\n"
                f"승률: {report['win_rate']:.1f}%\n"
                f"수익률: {report['return_pct']:.2%}\n"
                f"수익금: ${report['pnl']:,.2f}\n\n"
                f"상세 내역:\n"
                f"- 승리: {report['winning_trades']}건\n"
                f"- 패배: {report['losing_trades']}건\n"
                f"- 최대 수익: ${report['max_profit']:,.2f}\n"
                f"- 최대 손실: ${report['max_loss']:,.2f}"
            )
            
            return await self.send_message(message, 'daily_report')
            
        except Exception as e:
            logger.error(f"일일 리포트 알림 전송 실패: {str(e)}")
            return False
    
    async def send_error(self, error: str) -> bool:
        """
        에러 알림
        
        Args:
            error (str): 에러 메시지
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            message = (
                f"⚠️ <b>에러 발생</b>\n\n"
                f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"내용: {error}"
            )
            
            return await self.send_message(message, 'error')
            
        except Exception as e:
            logger.error(f"에러 알림 전송 실패: {str(e)}")
            return False

# 전역 인스턴스 생성
telegram_notifier = TelegramNotifier() 