"""
텔레그램 알림 시스템
"""

import telegram
from telegram.ext import Updater
import logging
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime

class TelegramNotifier:
    """텔레그램 알림 클래스"""
    
    def __init__(self, token: str, chat_id: str):
        """
        초기화
        
        Args:
            token (str): 텔레그램 봇 토큰
            chat_id (str): 채팅 ID
        """
        self.token = token
        self.chat_id = chat_id
        self.bot = telegram.Bot(token=token)
        self.logger = logging.getLogger(__name__)
        
    async def send_message(self, message: str) -> bool:
        """
        메시지 전송
        
        Args:
            message (str): 전송할 메시지
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
            return True
        except Exception as e:
            self.logger.error(f"텔레그램 메시지 전송 실패: {str(e)}")
            return False
            
    async def send_trade_signal(self, signal: Dict[str, Any]) -> bool:
        """
        거래 신호 알림 전송
        
        Args:
            signal (Dict[str, Any]): 거래 신호 정보
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            message = (
                f"<b>🚨 거래 신호 발생</b>\n\n"
                f"심볼: {signal['symbol']}\n"
                f"방향: {signal['direction']}\n"
                f"가격: {signal['price']}\n"
                f"강도: {signal['strength']}\n"
                f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            return await self.send_message(message)
        except Exception as e:
            self.logger.error(f"거래 신호 알림 전송 실패: {str(e)}")
            return False
            
    async def send_position_update(self, position: Dict[str, Any]) -> bool:
        """
        포지션 업데이트 알림 전송
        
        Args:
            position (Dict[str, Any]): 포지션 정보
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            message = (
                f"<b>📊 포지션 업데이트</b>\n\n"
                f"심볼: {position['symbol']}\n"
                f"방향: {position['side']}\n"
                f"수량: {position['size']}\n"
                f"진입가: {position['entry_price']}\n"
                f"현재가: {position['current_price']}\n"
                f"손익: {position['pnl']}\n"
                f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            return await self.send_message(message)
        except Exception as e:
            self.logger.error(f"포지션 업데이트 알림 전송 실패: {str(e)}")
            return False
            
    async def send_error_alert(self, error: Dict[str, Any]) -> bool:
        """
        에러 알림 전송
        
        Args:
            error (Dict[str, Any]): 에러 정보
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            message = (
                f"<b>⚠️ 에러 발생</b>\n\n"
                f"유형: {error['type']}\n"
                f"메시지: {error['message']}\n"
                f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            return await self.send_message(message)
        except Exception as e:
            self.logger.error(f"에러 알림 전송 실패: {str(e)}")
            return False
            
    async def send_performance_report(self, report: Dict[str, Any]) -> bool:
        """
        성과 리포트 전송
        
        Args:
            report (Dict[str, Any]): 성과 리포트 정보
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            message = (
                f"<b>📈 성과 리포트</b>\n\n"
                f"기간: {report['period']}\n"
                f"총 거래: {report['total_trades']}\n"
                f"승률: {report['win_rate']}%\n"
                f"수익률: {report['return']}%\n"
                f"샤프 비율: {report['sharpe_ratio']}\n"
                f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            return await self.send_message(message)
        except Exception as e:
            self.logger.error(f"성과 리포트 전송 실패: {str(e)}")
            return False 