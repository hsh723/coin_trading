"""
텔레그램 알림 모듈
"""

import asyncio
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    ConversationHandler
)
from src.utils.logger import setup_logger
from src.utils.config_loader import get_config
from src.trading.execution import OrderExecutor

# 대화 상태 정의
WAITING_FOR_CONFIRMATION = 1
WAITING_FOR_AMOUNT = 2
WAITING_FOR_PRICE = 3

class TelegramNotifier:
    """
    텔레그램 알림 클래스
    """
    
    def __init__(self, config: Dict[str, Any] = None, bot_token: str = None, chat_id: str = None):
        """
        텔레그램 알림 초기화
        
        Args:
            config (Dict[str, Any], optional): 설정 정보
            bot_token (str, optional): 텔레그램 봇 토큰
            chat_id (str, optional): 텔레그램 채팅 ID
        """
        self.config = config or {}
        
        # 매개변수로 전달된 토큰과 채팅 ID 사용
        self.token = bot_token
        self.chat_id = chat_id
        
        # 환경 변수에서 토큰과 채팅 ID를 가져오는 코드
        if not self.token:
            self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        if not self.chat_id:
            self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        
        # 토큰과 채팅 ID가 설정되었는지 확인
        if not self.token or not self.chat_id:
            print("텔레그램 알림을 사용하려면 봇 토큰과 채팅 ID를 설정하세요.")
        
        self.app = None
        self.logger = setup_logger()
        self.logger.info("TelegramNotifier initialized")
    
    async def initialize(self):
        """
        텔레그램 봇 초기화
        """
        try:
            if not self.token or not self.chat_id:
                raise ValueError("텔레그램 봇 토큰과 채팅 ID가 설정되지 않았습니다. 환경 변수를 확인하세요.")
            
            self.app = Application.builder().token(self.token).build()
            self.logger.info("Telegram bot initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Telegram bot: {str(e)}")
            raise
    
    async def send_message(self, message: str) -> bool:
        """
        메시지 전송
        
        Args:
            message (str): 전송할 메시지
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            if not self.app:
                await self.initialize()
            
            await self.app.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
            
            self.logger.info(f"Message sent: {message[:50]}...")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending message: {str(e)}")
            return False
    
    async def send_entry_signal(self, signal_data: Dict[str, Any], chart_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        진입 신호 전송
        
        Args:
            signal_data (Dict[str, Any]): 신호 데이터
            chart_data (Dict[str, Any], optional): 차트 데이터
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            # 메시지 포맷팅
            message = self._format_entry_signal(signal_data)
            
            # 메시지 전송
            success = await self.send_message(message)
            
            # 차트 이미지가 있는 경우 전송
            if success and chart_data:
                await self._send_chart_image(chart_data)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending entry signal: {str(e)}")
            return False
    
    async def send_exit_signal(self, signal_data: Dict[str, Any], chart_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        청산 신호 전송
        
        Args:
            signal_data (Dict[str, Any]): 신호 데이터
            chart_data (Dict[str, Any], optional): 차트 데이터
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            # 메시지 포맷팅
            message = self._format_exit_signal(signal_data)
            
            # 메시지 전송
            success = await self.send_message(message)
            
            # 차트 이미지가 있는 경우 전송
            if success and chart_data:
                await self._send_chart_image(chart_data)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending exit signal: {str(e)}")
            return False
    
    async def send_daily_report(self, performance_data: Dict[str, Any]) -> bool:
        """
        일일 리포트 전송
        
        Args:
            performance_data (Dict[str, Any]): 성과 데이터
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            # 메시지 포맷팅
            message = self._format_daily_report(performance_data)
            
            # 메시지 전송
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending daily report: {str(e)}")
            return False
    
    async def send_risk_alert(self, alert_type: str, details: Dict[str, Any]) -> bool:
        """
        위험 알림 전송
        
        Args:
            alert_type (str): 알림 유형
            details (Dict[str, Any]): 상세 정보
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            # 메시지 포맷팅
            message = self._format_risk_alert(alert_type, details)
            
            # 메시지 전송
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending risk alert: {str(e)}")
            return False
    
    def _format_entry_signal(self, signal_data: Dict[str, Any]) -> str:
        """
        진입 신호 메시지 포맷팅
        
        Args:
            signal_data (Dict[str, Any]): 신호 데이터
            
        Returns:
            str: 포맷팅된 메시지
        """
        symbol = signal_data.get('symbol', 'Unknown')
        signal_type = signal_data.get('type', 'Unknown')
        price = signal_data.get('price', 0.0)
        amount = signal_data.get('amount', 0.0)
        stop_loss = signal_data.get('stop_loss', 0.0)
        take_profit = signal_data.get('take_profit', 0.0)
        
        # 진입 이유
        trend_reason = signal_data.get('trend_reason', '')
        indicator_reason = signal_data.get('indicator_reason', '')
        pattern_reason = signal_data.get('pattern_reason', '')
        
        # 메시지 포맷팅
        message = (
            f"🔔 <b>진입 신호</b>\n\n"
            f"심볼: <code>{symbol}</code>\n"
            f"유형: <code>{signal_type}</code>\n"
            f"가격: <code>{price:.2f}</code>\n"
            f"수량: <code>{amount:.4f}</code>\n"
            f"손절가: <code>{stop_loss:.2f}</code>\n"
            f"익절가: <code>{take_profit:.2f}</code>\n\n"
            f"<b>진입 이유:</b>\n"
        )
        
        if trend_reason:
            message += f"• 추세: {trend_reason}\n"
        if indicator_reason:
            message += f"• 지표: {indicator_reason}\n"
        if pattern_reason:
            message += f"• 패턴: {pattern_reason}\n"
        
        message += f"\n시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return message
    
    def _format_exit_signal(self, signal_data: Dict[str, Any]) -> str:
        """
        청산 신호 메시지 포맷팅
        
        Args:
            signal_data (Dict[str, Any]): 신호 데이터
            
        Returns:
            str: 포맷팅된 메시지
        """
        symbol = signal_data.get('symbol', 'Unknown')
        position_type = signal_data.get('position_type', 'Unknown')
        entry_price = signal_data.get('entry_price', 0.0)
        exit_price = signal_data.get('exit_price', 0.0)
        amount = signal_data.get('amount', 0.0)
        pnl = signal_data.get('pnl', 0.0)
        pnl_percentage = signal_data.get('pnl_percentage', 0.0)
        
        # 청산 이유
        exit_reason = signal_data.get('exit_reason', '')
        
        # 메시지 포맷팅
        message = (
            f"🔔 <b>청산 신호</b>\n\n"
            f"심볼: <code>{symbol}</code>\n"
            f"포지션: <code>{position_type}</code>\n"
            f"진입가: <code>{entry_price:.2f}</code>\n"
            f"청산가: <code>{exit_price:.2f}</code>\n"
            f"수량: <code>{amount:.4f}</code>\n"
            f"손익: <code>{pnl:.2f} USDT ({pnl_percentage:.2f}%)</code>\n\n"
        )
        
        if exit_reason:
            message += f"<b>청산 이유:</b> {exit_reason}\n\n"
        
        message += f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return message
    
    def _format_daily_report(self, performance_data: Dict[str, Any]) -> str:
        """
        일일 리포트 메시지 포맷팅
        
        Args:
            performance_data (Dict[str, Any]): 성과 데이터
            
        Returns:
            str: 포맷팅된 메시지
        """
        date = performance_data.get('date', datetime.now().strftime('%Y-%m-%d'))
        total_trades = performance_data.get('total_trades', 0)
        win_rate = performance_data.get('win_rate', 0.0)
        return_value = performance_data.get('return', 0.0)
        pnl = performance_data.get('pnl', 0.0)
        long_positions = performance_data.get('long_positions', 0)
        short_positions = performance_data.get('short_positions', 0)
        max_drawdown = performance_data.get('max_drawdown', 0.0)
        sharpe_ratio = performance_data.get('sharpe_ratio', 0.0)
        calmar_ratio = performance_data.get('calmar_ratio', 0.0)
        
        # 메시지 포맷팅
        message = (
            f"📊 <b>일일 리포트 ({date})</b>\n\n"
            f"총 거래: <code>{total_trades}</code>\n"
            f"승률: <code>{win_rate:.2f}%</code>\n"
            f"수익률: <code>{return_value:.2f}%</code>\n"
            f"손익: <code>{pnl:.2f} USDT</code>\n"
            f"롱 포지션: <code>{long_positions}</code>\n"
            f"숏 포지션: <code>{short_positions}</code>\n"
            f"최대 낙폭: <code>{max_drawdown:.2f}%</code>\n"
            f"샤프 비율: <code>{sharpe_ratio:.2f}</code>\n"
            f"캘마 비율: <code>{calmar_ratio:.2f}</code>\n\n"
            f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        return message
    
    def _format_risk_alert(self, alert_type: str, details: Dict[str, Any]) -> str:
        """
        위험 알림 메시지 포맷팅
        
        Args:
            alert_type (str): 알림 유형
            details (Dict[str, Any]): 상세 정보
            
        Returns:
            str: 포맷팅된 메시지
        """
        # 알림 유형에 따른 이모지 설정
        emoji = "⚠️"
        if alert_type == "daily_loss_limit":
            emoji = "🔴"
        elif alert_type == "weekly_loss_limit":
            emoji = "🔴"
        elif alert_type == "consecutive_losses":
            emoji = "🟡"
        elif alert_type == "abnormal_volatility":
            emoji = "🟠"
        elif alert_type == "trend_reversal":
            emoji = "🔄"
        
        # 알림 유형에 따른 제목 설정
        title = "위험 알림"
        if alert_type == "daily_loss_limit":
            title = "일일 손실 한도 초과"
        elif alert_type == "weekly_loss_limit":
            title = "주간 손실 한도 초과"
        elif alert_type == "consecutive_losses":
            title = "연속 손실 발생"
        elif alert_type == "abnormal_volatility":
            title = "비정상 변동성 감지"
        elif alert_type == "trend_reversal":
            title = "추세 반전 감지"
        
        # 메시지 포맷팅
        message = f"{emoji} <b>{title}</b>\n\n"
        
        # 상세 정보 추가
        for key, value in details.items():
            if isinstance(value, float):
                message += f"{key}: <code>{value:.2f}</code>\n"
            else:
                message += f"{key}: <code>{value}</code>\n"
        
        message += f"\n시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return message
    
    async def _send_chart_image(self, chart_data: Dict[str, Any]) -> bool:
        """
        차트 이미지 전송
        
        Args:
            chart_data (Dict[str, Any]): 차트 데이터
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            # 차트 이미지 생성 로직 구현
            # 실제 구현에서는 matplotlib 등을 사용하여 차트 이미지 생성
            
            # 임시로 차트 이미지 전송 로직 생략
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending chart image: {str(e)}")
            return False
    
    async def close(self):
        """
        리소스 정리
        """
        try:
            if self.app:
                await self.app.shutdown()
                self.logger.info("Telegram bot shut down")
        except Exception as e:
            self.logger.error(f"Error shutting down Telegram bot: {str(e)}")
            raise 