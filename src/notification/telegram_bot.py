"""
텔레그램 알림 모듈
"""

import asyncio
import aiohttp
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
CHOOSING_ACTION, VIEW_POSITIONS, TOGGLE_STRATEGY = range(3)

class TelegramNotifier:
    """
    텔레그램 알림 클래스
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        bot_token: str,
        chat_id: str,
        parse_mode: str = 'HTML'
    ):
        """
        텔레그램 알림기 초기화
        
        Args:
            config (Dict[str, Any]): 설정 정보
            bot_token (str): 텔레그램 봇 토큰
            chat_id (str): 텔레그램 채팅 ID
            parse_mode (str): 메시지 파싱 모드
        """
        self.config = config
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.parse_mode = parse_mode
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.session = None
        self.logger = setup_logger()
        
        # 알림 설정
        self.notifications = config.get('telegram', {}).get('notifications', {})
        
        # 컴포넌트 초기화
        self.executor = OrderExecutor()
        self.application = Application.builder().token(bot_token).build()
        
        # 상태 변수
        self.is_strategy_active = True
    
    async def initialize(self):
        """알림기 초기화"""
        try:
            self.session = aiohttp.ClientSession()
            self.logger.info("텔레그램 알림기 초기화 완료")
            
            # 명령 핸들러 등록
            self.application.add_handler(CommandHandler("start", self._start_command))
            self.application.add_handler(CommandHandler("help", self._help_command))
            self.application.add_handler(CommandHandler("positions", self._positions_command))
            self.application.add_handler(CommandHandler("status", self._status_command))
            
            # 대화 핸들러 등록
            conv_handler = ConversationHandler(
                entry_points=[CommandHandler("menu", self._menu_command)],
                states={
                    CHOOSING_ACTION: [
                        CallbackQueryHandler(self._view_positions, pattern="^view_positions$"),
                        CallbackQueryHandler(self._toggle_strategy, pattern="^toggle_strategy$"),
                        CallbackQueryHandler(self._back_to_menu, pattern="^back$")
                    ],
                    VIEW_POSITIONS: [
                        CallbackQueryHandler(self._back_to_menu, pattern="^back$")
                    ],
                    TOGGLE_STRATEGY: [
                        CallbackQueryHandler(self._back_to_menu, pattern="^back$")
                    ]
                },
                fallbacks=[CommandHandler("cancel", self._cancel_command)]
            )
            self.application.add_handler(conv_handler)
            
            # 봇 시작
            await self.application.initialize()
            await self.application.start()
            await self.application.run_polling()
            
        except Exception as e:
            self.logger.error(f"Error starting Telegram bot: {str(e)}")
            raise
    
    async def close(self):
        """알림기 종료"""
        if self.session:
            await self.session.close()
    
    async def send_message(
        self,
        message: str,
        disable_notification: bool = False,
        reply_markup: Optional[Dict] = None
    ) -> bool:
        """
        메시지 전송
        
        Args:
            message (str): 전송할 메시지
            disable_notification (bool): 알림 비활성화 여부
            reply_markup (Optional[Dict]): 인라인 키보드 마크업
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            if not self.session:
                await self.initialize()
                
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': self.parse_mode,
                'disable_notification': disable_notification
            }
            
            if reply_markup:
                data['reply_markup'] = reply_markup
                
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    self.logger.debug(f"메시지 전송 성공: {message[:50]}...")
                    return True
                else:
                    error_text = await response.text()
                    self.logger.error(f"메시지 전송 실패: {error_text}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"메시지 전송 중 오류 발생: {str(e)}")
            return False
    
    async def notify_trade_signal(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        order_type: str
    ) -> None:
        """
        거래 신호 알림
        
        Args:
            symbol (str): 거래 심볼
            side (str): 포지션 방향
            amount (float): 주문 수량
            price (float): 주문 가격
            order_type (str): 주문 유형
        """
        try:
            emoji = "🟢" if side == "long" else "🔴"
            message = (
                f"{emoji} *{order_type.upper()} {side.upper()}*\n"
                f"Symbol: `{symbol}`\n"
                f"Amount: `{amount:.4f}`\n"
                f"Price: `{price:.2f}`\n"
                f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
            )
            
            await self.send_message(message, parse_mode="Markdown")
            
        except Exception as e:
            self.logger.error(f"Error sending trade signal: {str(e)}")
    
    async def notify_error(self, error: str, context: Optional[str] = None) -> None:
        """
        오류 알림
        
        Args:
            error (str): 오류 메시지
            context (str, optional): 오류 발생 컨텍스트
        """
        try:
            message = (
                f"⚠️ *ERROR ALERT*\n"
                f"Context: `{context or 'N/A'}`\n"
                f"Error: `{error}`\n"
                f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
            )
            
            await self.send_message(message, parse_mode="Markdown")
            
        except Exception as e:
            self.logger.error(f"Error sending error notification: {str(e)}")
    
    async def send_performance_report(
        self,
        daily_pnl: float,
        weekly_pnl: float,
        total_trades: int,
        win_rate: float,
        max_drawdown: float
    ) -> None:
        """
        성과 보고서 전송
        
        Args:
            daily_pnl (float): 일일 손익
            weekly_pnl (float): 주간 손익
            total_trades (int): 총 거래 횟수
            win_rate (float): 승률
            max_drawdown (float): 최대 낙폭
        """
        try:
            message = (
                f"📊 *Performance Report*\n\n"
                f"*Daily PnL:* `{daily_pnl:.2f}%`\n"
                f"*Weekly PnL:* `{weekly_pnl:.2f}%`\n"
                f"*Total Trades:* `{total_trades}`\n"
                f"*Win Rate:* `{win_rate:.2f}%`\n"
                f"*Max Drawdown:* `{max_drawdown:.2f}%`\n\n"
                f"Generated at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
            )
            
            await self.send_message(message, parse_mode="Markdown")
            
        except Exception as e:
            self.logger.error(f"Error sending performance report: {str(e)}")
    
    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        /start 명령어 처리
        """
        if str(update.effective_user.id) not in self.config['telegram']['allowed_users']:
            await update.message.reply_text("Unauthorized access.")
            return
        
        await update.message.reply_text(
            "Welcome to the Trading Bot! Use /help to see available commands."
        )
    
    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        /help 명령어 처리
        """
        if str(update.effective_user.id) not in self.config['telegram']['allowed_users']:
            await update.message.reply_text("Unauthorized access.")
            return
        
        help_text = (
            "Available commands:\n\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/menu - Show main menu\n"
            "/positions - View current positions\n"
            "/status - Check strategy status\n"
            "/cancel - Cancel current operation"
        )
        
        await update.message.reply_text(help_text)
    
    async def _menu_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """
        /menu 명령어 처리
        """
        if str(update.effective_user.id) not in self.config['telegram']['allowed_users']:
            await update.message.reply_text("Unauthorized access.")
            return ConversationHandler.END
        
        keyboard = [
            [
                InlineKeyboardButton("View Positions", callback_data="view_positions"),
                InlineKeyboardButton("Toggle Strategy", callback_data="toggle_strategy")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "Main Menu:",
            reply_markup=reply_markup
        )
        
        return CHOOSING_ACTION
    
    async def _view_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """
        포지션 조회 처리
        """
        query = update.callback_query
        await query.answer()
        
        positions = self.executor.get_active_positions()
        
        if not positions:
            message = "No active positions."
        else:
            message = "Current Positions:\n\n"
            for pos in positions:
                message += (
                    f"Symbol: {pos['symbol']}\n"
                    f"Side: {pos['side']}\n"
                    f"Size: {pos['size']}\n"
                    f"Entry Price: {pos['entry_price']}\n"
                    f"Unrealized PnL: {pos['unrealized_pnl']:.2f}\n\n"
                )
        
        keyboard = [[InlineKeyboardButton("Back to Menu", callback_data="back")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            text=message,
            reply_markup=reply_markup
        )
        
        return VIEW_POSITIONS
    
    async def _toggle_strategy(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """
        전략 활성화/비활성화 처리
        """
        query = update.callback_query
        await query.answer()
        
        self.is_strategy_active = not self.is_strategy_active
        status = "active" if self.is_strategy_active else "inactive"
        
        message = f"Strategy is now {status}."
        
        keyboard = [[InlineKeyboardButton("Back to Menu", callback_data="back")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            text=message,
            reply_markup=reply_markup
        )
        
        return TOGGLE_STRATEGY
    
    async def _back_to_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """
        메뉴로 돌아가기 처리
        """
        query = update.callback_query
        await query.answer()
        
        return await self._menu_command(update, context)
    
    async def _cancel_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """
        /cancel 명령어 처리
        """
        if str(update.effective_user.id) not in self.config['telegram']['allowed_users']:
            await update.message.reply_text("Unauthorized access.")
            return ConversationHandler.END
        
        await update.message.reply_text("Operation cancelled.")
        return ConversationHandler.END
    
    async def _positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        /positions 명령어 처리
        """
        if str(update.effective_user.id) not in self.config['telegram']['allowed_users']:
            await update.message.reply_text("Unauthorized access.")
            return
        
        positions = self.executor.get_active_positions()
        
        if not positions:
            await update.message.reply_text("No active positions.")
            return
        
        message = "Current Positions:\n\n"
        for pos in positions:
            message += (
                f"Symbol: {pos['symbol']}\n"
                f"Side: {pos['side']}\n"
                f"Size: {pos['size']}\n"
                f"Entry Price: {pos['entry_price']}\n"
                f"Unrealized PnL: {pos['unrealized_pnl']:.2f}\n\n"
            )
        
        await update.message.reply_text(message)
    
    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        /status 명령어 처리
        """
        if str(update.effective_user.id) not in self.config['telegram']['allowed_users']:
            await update.message.reply_text("Unauthorized access.")
            return
        
        status = "active" if self.is_strategy_active else "inactive"
        await update.message.reply_text(f"Strategy is currently {status}.") 