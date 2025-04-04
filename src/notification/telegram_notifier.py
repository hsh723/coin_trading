"""
텔레그램 알림 모듈
"""

import aiohttp
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json

class TelegramNotifier:
    """텔레그램 알림 클래스"""
    
    def __init__(self, bot_token: str, chat_id: str):
        """
        초기화
        
        Args:
            bot_token (str): 텔레그램 봇 토큰
            chat_id (str): 채팅 ID
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.logger = logging.getLogger(__name__)
        
    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        메시지 전송
        
        Args:
            text (str): 전송할 메시지
            parse_mode (str): 메시지 형식
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        return True
                    else:
                        self.logger.error(f"메시지 전송 실패: {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"메시지 전송 중 오류 발생: {str(e)}")
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
            text = (
                f"🔔 <b>거래 신호</b>\n\n"
                f"심볼: {signal['symbol']}\n"
                f"방향: {signal['side']}\n"
                f"가격: {signal['price']:.2f}\n"
                f"강도: {signal['strength']}\n"
                f"시간: {signal['timestamp']}\n"
                f"이유: {signal['reason']}"
            )
            
            return await self.send_message(text)
            
        except Exception as e:
            self.logger.error(f"거래 신호 알림 전송 중 오류 발생: {str(e)}")
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
            text = (
                f"📊 <b>포지션 업데이트</b>\n\n"
                f"심볼: {position['symbol']}\n"
                f"방향: {position['side']}\n"
                f"크기: {position['size']:.4f}\n"
                f"진입가: {position['entry_price']:.2f}\n"
                f"현재가: {position['current_price']:.2f}\n"
                f"손익: {position.get('pnl', 0):.2f}"
            )
            
            return await self.send_message(text)
            
        except Exception as e:
            self.logger.error(f"포지션 업데이트 알림 전송 중 오류 발생: {str(e)}")
            return False
            
    async def send_error_alert(self, error: Exception, context: str = '') -> bool:
        """
        에러 알림 전송
        
        Args:
            error (Exception): 에러 객체
            context (str): 에러 컨텍스트
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            text = (
                f"⚠️ <b>에러 발생</b>\n\n"
                f"컨텍스트: {context}\n"
                f"에러: {str(error)}\n"
                f"시간: {datetime.now()}"
            )
            
            return await self.send_message(text)
            
        except Exception as e:
            self.logger.error(f"에러 알림 전송 중 오류 발생: {str(e)}")
            return False
            
    async def send_performance_report(self, report: Dict[str, Any]) -> bool:
        """
        성과 리포트 알림 전송
        
        Args:
            report (Dict[str, Any]): 성과 리포트 정보
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            text = (
                f"📈 <b>성과 리포트</b>\n\n"
                f"기간: {report['period']}\n"
                f"총 거래: {report['total_trades']}\n"
                f"승률: {report['win_rate']:.2%}\n"
                f"수익률: {report['returns']:.2%}\n"
                f"샤프 비율: {report['sharpe_ratio']:.2f}"
            )
            
            return await self.send_message(text)
            
        except Exception as e:
            self.logger.error(f"성과 리포트 알림 전송 중 오류 발생: {str(e)}")
            return False
            
    async def send_news_alert(self, news: Dict[str, Any]) -> bool:
        """
        뉴스 알림 전송
        
        Args:
            news (Dict[str, Any]): 뉴스 정보
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            text = (
                f"📰 <b>중요 뉴스</b>\n\n"
                f"제목: {news['title']}\n"
                f"소스: {news['source']}\n"
                f"감성: {news['sentiment_label']}\n"
                f"영향도: {news['impact_label']}\n"
                f"시간: {news['timestamp']}"
            )
            
            return await self.send_message(text)
            
        except Exception as e:
            self.logger.error(f"뉴스 알림 전송 중 오류 발생: {str(e)}")
            return False 