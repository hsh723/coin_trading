import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from typing import Dict, Any, Optional, List
from ..utils.logger import setup_logger

class Notifier:
    """알림 시스템 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        알림 시스템 초기화
        
        Args:
            config (Dict[str, Any]): 알림 설정
                - telegram_token: Telegram 봇 토큰
                - telegram_chat_id: Telegram 채팅 ID
                - email_sender: 이메일 발신자 주소
                - email_password: 이메일 비밀번호
                - email_receiver: 이메일 수신자 주소
        """
        self.logger = setup_logger('notifier')
        self.config = config
        
        # Telegram 설정
        self.telegram_token = config.get('telegram_token')
        self.telegram_chat_id = config.get('telegram_chat_id')
        
        # 이메일 설정
        self.email_sender = config.get('email_sender')
        self.email_password = config.get('email_password')
        self.email_receiver = config.get('email_receiver')
        
    def send_telegram_message(self, message: str) -> bool:
        """
        Telegram 메시지 전송
        
        Args:
            message (str): 전송할 메시지
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            if not self.telegram_token or not self.telegram_chat_id:
                self.logger.warning("Telegram 설정이 없습니다.")
                return False
                
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, data=data)
            response.raise_for_status()
            
            self.logger.info("Telegram 메시지 전송 성공")
            return True
            
        except Exception as e:
            self.logger.error(f"Telegram 메시지 전송 실패: {str(e)}")
            return False
            
    def send_email(self, subject: str, message: str) -> bool:
        """
        이메일 전송
        
        Args:
            subject (str): 이메일 제목
            message (str): 이메일 내용
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            if not all([self.email_sender, self.email_password, self.email_receiver]):
                self.logger.warning("이메일 설정이 없습니다.")
                return False
                
            # 이메일 메시지 생성
            msg = MIMEMultipart()
            msg['From'] = self.email_sender
            msg['To'] = self.email_receiver
            msg['Subject'] = subject
            
            # HTML 형식의 메시지 추가
            msg.attach(MIMEText(message, 'html'))
            
            # SMTP 서버 연결 및 이메일 전송
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(self.email_sender, self.email_password)
                server.send_message(msg)
                
            self.logger.info("이메일 전송 성공")
            return True
            
        except Exception as e:
            self.logger.error(f"이메일 전송 실패: {str(e)}")
            return False
            
    def notify_trade_signal(self, signal: Dict[str, Any]) -> bool:
        """
        거래 신호 알림 전송
        
        Args:
            signal (Dict[str, Any]): 거래 신호 정보
                - symbol: 거래 심볼
                - signal: 신호 타입 (buy/sell)
                - price: 가격
                - strength: 신호 강도
                - reason: 신호 발생 이유
            
        Returns:
            bool: 알림 전송 성공 여부
        """
        try:
            # 메시지 생성
            message = f"""
            <b>거래 신호 발생</b>
            
            심볼: {signal['symbol']}
            신호: {signal['signal'].upper()}
            가격: {signal['price']}
            강도: {signal['strength']}
            이유: {signal['reason']}
            """
            
            # Telegram 메시지 전송
            telegram_success = self.send_telegram_message(message)
            
            # 이메일 전송
            email_success = self.send_email(
                f"거래 신호: {signal['symbol']} {signal['signal'].upper()}",
                message
            )
            
            return telegram_success or email_success
            
        except Exception as e:
            self.logger.error(f"거래 신호 알림 전송 실패: {str(e)}")
            return False
            
    def notify_trade_execution(self, trade: Dict[str, Any]) -> bool:
        """
        거래 실행 알림 전송
        
        Args:
            trade (Dict[str, Any]): 거래 정보
                - symbol: 거래 심볼
                - side: 거래 방향 (buy/sell)
                - price: 실행 가격
                - amount: 거래 수량
                - order_id: 주문 ID
                - status: 주문 상태
            
        Returns:
            bool: 알림 전송 성공 여부
        """
        try:
            # 메시지 생성
            message = f"""
            <b>거래 실행 완료</b>
            
            심볼: {trade['symbol']}
            방향: {trade['side'].upper()}
            가격: {trade['price']}
            수량: {trade['amount']}
            주문 ID: {trade['order_id']}
            상태: {trade['status']}
            """
            
            # Telegram 메시지 전송
            telegram_success = self.send_telegram_message(message)
            
            # 이메일 전송
            email_success = self.send_email(
                f"거래 실행: {trade['symbol']} {trade['side'].upper()}",
                message
            )
            
            return telegram_success or email_success
            
        except Exception as e:
            self.logger.error(f"거래 실행 알림 전송 실패: {str(e)}")
            return False
            
    def notify_position_update(self, position: Dict[str, Any]) -> bool:
        """
        포지션 업데이트 알림 전송
        
        Args:
            position (Dict[str, Any]): 포지션 정보
                - symbol: 거래 심볼
                - entry_price: 진입 가격
                - current_price: 현재 가격
                - amount: 포지션 수량
                - unrealized_pnl: 미실현 손익
                - stop_loss: 손절 가격
                - take_profit: 이익 실현 가격
            
        Returns:
            bool: 알림 전송 성공 여부
        """
        try:
            # 메시지 생성
            message = f"""
            <b>포지션 업데이트</b>
            
            심볼: {position['symbol']}
            진입 가격: {position['entry_price']}
            현재 가격: {position['current_price']}
            수량: {position['amount']}
            미실현 손익: {position['unrealized_pnl']}
            손절 가격: {position['stop_loss']}
            이익 실현 가격: {position['take_profit']}
            """
            
            # Telegram 메시지 전송
            telegram_success = self.send_telegram_message(message)
            
            # 이메일 전송
            email_success = self.send_email(
                f"포지션 업데이트: {position['symbol']}",
                message
            )
            
            return telegram_success or email_success
            
        except Exception as e:
            self.logger.error(f"포지션 업데이트 알림 전송 실패: {str(e)}")
            return False
            
    def notify_error(self, error: Exception, context: Optional[str] = None) -> bool:
        """
        에러 알림 전송
        
        Args:
            error (Exception): 발생한 에러
            context (Optional[str]): 에러 발생 컨텍스트
            
        Returns:
            bool: 알림 전송 성공 여부
        """
        try:
            # 메시지 생성
            message = f"""
            <b>에러 발생</b>
            
            에러: {str(error)}
            컨텍스트: {context if context else 'N/A'}
            """
            
            # Telegram 메시지 전송
            telegram_success = self.send_telegram_message(message)
            
            # 이메일 전송
            email_success = self.send_email(
                "트레이딩 봇 에러 발생",
                message
            )
            
            return telegram_success or email_success
            
        except Exception as e:
            self.logger.error(f"에러 알림 전송 실패: {str(e)}")
            return False
            
    def notify_performance_report(self, report: Dict[str, Any]) -> bool:
        """
        성과 리포트 알림 전송
        
        Args:
            report (Dict[str, Any]): 성과 리포트 정보
                - period: 리포트 기간
                - total_return: 총 수익률
                - win_rate: 승률
                - total_trades: 총 거래 횟수
                - max_drawdown: 최대 낙폭
                - sharpe_ratio: 샤프 비율
            
        Returns:
            bool: 알림 전송 성공 여부
        """
        try:
            # 메시지 생성
            message = f"""
            <b>성과 리포트</b>
            
            기간: {report['period']}
            총 수익률: {report['total_return']:.2%}
            승률: {report['win_rate']:.2%}
            총 거래 횟수: {report['total_trades']}
            최대 낙폭: {report['max_drawdown']:.2%}
            샤프 비율: {report['sharpe_ratio']:.2f}
            """
            
            # Telegram 메시지 전송
            telegram_success = self.send_telegram_message(message)
            
            # 이메일 전송
            email_success = self.send_email(
                f"성과 리포트: {report['period']}",
                message
            )
            
            return telegram_success or email_success
            
        except Exception as e:
            self.logger.error(f"성과 리포트 알림 전송 실패: {str(e)}")
            return False 