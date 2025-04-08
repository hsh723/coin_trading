"""
시스템 모니터링 및 알림 모듈
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import requests
from ..utils.config import load_config

class SystemMonitor:
    """시스템 모니터링 및 알림 클래스"""
    
    def __init__(self):
        """초기화"""
        self.config = load_config()
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
    def _setup_logging(self):
        """로깅 설정"""
        self.logger.setLevel(logging.INFO)
        
    def check_system_health(self) -> Dict[str, Any]:
        """
        시스템 상태 확인
        
        Returns:
            Dict[str, Any]: 시스템 상태 정보
        """
        health = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'components': {}
        }
        
        # 데이터베이스 연결 확인
        try:
            from ..database.manager import DatabaseManager
            db = DatabaseManager()
            db.test_connection()
            health['components']['database'] = 'healthy'
        except Exception as e:
            health['components']['database'] = f'error: {str(e)}'
            health['status'] = 'unhealthy'
            
        # API 연결 확인
        try:
            from ..exchange.binance_api import BinanceAPI
            api = BinanceAPI()
            api.test_connection()
            health['components']['api'] = 'healthy'
        except Exception as e:
            health['components']['api'] = f'error: {str(e)}'
            health['status'] = 'unhealthy'
            
        return health
        
    def send_notification(self, 
                         message: str, 
                         level: str = "info",
                         channel: str = "telegram") -> bool:
        """
        알림 전송
        
        Args:
            message (str): 알림 메시지
            level (str): 알림 레벨 (info, warning, error)
            channel (str): 알림 채널 (telegram, email)
            
        Returns:
            bool: 알림 전송 성공 여부
        """
        try:
            if channel == "telegram":
                return self._send_telegram(message, level)
            elif channel == "email":
                return self._send_email(message, level)
            else:
                self.logger.error(f"지원하지 않는 알림 채널: {channel}")
                return False
        except Exception as e:
            self.logger.error(f"알림 전송 실패: {str(e)}")
            return False
            
    def _send_telegram(self, message: str, level: str) -> bool:
        """
        텔레그램 알림 전송
        
        Args:
            message (str): 알림 메시지
            level (str): 알림 레벨
            
        Returns:
            bool: 알림 전송 성공 여부
        """
        try:
            bot_token = self.config.get('telegram', {}).get('bot_token')
            chat_id = self.config.get('telegram', {}).get('chat_id')
            
            if not bot_token or not chat_id:
                self.logger.error("텔레그램 설정이 없습니다.")
                return False
                
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": f"[{level.upper()}] {message}",
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, json=data)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"텔레그램 알림 전송 실패: {str(e)}")
            return False
            
    def _send_email(self, message: str, level: str) -> bool:
        """
        이메일 알림 전송
        
        Args:
            message (str): 알림 메시지
            level (str): 알림 레벨
            
        Returns:
            bool: 알림 전송 성공 여부
        """
        try:
            # 이메일 전송 로직 구현
            return True
        except Exception as e:
            self.logger.error(f"이메일 알림 전송 실패: {str(e)}")
            return False 