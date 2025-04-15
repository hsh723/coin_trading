import os
from dotenv import load_dotenv
from typing import Dict, Any
import logging

class EnvLoader:
    """환경 설정 로더 클래스"""
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        
    def _parse_value(self, value: str) -> Any:
        """환경 변수 값을 파싱"""
        if value is None:
            return None
            
        # 주석 제거
        value = value.split('#')[0].strip()
        
        # 빈 문자열 처리
        if not value:
            return None
            
        # 숫자 타입 변환 시도
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def load(self) -> None:
        """환경 변수 로드"""
        try:
            # .env 파일 로드
            load_dotenv()
            
            # 필수 환경 변수
            self.config.update({
                'EXCHANGE_API_KEY': self._parse_value(os.getenv('EXCHANGE_API_KEY', 'test_key')),
                'EXCHANGE_API_SECRET': self._parse_value(os.getenv('EXCHANGE_API_SECRET', 'test_secret')),
                'EXCHANGE_NAME': self._parse_value(os.getenv('EXCHANGE_NAME', 'binance')),
                'TRADING_MODE': self._parse_value(os.getenv('TRADING_MODE', 'testnet')),
                'LOG_LEVEL': self._parse_value(os.getenv('LOG_LEVEL', 'INFO')),
                'LOG_FILE': self._parse_value(os.getenv('LOG_FILE', 'trading.log')),
                
                # 뉴스 API 설정
                'CRYPTOPANIC_API_KEY': self._parse_value(os.getenv('CRYPTOPANIC_API_KEY', 'test_key')),
                'COINDESK_API_KEY': self._parse_value(os.getenv('COINDESK_API_KEY', 'test_key')),
                'NEWS_UPDATE_INTERVAL': self._parse_value(os.getenv('NEWS_UPDATE_INTERVAL', '300')),
                'NEWS_LANGUAGE': self._parse_value(os.getenv('NEWS_LANGUAGE', 'en')),
                'NEWS_SENTIMENT_THRESHOLD': self._parse_value(os.getenv('NEWS_SENTIMENT_THRESHOLD', '0.5')),
                
                # 텔레그램 설정
                'TELEGRAM_BOT_TOKEN': self._parse_value(os.getenv('TELEGRAM_BOT_TOKEN', 'test_token')),
                'TELEGRAM_CHAT_ID': self._parse_value(os.getenv('TELEGRAM_CHAT_ID', 'test_chat_id')),
                
                # 대시보드 설정
                'DASHBOARD_PASSWORD': self._parse_value(os.getenv('DASHBOARD_PASSWORD', 'test_password')),
                
                # 데이터베이스 설정
                'DB_HOST': self._parse_value(os.getenv('DB_HOST', 'localhost')),
                'DB_PORT': self._parse_value(os.getenv('DB_PORT', '5432')),
                'DB_NAME': self._parse_value(os.getenv('DB_NAME', 'trading_bot')),
                'DB_USER': self._parse_value(os.getenv('DB_USER', 'test_user')),
                'DB_PASSWORD': self._parse_value(os.getenv('DB_PASSWORD', 'test_password')),
                
                # AWS S3 설정
                'AWS_ACCESS_KEY_ID': self._parse_value(os.getenv('AWS_ACCESS_KEY_ID', 'test_key')),
                'AWS_SECRET_ACCESS_KEY': self._parse_value(os.getenv('AWS_SECRET_ACCESS_KEY', 'test_secret')),
                'AWS_S3_BUCKET': self._parse_value(os.getenv('AWS_S3_BUCKET', 'test_bucket')),
                
                # Google Drive 설정
                'GDRIVE_FOLDER_ID': self._parse_value(os.getenv('GDRIVE_FOLDER_ID', 'test_folder')),
            })
            
            # 필수 환경 변수 검증
            required_vars = [
                'EXCHANGE_API_KEY',
                'EXCHANGE_API_SECRET',
                'CRYPTOPANIC_API_KEY',
                'COINDESK_API_KEY'
            ]
            
            missing_vars = [var for var in required_vars if not self.config.get(var)]
            if missing_vars:
                raise ValueError(f"필수 환경 변수가 설정되지 않았습니다: {', '.join(missing_vars)}")
            
            self.logger.info("환경 변수 로드 완료")
            
        except Exception as e:
            self.logger.error(f"환경 변수 로드 중 오류 발생: {str(e)}")
            raise
            
    def get(self, key: str, default: Any = None) -> Any:
        """환경 변수 값 조회"""
        return self.config.get(key, default)
        
    def get_all(self) -> Dict[str, Any]:
        """모든 환경 변수 조회"""
        return self.config.copy() 