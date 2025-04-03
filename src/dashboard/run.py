"""
대시보드 실행 파일

이 모듈은 트레이딩 시스템의 웹 대시보드를 실행합니다.
"""

import argparse
import logging
from src.dashboard.app import DashboardApp
from src.utils.logger import setup_logger

# 로거 설정
logger = logging.getLogger(__name__)

def main():
    """메인 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='트레이딩 대시보드')
    parser.add_argument('--host', default='0.0.0.0', help='호스트 주소')
    parser.add_argument('--port', type=int, default=5000, help='포트 번호')
    
    args = parser.parse_args()
    
    try:
        # 로거 설정
        setup_logger()
        
        # 대시보드 초기화 및 실행
        dashboard = DashboardApp()
        dashboard.run(host=args.host, port=args.port)
        
    except KeyboardInterrupt:
        logger.info("대시보드 종료")
    except Exception as e:
        logger.error(f"대시보드 실행 실패: {str(e)}")
        raise

if __name__ == "__main__":
    main() 