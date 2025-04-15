import numpy as np
import pandas as pd
import logging
import asyncio
from datetime import datetime, timedelta
import os
import sys
import queue
import threading
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 경로 추가하여 모듈 import 가능하게 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.ml.bayesian.monitoring_dashboard import MonitoringDashboard

class MockDataGenerator:
    """모의 데이터 생성기"""
    
    def __init__(self, data_queue: queue.Queue):
        self.data_queue = data_queue
        self.running = False
        self.current_price = 50000.0
        self.current_position = 0.0
        self.portfolio_value = 10000.0
    
    def start(self):
        """데이터 생성 시작"""
        self.running = True
        self.thread = threading.Thread(target=self._generate_data)
        self.thread.start()
    
    def stop(self):
        """데이터 생성 중지"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
    
    def _generate_data(self):
        """모의 데이터 생성"""
        while self.running:
            try:
                # 가격 데이터 생성
                price_change = np.random.normal(0, 100)
                self.current_price += price_change
                self.data_queue.put({'price': self.current_price})
                
                # 포지션 데이터 생성
                if np.random.random() < 0.1:  # 10% 확률로 포지션 변경
                    self.current_position = np.random.uniform(-0.1, 0.1)
                self.data_queue.put({'position': self.current_position})
                
                # 포트폴리오 가치 업데이트
                self.portfolio_value *= (1 + self.current_position * price_change / self.current_price)
                self.data_queue.put({'portfolio_value': self.portfolio_value})
                
                # 거래 데이터 생성
                if np.random.random() < 0.05:  # 5% 확률로 거래 발생
                    trade = {
                        'side': 'buy' if np.random.random() < 0.5 else 'sell',
                        'size': abs(self.current_position),
                        'price': self.current_price
                    }
                    self.data_queue.put({'trade': trade})
                
                # 성과 지표 업데이트
                metrics = {
                    'sharpe_ratio': np.random.normal(1.5, 0.2),
                    'max_drawdown': np.random.uniform(0.05, 0.15),
                    'win_rate': np.random.uniform(0.5, 0.7)
                }
                self.data_queue.put({'metrics': metrics})
                
                # 알림 생성
                if np.random.random() < 0.02:  # 2% 확률로 알림 생성
                    alert = {
                        'type': 'warning' if np.random.random() < 0.5 else 'info',
                        'message': '시스템 상태 정상' if np.random.random() < 0.5 else '리스크 한도 초과'
                    }
                    self.data_queue.put({'alert': alert})
                
                # 업데이트 간격 대기
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"데이터 생성 중 오류 발생: {e}")

def main():
    """대시보드 실행 예제"""
    try:
        # 데이터 큐 생성
        data_queue = queue.Queue()
        
        # 모의 데이터 생성기 초기화
        data_generator = MockDataGenerator(data_queue)
        
        # 대시보드 초기화
        dashboard = MonitoringDashboard(
            data_queue=data_queue,
            update_interval=1,
            save_dir="./dashboard_data"
        )
        
        # 데이터 생성 시작
        data_generator.start()
        
        # 대시보드 시작
        dashboard.start()
        
    except Exception as e:
        logger.error(f"예제 실행 중 오류 발생: {e}")
    finally:
        # 정리
        if 'data_generator' in locals():
            data_generator.stop()
        if 'dashboard' in locals():
            dashboard.stop()

if __name__ == "__main__":
    main() 