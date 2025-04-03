"""
장기 실행 안정성 테스트
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
import psutil
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from src.utils.logger import setup_logger
from src.utils.optimization import memory_manager, performance_monitor
from src.utils.recovery import system_monitor
from src.trading.live_simulator import LiveSimulator
from src.trading.strategy import IntegratedStrategy
from src.utils.config_loader import get_config

# 로거 설정
logger = setup_logger('stability_test')

class StabilityTest:
    def __init__(self, duration_days: int = 7):
        """
        안정성 테스트 초기화
        
        Args:
            duration_days (int): 테스트 기간 (일)
        """
        self.duration_days = duration_days
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(days=duration_days)
        self.config = get_config()
        self.metrics: Dict[str, List[Any]] = {
            'timestamp': [],
            'memory_usage': [],
            'cpu_usage': [],
            'error_count': [],
            'operation_times': []
        }
    
    async def run(self):
        """안정성 테스트 실행"""
        try:
            # 시뮬레이터 초기화
            simulator = LiveSimulator(
                exchange_name=self.config['exchange']['name'],
                initial_capital=self.config['trading']['initial_capital'],
                speed=10.0  # 가속 테스트
            )
            await simulator.initialize()
            
            # 전략 초기화
            strategy = IntegratedStrategy()
            
            logger.info(f"안정성 테스트 시작 (기간: {self.duration_days}일)")
            
            while datetime.now() < self.end_time:
                try:
                    # 시스템 메트릭 수집
                    self._collect_metrics()
                    
                    # 시뮬레이션 실행
                    await simulator.run_simulation(strategy)
                    
                    # 메모리 최적화
                    memory_manager.optimize_memory()
                    
                    # 1시간 대기
                    await asyncio.sleep(3600)
                    
                except Exception as e:
                    logger.error(f"테스트 실행 중 오류 발생: {str(e)}")
                    system_monitor.record_error()
                    await asyncio.sleep(60)  # 오류 발생 시 1분 대기
            
            # 테스트 결과 분석
            self._analyze_results()
            
        except Exception as e:
            logger.error(f"안정성 테스트 실패: {str(e)}")
            raise
        finally:
            if 'simulator' in locals():
                await simulator.close()
    
    def _collect_metrics(self):
        """시스템 메트릭 수집"""
        current_time = datetime.now()
        process = psutil.Process()
        
        self.metrics['timestamp'].append(current_time)
        self.metrics['memory_usage'].append(process.memory_percent())
        self.metrics['cpu_usage'].append(process.cpu_percent())
        self.metrics['error_count'].append(system_monitor.error_count)
        
        # 작업 시간 통계 수집
        operation_stats = {}
        for operation in performance_monitor.operation_times:
            stats = performance_monitor.get_operation_stats(operation)
            operation_stats[operation] = stats
        
        self.metrics['operation_times'].append(operation_stats)
    
    def _analyze_results(self):
        """테스트 결과 분석"""
        # 메트릭을 데이터프레임으로 변환
        df = pd.DataFrame(self.metrics)
        
        # 메모리 사용량 분석
        memory_stats = {
            'avg_usage': df['memory_usage'].mean(),
            'max_usage': df['memory_usage'].max(),
            'min_usage': df['memory_usage'].min(),
            'std_usage': df['memory_usage'].std()
        }
        
        # CPU 사용량 분석
        cpu_stats = {
            'avg_usage': df['cpu_usage'].mean(),
            'max_usage': df['cpu_usage'].max(),
            'min_usage': df['cpu_usage'].min(),
            'std_usage': df['cpu_usage'].std()
        }
        
        # 오류 분석
        error_stats = {
            'total_errors': df['error_count'].iloc[-1],
            'error_rate': df['error_count'].iloc[-1] / self.duration_days
        }
        
        # 작업 시간 분석
        operation_stats = {}
        for operation in performance_monitor.operation_times:
            stats = performance_monitor.get_operation_stats(operation)
            operation_stats[operation] = stats
        
        # 결과 출력
        logger.info("=== 안정성 테스트 결과 ===")
        logger.info(f"테스트 기간: {self.duration_days}일")
        logger.info(f"시작 시간: {self.start_time}")
        logger.info(f"종료 시간: {datetime.now()}")
        logger.info("\n메모리 사용량:")
        for key, value in memory_stats.items():
            logger.info(f"{key}: {value:.2f}%")
        logger.info("\nCPU 사용량:")
        for key, value in cpu_stats.items():
            logger.info(f"{key}: {value:.2f}%")
        logger.info("\n오류 통계:")
        for key, value in error_stats.items():
            logger.info(f"{key}: {value}")
        logger.info("\n작업 시간 통계:")
        for operation, stats in operation_stats.items():
            logger.info(f"\n{operation}:")
            for key, value in stats.items():
                logger.info(f"{key}: {value:.4f}초")
        
        # 결과 저장
        results = {
            'memory_stats': memory_stats,
            'cpu_stats': cpu_stats,
            'error_stats': error_stats,
            'operation_stats': operation_stats
        }
        
        # 결과를 JSON 파일로 저장
        import json
        with open('stability_test_results.json', 'w') as f:
            json.dump(results, f, indent=4)

async def main():
    """메인 함수"""
    try:
        test = StabilityTest(duration_days=7)
        await test.run()
    except KeyboardInterrupt:
        logger.info("테스트 중단")
        sys.exit(0)
    except Exception as e:
        logger.error(f"테스트 실패: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 