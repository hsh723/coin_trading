"""
스케일링 관리 모듈
"""

import logging
import threading
import queue
import time
from typing import Dict, List, Optional
from datetime import datetime
import psutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class ScalingManager:
    """스케일링 관리 클래스"""
    
    def __init__(self, db_manager, max_workers: int = 10):
        """
        초기화
        
        Args:
            db_manager: 데이터베이스 관리자
            max_workers (int): 최대 작업자 수
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers
        self.worker_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = queue.Queue()
        self.active_tasks = {}
        self.resource_limits = {
            'cpu_percent': 80,
            'memory_percent': 80,
            'disk_percent': 80
        }
        
    def add_trading_task(self,
                        symbol: str,
                        strategy: str,
                        params: Dict) -> bool:
        """
        트레이딩 작업 추가
        
        Args:
            symbol (str): 코인 심볼
            strategy (str): 전략 이름
            params (Dict): 전략 파라미터
            
        Returns:
            bool: 작업 추가 성공 여부
        """
        try:
            # 리소스 사용량 확인
            if not self._check_resource_availability():
                self.logger.warning("리소스 한계 도달로 작업 추가 실패")
                return False
                
            # 작업 큐에 추가
            task_id = f"{symbol}_{strategy}_{datetime.now().timestamp()}"
            task = {
                'id': task_id,
                'symbol': symbol,
                'strategy': strategy,
                'params': params,
                'status': 'pending',
                'start_time': None,
                'end_time': None
            }
            
            self.task_queue.put(task)
            self.active_tasks[task_id] = task
            
            # 작업자 풀에서 작업 실행
            self.worker_pool.submit(self._execute_trading_task, task)
            
            return True
            
        except Exception as e:
            self.logger.error(f"트레이딩 작업 추가 실패: {str(e)}")
            return False
            
    def _execute_trading_task(self, task: Dict):
        """
        트레이딩 작업 실행
        
        Args:
            task (Dict): 작업 정보
        """
        try:
            # 작업 상태 업데이트
            task['status'] = 'running'
            task['start_time'] = datetime.now()
            
            # 전략 실행
            strategy = self._get_strategy_instance(
                task['strategy'],
                task['params']
            )
            
            result = strategy.execute(task['symbol'])
            
            # 작업 완료 처리
            task['status'] = 'completed'
            task['end_time'] = datetime.now()
            task['result'] = result
            
            # 결과 저장
            self._save_task_result(task)
            
        except Exception as e:
            self.logger.error(f"트레이딩 작업 실행 실패: {str(e)}")
            task['status'] = 'failed'
            task['error'] = str(e)
            
        finally:
            # 작업 완료 후 리소스 정리
            self._cleanup_task(task)
            
    def _get_strategy_instance(self,
                             strategy_name: str,
                             params: Dict):
        """
        전략 인스턴스 생성
        
        Args:
            strategy_name (str): 전략 이름
            params (Dict): 전략 파라미터
            
        Returns:
            전략 인스턴스
        """
        try:
            # 전략 클래스 동적 임포트
            module = __import__(
                f"src.strategy.{strategy_name}",
                fromlist=[strategy_name]
            )
            strategy_class = getattr(module, strategy_name)
            
            return strategy_class(self.db_manager, **params)
            
        except Exception as e:
            self.logger.error(f"전략 인스턴스 생성 실패: {str(e)}")
            raise
            
    def _save_task_result(self, task: Dict):
        """
        작업 결과 저장
        
        Args:
            task (Dict): 작업 정보
        """
        try:
            self.db_manager.save_trading_result({
                'task_id': task['id'],
                'symbol': task['symbol'],
                'strategy': task['strategy'],
                'start_time': task['start_time'],
                'end_time': task['end_time'],
                'status': task['status'],
                'result': task.get('result'),
                'error': task.get('error')
            })
            
        except Exception as e:
            self.logger.error(f"작업 결과 저장 실패: {str(e)}")
            
    def _cleanup_task(self, task: Dict):
        """
        작업 정리
        
        Args:
            task (Dict): 작업 정보
        """
        try:
            # 활성 작업에서 제거
            if task['id'] in self.active_tasks:
                del self.active_tasks[task['id']]
                
            # 리소스 사용량 확인
            self._check_resource_usage()
            
        except Exception as e:
            self.logger.error(f"작업 정리 실패: {str(e)}")
            
    def _check_resource_availability(self) -> bool:
        """
        리소스 가용성 확인
        
        Returns:
            bool: 리소스 가용 여부
        """
        try:
            # CPU 사용량 확인
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > self.resource_limits['cpu_percent']:
                return False
                
            # 메모리 사용량 확인
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > self.resource_limits['memory_percent']:
                return False
                
            # 디스크 사용량 확인
            disk_percent = psutil.disk_usage('/').percent
            if disk_percent > self.resource_limits['disk_percent']:
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"리소스 가용성 확인 실패: {str(e)}")
            return False
            
    def _check_resource_usage(self):
        """리소스 사용량 확인 및 조정"""
        try:
            # 활성 작업 수 확인
            active_count = len(self.active_tasks)
            
            # 리소스 사용량에 따른 작업자 수 조정
            if active_count > self.max_workers:
                self.max_workers = min(
                    active_count + 2,
                    psutil.cpu_count() * 2
                )
                self.worker_pool._max_workers = self.max_workers
                
        except Exception as e:
            self.logger.error(f"리소스 사용량 확인 실패: {str(e)}")
            
    def get_task_status(self, task_id: str) -> Dict:
        """
        작업 상태 조회
        
        Args:
            task_id (str): 작업 ID
            
        Returns:
            Dict: 작업 상태
        """
        try:
            return self.active_tasks.get(task_id, {})
            
        except Exception as e:
            self.logger.error(f"작업 상태 조회 실패: {str(e)}")
            return {}
            
    def get_all_tasks(self) -> List[Dict]:
        """
        모든 작업 조회
        
        Returns:
            List[Dict]: 작업 목록
        """
        try:
            return list(self.active_tasks.values())
            
        except Exception as e:
            self.logger.error(f"작업 목록 조회 실패: {str(e)}")
            return []
            
    def stop_task(self, task_id: str) -> bool:
        """
        작업 중지
        
        Args:
            task_id (str): 작업 ID
            
        Returns:
            bool: 중지 성공 여부
        """
        try:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task['status'] = 'stopped'
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"작업 중지 실패: {str(e)}")
            return False
            
    def shutdown(self):
        """스케일링 관리자 종료"""
        try:
            # 작업자 풀 종료
            self.worker_pool.shutdown(wait=True)
            
            # 활성 작업 정리
            for task in self.active_tasks.values():
                if task['status'] == 'running':
                    task['status'] = 'stopped'
                    
        except Exception as e:
            self.logger.error(f"스케일링 관리자 종료 실패: {str(e)}") 