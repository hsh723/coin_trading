"""
실시간 처리 시스템 장애 복구 관리자
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import pickle
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SystemState:
    timestamp: datetime
    active_orders: Dict
    positions: Dict
    account_balance: float
    running_tasks: List[str]
    last_processed_event: Dict

class RecoveryManager:
    def __init__(self, config: Dict = None):
        """
        장애 복구 관리자 초기화
        
        Args:
            config (Dict): 설정 정보
        """
        self.config = config or {
            'checkpoint_interval': 60,  # 체크포인트 저장 주기 (초)
            'max_checkpoints': 5,  # 최대 체크포인트 수
            'state_dir': 'data/recovery',  # 상태 저장 디렉토리
            'recovery_timeout': 300  # 복구 타임아웃 (초)
        }
        
        self.state_dir = Path(self.config['state_dir'])
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_state: Optional[SystemState] = None
        self.last_checkpoint_time = datetime.now()
        
    async def start(self):
        """장애 복구 관리자 시작"""
        while True:
            try:
                await self._save_checkpoint()
                await asyncio.sleep(self.config['checkpoint_interval'])
            except Exception as e:
                logger.error(f"체크포인트 저장 중 오류 발생: {str(e)}")
                
    async def save_state(self, state: SystemState):
        """
        시스템 상태 저장
        
        Args:
            state (SystemState): 저장할 시스템 상태
        """
        try:
            self.current_state = state
            await self._save_checkpoint()
            logger.info("시스템 상태 저장 완료")
        except Exception as e:
            logger.error(f"상태 저장 중 오류 발생: {str(e)}")
            
    async def recover_state(self) -> Optional[SystemState]:
        """
        시스템 상태 복구
        
        Returns:
            Optional[SystemState]: 복구된 시스템 상태
        """
        try:
            # 최신 체크포인트 찾기
            checkpoints = sorted(self.state_dir.glob('checkpoint_*.pkl'))
            if not checkpoints:
                logger.warning("복구할 체크포인트가 없습니다")
                return None
                
            # 최신 체크포인트 로드
            latest_checkpoint = checkpoints[-1]
            with open(latest_checkpoint, 'rb') as f:
                state = pickle.load(f)
                
            self.current_state = state
            logger.info(f"시스템 상태 복구 완료: {latest_checkpoint}")
            return state
            
        except Exception as e:
            logger.error(f"상태 복구 중 오류 발생: {str(e)}")
            return None
            
    async def _save_checkpoint(self):
        """체크포인트 저장"""
        try:
            if not self.current_state:
                return
                
            # 체크포인트 파일명 생성
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_file = self.state_dir / f'checkpoint_{timestamp}.pkl'
            
            # 체크포인트 저장
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(self.current_state, f)
                
            # 오래된 체크포인트 정리
            await self._cleanup_old_checkpoints()
            
            self.last_checkpoint_time = datetime.now()
            logger.info(f"체크포인트 저장 완료: {checkpoint_file}")
            
        except Exception as e:
            logger.error(f"체크포인트 저장 중 오류 발생: {str(e)}")
            
    async def _cleanup_old_checkpoints(self):
        """오래된 체크포인트 정리"""
        try:
            checkpoints = sorted(self.state_dir.glob('checkpoint_*.pkl'))
            while len(checkpoints) > self.config['max_checkpoints']:
                oldest_checkpoint = checkpoints.pop(0)
                oldest_checkpoint.unlink()
                logger.info(f"오래된 체크포인트 삭제: {oldest_checkpoint}")
                
        except Exception as e:
            logger.error(f"체크포인트 정리 중 오류 발생: {str(e)}")
            
    async def verify_state(self, state: SystemState) -> bool:
        """
        상태 유효성 검증
        
        Args:
            state (SystemState): 검증할 시스템 상태
            
        Returns:
            bool: 유효성 검증 결과
        """
        try:
            # 타임스탬프 검증
            if (datetime.now() - state.timestamp).total_seconds() > self.config['recovery_timeout']:
                logger.warning("상태 데이터가 너무 오래되었습니다")
                return False
                
            # 계좌 잔고 검증
            if state.account_balance < 0:
                logger.warning("잘못된 계좌 잔고")
                return False
                
            # 포지션 데이터 검증
            for position in state.positions.values():
                if not self._verify_position(position):
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"상태 검증 중 오류 발생: {str(e)}")
            return False
            
    def _verify_position(self, position: Dict) -> bool:
        """
        포지션 데이터 검증
        
        Args:
            position (Dict): 검증할 포지션 데이터
            
        Returns:
            bool: 검증 결과
        """
        try:
            required_fields = ['symbol', 'size', 'entry_price', 'current_price']
            if not all(field in position for field in required_fields):
                logger.warning(f"필수 포지션 필드 누락: {position}")
                return False
                
            if position['size'] == 0:
                logger.warning(f"크기가 0인 포지션: {position}")
                return False
                
            if position['entry_price'] <= 0 or position['current_price'] <= 0:
                logger.warning(f"잘못된 가격 데이터: {position}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"포지션 검증 중 오류 발생: {str(e)}")
            return False 