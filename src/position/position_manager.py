"""
포지션 관리 모듈
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

class PositionManager:
    """포지션 관리 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        포지션 매니저 초기화
        
        Args:
            config (Dict[str, Any]): 설정 정보
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 포지션 정보 초기화
        self.positions = {}
        self.position_history = []
        
        # 설정값 초기화
        self.max_positions = config.get('max_positions', 10)
        self.max_position_size = config.get('max_position_size', 1.0)
        self.max_leverage = config.get('max_leverage', 3.0)
        
    async def initialize(self):
        """포지션 매니저 초기화"""
        try:
            self.logger.info("포지션 매니저 초기화 완료")
        except Exception as e:
            self.logger.error(f"포지션 매니저 초기화 실패: {str(e)}")
            raise
            
    async def add_position(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """
        포지션 추가
        
        Args:
            position (Dict[str, Any]): 포지션 정보
            
        Returns:
            Dict[str, Any]: 포지션 추가 결과
        """
        try:
            symbol = position['symbol']
            
            # 기존 포지션이 있는 경우 업데이트
            if symbol in self.positions:
                self.positions[symbol].update(position)
            else:
                # 새로운 포지션 추가
                self.positions[symbol] = position
                
            # 포지션 이력 추가
            self.position_history.append({
                'timestamp': datetime.now(),
                'action': 'add',
                'symbol': symbol,
                'position': position.copy()
            })
            
            return {
                'success': True,
                'position': self.positions[symbol]
            }
            
        except Exception as e:
            self.logger.error(f"포지션 추가 실패: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def update_position(self, symbol: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        포지션 업데이트
        
        Args:
            symbol (str): 심볼
            updates (Dict[str, Any]): 업데이트할 정보
            
        Returns:
            Dict[str, Any]: 포지션 업데이트 결과
        """
        try:
            if symbol not in self.positions:
                return {
                    'success': False,
                    'error': f"포지션이 존재하지 않음: {symbol}"
                }
                
            # 포지션 업데이트
            self.positions[symbol].update(updates)
            
            # 포지션 이력 추가
            self.position_history.append({
                'timestamp': datetime.now(),
                'action': 'update',
                'symbol': symbol,
                'updates': updates.copy()
            })
            
            return {
                'success': True,
                'position': self.positions[symbol]
            }
            
        except Exception as e:
            self.logger.error(f"포지션 업데이트 실패: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def close_position(self, symbol: str) -> Dict[str, Any]:
        """
        포지션 종료
        
        Args:
            symbol (str): 심볼
            
        Returns:
            Dict[str, Any]: 포지션 종료 결과
        """
        try:
            if symbol not in self.positions:
                return {
                    'success': False,
                    'error': f"포지션이 존재하지 않음: {symbol}"
                }
                
            # 포지션 종료
            position = self.positions.pop(symbol)
            
            # 포지션 이력 추가
            self.position_history.append({
                'timestamp': datetime.now(),
                'action': 'close',
                'symbol': symbol,
                'position': position.copy()
            })
            
            return {
                'success': True,
                'position': position
            }
            
        except Exception as e:
            self.logger.error(f"포지션 종료 실패: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        포지션 조회
        
        Args:
            symbol (str): 심볼
            
        Returns:
            Dict[str, Any]: 포지션 정보
        """
        try:
            if symbol not in self.positions:
                return {
                    'success': False,
                    'error': f"포지션이 존재하지 않음: {symbol}"
                }
                
            return {
                'success': True,
                'position': self.positions[symbol]
            }
            
        except Exception as e:
            self.logger.error(f"포지션 조회 실패: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def get_all_positions(self) -> Dict[str, Any]:
        """
        전체 포지션 조회
        
        Returns:
            Dict[str, Any]: 전체 포지션 정보
        """
        try:
            return {
                'success': True,
                'positions': self.positions.copy()
            }
            
        except Exception as e:
            self.logger.error(f"전체 포지션 조회 실패: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def get_position_history(self) -> Dict[str, Any]:
        """
        포지션 이력 조회
        
        Returns:
            Dict[str, Any]: 포지션 이력
        """
        try:
            return {
                'success': True,
                'history': self.position_history.copy()
            }
            
        except Exception as e:
            self.logger.error(f"포지션 이력 조회 실패: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def close(self):
        """포지션 매니저 종료"""
        try:
            # 모든 포지션 종료
            for symbol in list(self.positions.keys()):
                await self.close_position(symbol)
                
            self.logger.info("포지션 매니저 종료 완료")
            
        except Exception as e:
            self.logger.error(f"포지션 매니저 종료 실패: {str(e)}")
            raise 