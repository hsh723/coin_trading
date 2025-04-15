from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """포지션 정보를 저장하는 데이터 클래스"""
    symbol: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    liquidation_price: float
    margin_ratio: float
    timestamp: datetime

class PositionManager:
    """포지션 관리자"""
    
    def __init__(self, config: dict = None):
        """초기화"""
        self.config = config or {}
        self.positions = {}
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self) -> bool:
        """포지션 관리자 초기화"""
        try:
            # 포지션 데이터 초기화
            self.positions = {}
            self.position_history = []
            self.position_metrics = {}
            
            # 설정 로드
            self.max_position_size = self.config.get('max_position_size', 1000.0)
            self.max_leverage = self.config.get('max_leverage', 10.0)
            self.stop_loss_pct = self.config.get('stop_loss_pct', 0.05)
            self.take_profit_pct = self.config.get('take_profit_pct', 0.1)
            
            self.logger.info("포지션 관리자가 성공적으로 초기화되었습니다")
            return True
            
        except Exception as e:
            self.logger.error(f"포지션 관리자 초기화 중 오류 발생: {str(e)}")
            return False
            
    async def get_position(self, symbol: str) -> Dict[str, Any]:
        """포지션 조회"""
        try:
            if symbol not in self.positions:
                return {
                    'symbol': symbol,
                    'size': 0.0,
                    'entry_price': 0.0,
                    'liquidation_price': 0.0,
                    'margin_ratio': 0.0,
                    'unrealized_pnl': 0.0,
                    'side': None,
                    'timestamp': datetime.now()
                }
            return self.positions[symbol]
            
        except Exception as e:
            self.logger.error(f"포지션 조회 실패: {str(e)}")
            raise
            
    async def adjust_position(self, symbol: str, target_size: float) -> Dict[str, Any]:
        """포지션 조정"""
        try:
            current_position = await self.get_position(symbol)
            
            # 포지션 조정 로직 구현
            self.positions[symbol] = {
                'symbol': symbol,
                'size': target_size,
                'entry_price': 0.0,
                'liquidation_price': 0.0,
                'margin_ratio': 0.0,
                'unrealized_pnl': 0.0,
                'side': 'LONG' if target_size > 0 else 'SHORT' if target_size < 0 else None,
                'timestamp': datetime.now(),
                'success': True
            }
            
            return self.positions[symbol]
            
        except Exception as e:
            self.logger.error(f"포지션 조정 실패: {str(e)}")
            raise
            
    async def update_position(self, symbol: str, execution_data: Dict[str, Any]) -> Position:
        """
        포지션 업데이트
        
        Args:
            symbol (str): 거래 심볼
            execution_data (Dict[str, Any]): 실행 데이터
            
        Returns:
            Position: 업데이트된 포지션
        """
        try:
            if symbol not in self.positions:
                self.positions[symbol] = self._create_position(execution_data)
            else:
                self._modify_position(symbol, execution_data)
                
            return await self._calculate_position_metrics(symbol)
            
        except Exception as e:
            logger.error(f"포지션 업데이트 실패: {str(e)}")
            raise
            
    def _create_position(self, execution_data: Dict[str, Any]) -> Position:
        """
        새로운 포지션 생성
        
        Args:
            execution_data (Dict[str, Any]): 실행 데이터
            
        Returns:
            Position: 생성된 포지션
        """
        return Position(
            symbol=execution_data['symbol'],
            size=execution_data['size'],
            entry_price=execution_data['price'],
            current_price=execution_data['price'],
            unrealized_pnl=0.0,
            liquidation_price=self._calculate_liquidation_price(execution_data),
            margin_ratio=self._calculate_margin_ratio(execution_data),
            timestamp=datetime.now()
        )
        
    def _modify_position(self, symbol: str, execution_data: Dict[str, Any]):
        """
        기존 포지션 수정
        
        Args:
            symbol (str): 거래 심볼
            execution_data (Dict[str, Any]): 실행 데이터
        """
        position = self.positions[symbol]
        new_size = position.size + execution_data['size']
        
        if new_size == 0:
            del self.positions[symbol]
        else:
            # 진입 가격 재계산
            total_value = position.entry_price * position.size
            new_value = execution_data['price'] * execution_data['size']
            position.entry_price = (total_value + new_value) / new_size
            
            position.size = new_size
            position.current_price = execution_data['price']
            position.timestamp = datetime.now()
            
    async def _calculate_position_metrics(self, symbol: str) -> Position:
        """
        포지션 메트릭 계산
        
        Args:
            symbol (str): 거래 심볼
            
        Returns:
            Position: 계산된 메트릭이 포함된 포지션
        """
        position = self.positions[symbol]
        position.unrealized_pnl = (
            (position.current_price - position.entry_price) * position.size
        )
        position.liquidation_price = self._calculate_liquidation_price({
            'symbol': symbol,
            'size': position.size,
            'price': position.current_price
        })
        position.margin_ratio = self._calculate_margin_ratio({
            'symbol': symbol,
            'size': position.size,
            'price': position.current_price
        })
        
        return position
        
    def _calculate_liquidation_price(self, execution_data: Dict[str, Any]) -> float:
        """
        청산 가격 계산
        
        Args:
            execution_data (Dict[str, Any]): 실행 데이터
            
        Returns:
            float: 청산 가격
        """
        try:
            position_value = execution_data['size'] * execution_data['price']
            leverage = self.config['max_leverage']
            maintenance_margin = position_value * self.config['min_margin_ratio']
            
            # 롱 포지션의 경우
            if execution_data['size'] > 0:
                liquidation_price = execution_data['price'] * (1 - 1/leverage + self.config['min_margin_ratio'])
            # 숏 포지션의 경우
            else:
                liquidation_price = execution_data['price'] * (1 + 1/leverage - self.config['min_margin_ratio'])
                
            return liquidation_price
            
        except Exception as e:
            logger.error(f"청산 가격 계산 실패: {str(e)}")
            return 0.0
        
    def _calculate_margin_ratio(self, execution_data: Dict[str, Any]) -> float:
        """
        마진 비율 계산
        
        Args:
            execution_data (Dict[str, Any]): 실행 데이터
            
        Returns:
            float: 마진 비율
        """
        try:
            position_value = execution_data['size'] * execution_data['price']
            leverage = self.config['max_leverage']
            initial_margin = position_value / leverage
            
            # 마진 비율 = (포지션 가치 - 초기 마진) / 포지션 가치
            margin_ratio = (position_value - initial_margin) / position_value
            
            return margin_ratio
            
        except Exception as e:
            logger.error(f"마진 비율 계산 실패: {str(e)}")
            return 0.0 