"""
실행 시스템 설정 관리 모듈
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """실행 시스템 설정 관리"""
    
    def __init__(self, config_dir: str = 'config/execution'):
        """
        설정 관리자 초기화
        
        Args:
            config_dir (str): 설정 디렉토리
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 기본 설정
        self.default_config = {
            'execution': {
                'max_retries': 3,
                'retry_delay': 1.0,
                'timeout': 30.0,
                'max_slippage': 0.01,
                'min_order_size': 0.001
            },
            'risk': {
                'max_position_size': 1.0,
                'max_leverage': 10.0,
                'max_drawdown': 0.1,
                'stop_loss': 0.05,
                'take_profit': 0.1
            },
            'performance': {
                'latency_threshold': 1000.0,
                'success_rate_threshold': 0.95,
                'fill_rate_threshold': 0.9
            },
            'logging': {
                'log_level': 'INFO',
                'log_dir': 'logs/execution',
                'max_log_size': 10485760,  # 10MB
                'backup_count': 5
            },
            'notification': {
                'enabled': True,
                'types': ['telegram'],
                'telegram': {
                    'token': '',
                    'chat_id': ''
                }
            }
        }
        
        # 현재 설정
        self.config = self.default_config.copy()
        
        # 설정 이력
        self.config_history = []
        
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """
        설정 파일 로드
        
        Args:
            config_file (str): 설정 파일 경로
            
        Returns:
            Dict[str, Any]: 설정 데이터
        """
        try:
            config_path = self.config_dir / config_file
            
            if not config_path.exists():
                logger.warning(f"설정 파일이 없습니다: {config_file}")
                return self.default_config
                
            # 파일 확장자에 따라 로드
            if config_path.suffix == '.json':
                with open(config_path, 'r') as f:
                    config = json.load(f)
            elif config_path.suffix in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {config_path.suffix}")
                
            # 기본 설정과 병합
            self._merge_config(config)
            
            # 설정 이력 기록
            self._record_config_change('load', config_file)
            
            logger.info(f"설정 파일 로드 완료: {config_file}")
            return self.config
            
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {str(e)}")
            return self.default_config
            
    def save_config(
        self,
        config_file: str,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        설정 파일 저장
        
        Args:
            config_file (str): 설정 파일 경로
            config (Optional[Dict[str, Any]]): 저장할 설정
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            config_path = self.config_dir / config_file
            
            # 저장할 설정이 없으면 현재 설정 사용
            if config is None:
                config = self.config
                
            # 파일 확장자에 따라 저장
            if config_path.suffix == '.json':
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=4)
            elif config_path.suffix in ['.yaml', '.yml']:
                with open(config_path, 'w') as f:
                    yaml.dump(config, f)
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {config_path.suffix}")
                
            # 설정 이력 기록
            self._record_config_change('save', config_file)
            
            logger.info(f"설정 파일 저장 완료: {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"설정 파일 저장 실패: {str(e)}")
            return False
            
    def update_config(
        self,
        updates: Dict[str, Any],
        save: bool = True
    ) -> bool:
        """
        설정 업데이트
        
        Args:
            updates (Dict[str, Any]): 업데이트할 설정
            save (bool): 저장 여부
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            # 설정 업데이트
            self._merge_config(updates)
            
            # 설정 이력 기록
            self._record_config_change('update', str(updates))
            
            # 설정 저장
            if save:
                return self.save_config('config.yaml')
                
            return True
            
        except Exception as e:
            logger.error(f"설정 업데이트 실패: {str(e)}")
            return False
            
    def get_config(
        self,
        key: Optional[str] = None,
        default: Any = None
    ) -> Any:
        """
        설정 조회
        
        Args:
            key (Optional[str]): 설정 키
            default (Any): 기본값
            
        Returns:
            Any: 설정 값
        """
        try:
            if key is None:
                return self.config
                
            # 키 경로 분리
            keys = key.split('.')
            value = self.config
            
            # 중첩된 키 조회
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
                    
            return value
            
        except Exception as e:
            logger.error(f"설정 조회 실패: {str(e)}")
            return default
            
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        설정 유효성 검사
        
        Args:
            config (Dict[str, Any]): 검사할 설정
            
        Returns:
            bool: 유효성 여부
        """
        try:
            # 필수 설정 확인
            required_sections = [
                'execution',
                'risk',
                'performance',
                'logging',
                'notification'
            ]
            
            for section in required_sections:
                if section not in config:
                    logger.error(f"필수 설정이 없습니다: {section}")
                    return False
                    
            # 실행 설정 검사
            execution = config['execution']
            if not all(k in execution for k in [
                'max_retries',
                'retry_delay',
                'timeout',
                'max_slippage',
                'min_order_size'
            ]):
                logger.error("실행 설정이 잘못되었습니다")
                return False
                
            # 위험 설정 검사
            risk = config['risk']
            if not all(k in risk for k in [
                'max_position_size',
                'max_leverage',
                'max_drawdown',
                'stop_loss',
                'take_profit'
            ]):
                logger.error("위험 설정이 잘못되었습니다")
                return False
                
            # 성능 설정 검사
            performance = config['performance']
            if not all(k in performance for k in [
                'latency_threshold',
                'success_rate_threshold',
                'fill_rate_threshold'
            ]):
                logger.error("성능 설정이 잘못되었습니다")
                return False
                
            # 로깅 설정 검사
            logging = config['logging']
            if not all(k in logging for k in [
                'log_level',
                'log_dir',
                'max_log_size',
                'backup_count'
            ]):
                logger.error("로깅 설정이 잘못되었습니다")
                return False
                
            # 알림 설정 검사
            notification = config['notification']
            if not all(k in notification for k in [
                'enabled',
                'types',
                'telegram'
            ]):
                logger.error("알림 설정이 잘못되었습니다")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"설정 유효성 검사 실패: {str(e)}")
            return False
            
    def _merge_config(self, updates: Dict[str, Any]) -> None:
        """
        설정 병합
        
        Args:
            updates (Dict[str, Any]): 업데이트할 설정
        """
        def merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> None:
            for key, value in update.items():
                if (
                    key in base and
                    isinstance(base[key], dict) and
                    isinstance(value, dict)
                ):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
                    
        merge_dict(self.config, updates)
        
    def _record_config_change(
        self,
        action: str,
        details: str
    ) -> None:
        """
        설정 변경 이력 기록
        
        Args:
            action (str): 변경 작업
            details (str): 변경 상세
        """
        self.config_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        })
        
    def get_config_history(self) -> list:
        """
        설정 이력 조회
        
        Returns:
            list: 설정 이력
        """
        return self.config_history
        
    def reset_config(self) -> None:
        """설정 초기화"""
        self.config = self.default_config.copy()
        self._record_config_change('reset', 'default') 