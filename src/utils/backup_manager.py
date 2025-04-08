"""
자동 백업 및 복구 시스템
"""

import os
import shutil
import logging
from datetime import datetime
import json
from typing import Dict, Any, Optional
import asyncio
import aiofiles

class BackupManager:
    """백업 관리자 클래스"""
    
    def __init__(self, backup_dir: str, retention_days: int = 7):
        """
        초기화
        
        Args:
            backup_dir (str): 백업 디렉토리 경로
            retention_days (int): 백업 보관 기간 (일)
        """
        self.backup_dir = backup_dir
        self.retention_days = retention_days
        self.logger = logging.getLogger(__name__)
        
        # 백업 디렉토리 생성
        os.makedirs(backup_dir, exist_ok=True)
        
    async def backup_database(self, db_path: str) -> bool:
        """
        데이터베이스 백업
        
        Args:
            db_path (str): 데이터베이스 파일 경로
            
        Returns:
            bool: 백업 성공 여부
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(
                self.backup_dir,
                f'db_backup_{timestamp}.db'
            )
            
            # 데이터베이스 파일 복사
            shutil.copy2(db_path, backup_path)
            
            # 백업 메타데이터 저장
            await self._save_metadata({
                'type': 'database',
                'timestamp': timestamp,
                'original_path': db_path,
                'backup_path': backup_path
            })
            
            self.logger.info(f"데이터베이스 백업 완료: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"데이터베이스 백업 실패: {str(e)}")
            return False
            
    async def backup_config(self, config: Dict[str, Any]) -> bool:
        """
        설정 파일 백업
        
        Args:
            config (Dict[str, Any]): 설정 데이터
            
        Returns:
            bool: 백업 성공 여부
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(
                self.backup_dir,
                f'config_backup_{timestamp}.json'
            )
            
            # 설정 데이터 저장
            async with aiofiles.open(backup_path, 'w') as f:
                await f.write(json.dumps(config, indent=2))
                
            # 백업 메타데이터 저장
            await self._save_metadata({
                'type': 'config',
                'timestamp': timestamp,
                'backup_path': backup_path
            })
            
            self.logger.info(f"설정 파일 백업 완료: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"설정 파일 백업 실패: {str(e)}")
            return False
            
    async def restore_database(self, backup_path: str, target_path: str) -> bool:
        """
        데이터베이스 복구
        
        Args:
            backup_path (str): 백업 파일 경로
            target_path (str): 복구 대상 경로
            
        Returns:
            bool: 복구 성공 여부
        """
        try:
            # 백업 파일 복사
            shutil.copy2(backup_path, target_path)
            
            self.logger.info(f"데이터베이스 복구 완료: {target_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"데이터베이스 복구 실패: {str(e)}")
            return False
            
    async def restore_config(self, backup_path: str) -> Optional[Dict[str, Any]]:
        """
        설정 파일 복구
        
        Args:
            backup_path (str): 백업 파일 경로
            
        Returns:
            Optional[Dict[str, Any]]: 복구된 설정 데이터
        """
        try:
            async with aiofiles.open(backup_path, 'r') as f:
                content = await f.read()
                config = json.loads(content)
                
            self.logger.info(f"설정 파일 복구 완료: {backup_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"설정 파일 복구 실패: {str(e)}")
            return None
            
    async def cleanup_old_backups(self) -> bool:
        """
        오래된 백업 파일 정리
        
        Returns:
            bool: 정리 성공 여부
        """
        try:
            now = datetime.now()
            for filename in os.listdir(self.backup_dir):
                file_path = os.path.join(self.backup_dir, filename)
                if os.path.isfile(file_path):
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if (now - file_time).days > self.retention_days:
                        os.remove(file_path)
                        self.logger.info(f"오래된 백업 파일 삭제: {file_path}")
                        
            return True
            
        except Exception as e:
            self.logger.error(f"백업 파일 정리 실패: {str(e)}")
            return False
            
    async def _save_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        백업 메타데이터 저장
        
        Args:
            metadata (Dict[str, Any]): 메타데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            metadata_path = os.path.join(
                self.backup_dir,
                'backup_metadata.json'
            )
            
            # 기존 메타데이터 로드
            existing_metadata = []
            if os.path.exists(metadata_path):
                async with aiofiles.open(metadata_path, 'r') as f:
                    content = await f.read()
                    existing_metadata = json.loads(content)
                    
            # 새 메타데이터 추가
            existing_metadata.append(metadata)
            
            # 메타데이터 저장
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(json.dumps(existing_metadata, indent=2))
                
            return True
            
        except Exception as e:
            self.logger.error(f"메타데이터 저장 실패: {str(e)}")
            return False 