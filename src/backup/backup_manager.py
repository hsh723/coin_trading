"""
데이터 백업 및 복구 관리 모듈
"""

import os
import shutil
import zipfile
import logging
from datetime import datetime
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..utils.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

class BackupManager:
    """백업 관리 클래스"""
    
    def __init__(
        self,
        database_manager: Optional[DatabaseManager] = None,
        backup_dir: str = "backups",
        max_backups: int = 10
    ):
        self.database_manager = database_manager
        self.backup_dir = Path(backup_dir)
        self.max_backups = max_backups
        self.logger = logging.getLogger(__name__)
        
        # 백업 디렉토리 생성
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    async def create_backup(
        self,
        include_database: bool = True,
        include_config: bool = True,
        include_logs: bool = True,
        include_strategies: bool = True
    ) -> str:
        """백업 생성"""
        try:
            # 백업 이름 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"
            backup_path = self.backup_dir / backup_name
            
            # 백업 디렉토리 생성
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # 메타데이터 초기화
            metadata = {
                'name': backup_name,
                'timestamp': timestamp,
                'include_database': include_database,
                'include_config': include_config,
                'include_logs': include_logs,
                'include_strategies': include_strategies
            }
            
            # 데이터베이스 백업
            if include_database and self.database_manager:
                await self._backup_database(backup_path)
            
            # 설정 파일 백업
            if include_config:
                await self._backup_config(backup_path)
            
            # 로그 파일 백업
            if include_logs:
                await self._backup_logs(backup_path)
            
            # 전략 파일 백업
            if include_strategies:
                await self._backup_strategies(backup_path)
            
            # 메타데이터 저장
            with open(backup_path / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 백업 압축
            zip_path = self.backup_dir / f"{backup_name}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(backup_path):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(backup_path)
                        zipf.write(file_path, arcname)
            
            # 임시 디렉토리 삭제
            shutil.rmtree(backup_path)
            
            # 오래된 백업 정리
            await self._cleanup_old_backups()
            
            self.logger.info(f"백업 생성 완료: {backup_name}")
            return backup_name
            
        except Exception as e:
            self.logger.error(f"백업 생성 중 오류 발생: {str(e)}")
            raise
    
    async def restore_backup(self, backup_name: str) -> bool:
        """백업 복구"""
        try:
            # 백업 파일 경로
            zip_path = self.backup_dir / f"{backup_name}.zip"
            if not zip_path.exists():
                self.logger.error(f"백업 파일이 존재하지 않습니다: {backup_name}")
                return False
            
            # 임시 디렉토리 생성
            temp_dir = self.backup_dir / f"temp_{backup_name}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # 백업 압축 해제
            with zipfile.ZipFile(zip_path, "r") as zipf:
                zipf.extractall(temp_dir)
            
            # 메타데이터 로드
            metadata_path = temp_dir / "metadata.json"
            if not metadata_path.exists():
                self.logger.error("메타데이터 파일이 존재하지 않습니다.")
                return False
            
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # 데이터베이스 복구
            if metadata.get('include_database', False) and self.database_manager:
                await self._restore_database(temp_dir)
            
            # 설정 파일 복구
            if metadata.get('include_config', False):
                await self._restore_config(temp_dir)
            
            # 로그 파일 복구
            if metadata.get('include_logs', False):
                await self._restore_logs(temp_dir)
            
            # 전략 파일 복구
            if metadata.get('include_strategies', False):
                await self._restore_strategies(temp_dir)
            
            # 임시 디렉토리 삭제
            shutil.rmtree(temp_dir)
            
            self.logger.info(f"백업 복구 완료: {backup_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"백업 복구 중 오류 발생: {str(e)}")
            return False
    
    async def list_backups(self) -> List[Dict[str, Any]]:
        """백업 목록 조회"""
        try:
            backups = []
            
            for zip_file in self.backup_dir.glob("backup_*.zip"):
                backup_name = zip_file.stem
                timestamp = backup_name.split("_")[1]
                
                # 메타데이터 로드
                with zipfile.ZipFile(zip_file, "r") as zipf:
                    try:
                        with zipf.open("metadata.json") as f:
                            metadata = json.load(f)
                    except KeyError:
                        metadata = {
                            'name': backup_name,
                            'timestamp': timestamp,
                            'include_database': False,
                            'include_config': False,
                            'include_logs': False,
                            'include_strategies': False
                        }
                
                backups.append({
                    'name': backup_name,
                    'timestamp': timestamp,
                    'size': zip_file.stat().st_size,
                    'metadata': metadata
                })
            
            # 타임스탬프 기준 정렬
            backups.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return backups
            
        except Exception as e:
            self.logger.error(f"백업 목록 조회 중 오류 발생: {str(e)}")
            return []
    
    async def delete_backup(self, backup_name: str) -> bool:
        """백업 삭제"""
        try:
            zip_path = self.backup_dir / f"{backup_name}.zip"
            if not zip_path.exists():
                self.logger.error(f"백업 파일이 존재하지 않습니다: {backup_name}")
                return False
            
            zip_path.unlink()
            self.logger.info(f"백업 삭제 완료: {backup_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"백업 삭제 중 오류 발생: {str(e)}")
            return False
    
    async def _backup_database(self, backup_path: Path):
        """데이터베이스 백업"""
        if self.database_manager:
            await self.database_manager.backup(backup_path / "database")
    
    async def _backup_config(self, backup_path: Path):
        """설정 파일 백업"""
        config_dir = Path("config")
        if config_dir.exists():
            shutil.copytree(config_dir, backup_path / "config")
    
    async def _backup_logs(self, backup_path: Path):
        """로그 파일 백업"""
        log_dir = Path("logs")
        if log_dir.exists():
            shutil.copytree(log_dir, backup_path / "logs")
    
    async def _backup_strategies(self, backup_path: Path):
        """전략 파일 백업"""
        strategy_dir = Path("src/strategy")
        if strategy_dir.exists():
            shutil.copytree(strategy_dir, backup_path / "strategy")
    
    async def _restore_database(self, backup_path: Path):
        """데이터베이스 복구"""
        if self.database_manager:
            await self.database_manager.restore(backup_path / "database")
    
    async def _restore_config(self, backup_path: Path):
        """설정 파일 복구"""
        config_path = backup_path / "config"
        if config_path.exists():
            shutil.rmtree("config", ignore_errors=True)
            shutil.copytree(config_path, "config")
    
    async def _restore_logs(self, backup_path: Path):
        """로그 파일 복구"""
        log_path = backup_path / "logs"
        if log_path.exists():
            shutil.rmtree("logs", ignore_errors=True)
            shutil.copytree(log_path, "logs")
    
    async def _restore_strategies(self, backup_path: Path):
        """전략 파일 복구"""
        strategy_path = backup_path / "strategy"
        if strategy_path.exists():
            shutil.rmtree("src/strategy", ignore_errors=True)
            shutil.copytree(strategy_path, "src/strategy")
    
    async def _cleanup_old_backups(self):
        """오래된 백업 정리"""
        try:
            backups = await self.list_backups()
            if len(backups) > self.max_backups:
                for backup in backups[self.max_backups:]:
                    await self.delete_backup(backup['name'])
                
        except Exception as e:
            self.logger.error(f"오래된 백업 정리 중 오류 발생: {str(e)}") 