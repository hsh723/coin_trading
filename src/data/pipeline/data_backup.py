from typing import Dict, List
import shutil
import os
from dataclasses import dataclass

@dataclass
class BackupResult:
    success: bool
    backup_path: str
    backup_size: int
    error_message: str = None

class DataBackupManager:
    def __init__(self, backup_config: Dict):
        self.backup_config = backup_config
        self.backup_path = backup_config['path']
        
    async def create_backup(self, data_path: str, metadata: Dict) -> BackupResult:
        """데이터 백업 생성"""
        try:
            backup_name = self._generate_backup_name(metadata)
            backup_path = os.path.join(self.backup_path, backup_name)
            
            shutil.make_archive(
                backup_path,
                'zip',
                data_path
            )
            
            return BackupResult(
                success=True,
                backup_path=backup_path,
                backup_size=os.path.getsize(f"{backup_path}.zip")
            )
        except Exception as e:
            return BackupResult(
                success=False,
                backup_path="",
                backup_size=0,
                error_message=str(e)
            )
