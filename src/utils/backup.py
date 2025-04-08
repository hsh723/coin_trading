"""
백업 및 복구 모듈
"""

import os
import shutil
import logging
from datetime import datetime
from typing import Optional
import json
import zipfile
from pathlib import Path
import boto3
import schedule
import time
import hashlib
from google.cloud import storage
from azure.storage.blob import BlobServiceClient
from ..utils.logger import setup_logger
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

logger = setup_logger()

class BackupSystem:
    """
    데이터 백업 및 복구 시스템
    
    Attributes:
        backup_dir (str): 로컬 백업 디렉토리
        cloud_provider (str): 클라우드 스토리지 제공자
        cloud_config (Dict): 클라우드 설정
        retention_days (int): 백업 보관 기간
    """
    
    def __init__(
        self,
        backup_dir: str = 'backups',
        cloud_provider: str = 'aws',
        cloud_config: Optional[Dict] = None,
        retention_days: int = 30
    ):
        self.backup_dir = Path(backup_dir)
        self.cloud_provider = cloud_provider
        self.cloud_config = cloud_config or {}
        self.retention_days = retention_days
        
        # 백업 디렉토리 생성
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 클라우드 클라이언트 초기화
        self.cloud_client = self._init_cloud_client()
        
        # 백업 메타데이터 파일
        self.metadata_file = self.backup_dir / 'backup_metadata.json'
        self._load_metadata()
    
    def _init_cloud_client(self):
        """클라우드 스토리지 클라이언트를 초기화합니다."""
        if self.cloud_provider == 'aws':
            return boto3.client(
                's3',
                aws_access_key_id=self.cloud_config.get('aws_access_key'),
                aws_secret_access_key=self.cloud_config.get('aws_secret_key'),
                region_name=self.cloud_config.get('aws_region')
            )
        elif self.cloud_provider == 'gcp':
            return storage.Client()
        elif self.cloud_provider == 'azure':
            return BlobServiceClient.from_connection_string(
                self.cloud_config.get('azure_connection_string')
            )
        return None
    
    def _load_metadata(self):
        """백업 메타데이터를 로드합니다."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'backups': [],
                'last_full_backup': None,
                'last_incremental_backup': None
            }
    
    def _save_metadata(self):
        """백업 메타데이터를 저장합니다."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """
        파일의 체크섬을 계산합니다.
        
        Args:
            file_path (Path): 파일 경로
            
        Returns:
            str: 체크섬
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b''):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _verify_backup(self, backup_path: Path) -> bool:
        """
        백업의 무결성을 검증합니다.
        
        Args:
            backup_path (Path): 백업 파일 경로
            
        Returns:
            bool: 검증 결과
        """
        try:
            # 체크섬 검증
            stored_checksum = self.metadata['backups'][-1]['checksum']
            current_checksum = self._calculate_checksum(backup_path)
            
            if stored_checksum != current_checksum:
                logger.error(f"Backup verification failed: checksum mismatch")
                return False
            
            # 압축 파일 검증
            if backup_path.suffix == '.tar.gz':
                import tarfile
                with tarfile.open(backup_path, 'r:gz') as tar:
                    tar.getmembers()
            
            return True
            
        except Exception as e:
            logger.error(f"Backup verification failed: {str(e)}")
            return False
    
    def create_backup(self, source_dir: str, backup_type: str = 'full') -> bool:
        """
        백업을 생성합니다.
        
        Args:
            source_dir (str): 소스 디렉토리
            backup_type (str): 백업 유형 ('full' 또는 'incremental')
            
        Returns:
            bool: 백업 성공 여부
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"backup_{backup_type}_{timestamp}"
            backup_path = self.backup_dir / f"{backup_name}.tar.gz"
            
            # 백업 생성
            import tarfile
            with tarfile.open(backup_path, 'w:gz') as tar:
                tar.add(source_dir, arcname=os.path.basename(source_dir))
            
            # 체크섬 계산
            checksum = self._calculate_checksum(backup_path)
            
            # 메타데이터 업데이트
            backup_info = {
                'name': backup_name,
                'type': backup_type,
                'timestamp': timestamp,
                'path': str(backup_path),
                'checksum': checksum,
                'size': backup_path.stat().st_size
            }
            
            self.metadata['backups'].append(backup_info)
            
            if backup_type == 'full':
                self.metadata['last_full_backup'] = timestamp
            else:
                self.metadata['last_incremental_backup'] = timestamp
            
            self._save_metadata()
            
            # 클라우드 업로드
            if self.cloud_client:
                self._upload_to_cloud(backup_path)
            
            # 무결성 검증
            if not self._verify_backup(backup_path):
                raise Exception("Backup verification failed")
            
            logger.info(f"Backup created successfully: {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"Backup creation failed: {str(e)}")
            return False
    
    def _upload_to_cloud(self, backup_path: Path):
        """
        백업을 클라우드 스토리지에 업로드합니다.
        
        Args:
            backup_path (Path): 백업 파일 경로
        """
        try:
            if self.cloud_provider == 'aws':
                self.cloud_client.upload_file(
                    str(backup_path),
                    self.cloud_config['aws_bucket'],
                    backup_path.name
                )
            elif self.cloud_provider == 'gcp':
                bucket = self.cloud_client.bucket(self.cloud_config['gcp_bucket'])
                blob = bucket.blob(backup_path.name)
                blob.upload_from_filename(str(backup_path))
            elif self.cloud_provider == 'azure':
                container_client = self.cloud_client.get_container_client(
                    self.cloud_config['azure_container']
                )
                blob_client = container_client.get_blob_client(backup_path.name)
                with open(backup_path, 'rb') as f:
                    blob_client.upload_blob(f, overwrite=True)
                    
        except Exception as e:
            logger.error(f"Cloud upload failed: {str(e)}")
    
    def restore_backup(self, backup_name: str, target_dir: str) -> bool:
        """
        백업을 복원합니다.
        
        Args:
            backup_name (str): 백업 이름
            target_dir (str): 복원 대상 디렉토리
            
        Returns:
            bool: 복원 성공 여부
        """
        try:
            # 백업 정보 찾기
            backup_info = next(
                (b for b in self.metadata['backups'] if b['name'] == backup_name),
                None
            )
            
            if not backup_info:
                raise Exception(f"Backup not found: {backup_name}")
            
            backup_path = Path(backup_info['path'])
            
            # 클라우드에서 다운로드
            if not backup_path.exists() and self.cloud_client:
                self._download_from_cloud(backup_path)
            
            # 무결성 검증
            if not self._verify_backup(backup_path):
                raise Exception("Backup verification failed")
            
            # 백업 복원
            import tarfile
            with tarfile.open(backup_path, 'r:gz') as tar:
                tar.extractall(target_dir)
            
            logger.info(f"Backup restored successfully: {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"Backup restoration failed: {str(e)}")
            return False
    
    def _download_from_cloud(self, backup_path: Path):
        """
        백업을 클라우드 스토리지에서 다운로드합니다.
        
        Args:
            backup_path (Path): 백업 파일 경로
        """
        try:
            if self.cloud_provider == 'aws':
                self.cloud_client.download_file(
                    self.cloud_config['aws_bucket'],
                    backup_path.name,
                    str(backup_path)
                )
            elif self.cloud_provider == 'gcp':
                bucket = self.cloud_client.bucket(self.cloud_config['gcp_bucket'])
                blob = bucket.blob(backup_path.name)
                blob.download_to_filename(str(backup_path))
            elif self.cloud_provider == 'azure':
                container_client = self.cloud_client.get_container_client(
                    self.cloud_config['azure_container']
                )
                blob_client = container_client.get_blob_client(backup_path.name)
                with open(backup_path, 'wb') as f:
                    f.write(blob_client.download_blob().readall())
                    
        except Exception as e:
            logger.error(f"Cloud download failed: {str(e)}")
    
    def cleanup_old_backups(self):
        """오래된 백업을 정리합니다."""
        try:
            current_time = datetime.now()
            
            for backup in self.metadata['backups'][:]:
                backup_time = datetime.strptime(backup['timestamp'], '%Y%m%d_%H%M%S')
                age_days = (current_time - backup_time).days
                
                if age_days > self.retention_days:
                    # 로컬 파일 삭제
                    backup_path = Path(backup['path'])
                    if backup_path.exists():
                        backup_path.unlink()
                    
                    # 클라우드 파일 삭제
                    if self.cloud_client:
                        self._delete_from_cloud(backup_path)
                    
                    # 메타데이터에서 제거
                    self.metadata['backups'].remove(backup)
            
            self._save_metadata()
            
        except Exception as e:
            logger.error(f"Backup cleanup failed: {str(e)}")
    
    def _delete_from_cloud(self, backup_path: Path):
        """
        백업을 클라우드 스토리지에서 삭제합니다.
        
        Args:
            backup_path (Path): 백업 파일 경로
        """
        try:
            if self.cloud_provider == 'aws':
                self.cloud_client.delete_object(
                    Bucket=self.cloud_config['aws_bucket'],
                    Key=backup_path.name
                )
            elif self.cloud_provider == 'gcp':
                bucket = self.cloud_client.bucket(self.cloud_config['gcp_bucket'])
                bucket.delete_blob(backup_path.name)
            elif self.cloud_provider == 'azure':
                container_client = self.cloud_client.get_container_client(
                    self.cloud_config['azure_container']
                )
                container_client.delete_blob(backup_path.name)
                
        except Exception as e:
            logger.error(f"Cloud deletion failed: {str(e)}")
    
    def schedule_backup(self, source_dir: str, schedule_type: str = 'daily'):
        """
        백업 스케줄을 설정합니다.
        
        Args:
            source_dir (str): 소스 디렉토리
            schedule_type (str): 스케줄 유형 ('daily', 'weekly', 'monthly')
        """
        try:
            if schedule_type == 'daily':
                schedule.every().day.at('00:00').do(
                    self.create_backup, source_dir, 'incremental'
                )
                schedule.every().sunday.at('00:00').do(
                    self.create_backup, source_dir, 'full'
                )
            elif schedule_type == 'weekly':
                schedule.every().sunday.at('00:00').do(
                    self.create_backup, source_dir, 'full'
                )
            elif schedule_type == 'monthly':
                schedule.every().month.at('00:00').do(
                    self.create_backup, source_dir, 'full'
                )
            
            # 오래된 백업 정리 스케줄
            schedule.every().day.at('01:00').do(self.cleanup_old_backups)
            
            # 스케줄러 실행
            while True:
                schedule.run_pending()
                time.sleep(60)
                
        except Exception as e:
            logger.error(f"Backup scheduling failed: {str(e)}")
    
    def list_backups(self) -> List[Dict]:
        """
        백업 목록을 조회합니다.
        
        Returns:
            List[Dict]: 백업 목록
        """
        return self.metadata['backups']
    
    def get_backup_info(self, backup_name: str) -> Optional[Dict]:
        """
        특정 백업의 정보를 조회합니다.
        
        Args:
            backup_name (str): 백업 이름
            
        Returns:
            Optional[Dict]: 백업 정보
        """
        return next(
            (b for b in self.metadata['backups'] if b['name'] == backup_name),
            None
        )

class BackupManager:
    """백업 관리자 클래스"""
    
    def __init__(self, backup_dir: str = "backups"):
        """
        초기화
        
        Args:
            backup_dir (str): 백업 디렉토리 경로
        """
        self.backup_dir = Path(backup_dir)
        self.logger = logging.getLogger(__name__)
        self._init_backup_dir()
        
    def _init_backup_dir(self):
        """백업 디렉토리 초기화"""
        try:
            os.makedirs(self.backup_dir, exist_ok=True)
        except Exception as e:
            self.logger.error(f"백업 디렉토리 생성 실패: {str(e)}")
            raise
            
    def create_backup(self, 
                     db_path: str,
                     config_path: str,
                     description: Optional[str] = None) -> str:
        """
        백업 생성
        
        Args:
            db_path (str): 데이터베이스 파일 경로
            config_path (str): 설정 파일 경로
            description (Optional[str]): 백업 설명
            
        Returns:
            str: 백업 파일 경로
        """
        try:
            # 백업 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"
            if description:
                backup_name += f"_{description.replace(' ', '_')}"
            backup_path = os.path.join(self.backup_dir, f"{backup_name}.zip")
            
            # 백업 파일 생성
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 데이터베이스 파일 백업
                if os.path.exists(db_path):
                    zipf.write(db_path, os.path.basename(db_path))
                
                # 설정 파일 백업
                if os.path.exists(config_path):
                    zipf.write(config_path, os.path.basename(config_path))
                
                # 메타데이터 추가
                metadata = {
                    'timestamp': timestamp,
                    'description': description,
                    'files': [
                        os.path.basename(db_path),
                        os.path.basename(config_path)
                    ]
                }
                zipf.writestr('metadata.json', json.dumps(metadata))
            
            self.logger.info(f"백업 생성 완료: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"백업 생성 실패: {str(e)}")
            raise
            
    def restore_backup(self, backup_path: str, restore_dir: str) -> bool:
        """
        백업 복구
        
        Args:
            backup_path (str): 백업 파일 경로
            restore_dir (str): 복구할 디렉토리
            
        Returns:
            bool: 복구 성공 여부
        """
        try:
            # 복구 디렉토리 생성
            os.makedirs(restore_dir, exist_ok=True)
            
            # 백업 파일 압축 해제
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                # 메타데이터 확인
                metadata = json.loads(zipf.read('metadata.json'))
                
                # 파일 복구
                for file in metadata['files']:
                    zipf.extract(file, restore_dir)
            
            self.logger.info(f"백업 복구 완료: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"백업 복구 실패: {str(e)}")
            return False
            
    def list_backups(self) -> list:
        """
        백업 목록 조회
        
        Returns:
            list: 백업 목록
        """
        try:
            backups = []
            for file in os.listdir(self.backup_dir):
                if file.endswith('.zip'):
                    backup_path = os.path.join(self.backup_dir, file)
                    with zipfile.ZipFile(backup_path, 'r') as zipf:
                        metadata = json.loads(zipf.read('metadata.json'))
                        backups.append({
                            'name': file,
                            'path': backup_path,
                            'timestamp': metadata['timestamp'],
                            'description': metadata['description']
                        })
            
            return sorted(backups, key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"백업 목록 조회 실패: {str(e)}")
            return []
            
    def delete_backup(self, backup_path: str) -> bool:
        """
        백업 삭제
        
        Args:
            backup_path (str): 백업 파일 경로
            
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            if os.path.exists(backup_path):
                os.remove(backup_path)
                self.logger.info(f"백업 삭제 완료: {backup_path}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"백업 삭제 실패: {str(e)}")
            return False
            
    def cleanup_old_backups(self, max_backups: int = 10) -> bool:
        """
        오래된 백업 정리
        
        Args:
            max_backups (int): 유지할 최대 백업 수
            
        Returns:
            bool: 정리 성공 여부
        """
        try:
            backups = self.list_backups()
            if len(backups) > max_backups:
                for backup in backups[max_backups:]:
                    self.delete_backup(backup['path'])
                self.logger.info(f"오래된 백업 {len(backups) - max_backups}개 정리 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"백업 정리 실패: {str(e)}")
            return False 