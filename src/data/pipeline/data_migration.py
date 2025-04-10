from typing import Dict, List
import pandas as pd
from dataclasses import dataclass

@dataclass
class MigrationStatus:
    total_records: int
    migrated_records: int
    failed_records: int
    errors: List[str]

class DataMigrationManager:
    def __init__(self, source_config: Dict, target_config: Dict):
        self.source_config = source_config
        self.target_config = target_config
        
    async def migrate_data(self, table_name: str) -> MigrationStatus:
        """데이터 마이그레이션 실행"""
        try:
            source_data = await self._fetch_source_data(table_name)
            transformed_data = self._transform_data(source_data)
            migration_result = await self._write_target_data(
                table_name, transformed_data
            )
            
            return migration_result
        except Exception as e:
            await self._handle_migration_error(e)
