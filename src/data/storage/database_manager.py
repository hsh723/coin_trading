import sqlite3
from typing import Dict, List
import pandas as pd
from dataclasses import dataclass

@dataclass
class DatabaseConfig:
    db_path: str
    table_schemas: Dict[str, List[str]]
    max_connections: int = 5

class DatabaseManager:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection_pool = []
        
    async def store_market_data(self, data: pd.DataFrame, table_name: str) -> bool:
        """시장 데이터 저장"""
        try:
            conn = await self._get_connection()
            data.to_sql(table_name, conn, if_exists='append', index=False)
            return True
        except Exception as e:
            self._handle_error(e)
            return False
        finally:
            await self._release_connection(conn)
