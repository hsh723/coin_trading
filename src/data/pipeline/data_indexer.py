from typing import Dict, List
import pandas as pd
from dataclasses import dataclass

@dataclass
class IndexMetadata:
    index_name: str
    column_names: List[str]
    unique_values: int
    last_updated: float

class DataIndexer:
    def __init__(self, index_config: Dict = None):
        self.index_config = index_config or {
            'default_index': ['timestamp', 'symbol']
        }
        self.indexes = {}
        
    async def create_index(self, data: pd.DataFrame, 
                          columns: List[str]) -> IndexMetadata:
        """데이터 인덱스 생성"""
        index_name = '_'.join(columns)
        index_df = data[columns].copy()
        index_df.sort_values(columns, inplace=True)
        
        self.indexes[index_name] = index_df
        
        return IndexMetadata(
            index_name=index_name,
            column_names=columns,
            unique_values=len(index_df),
            last_updated=pd.Timestamp.now().timestamp()
        )
