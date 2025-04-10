from typing import Dict, List
from dataclasses import dataclass

@dataclass
class DatasetInfo:
    dataset_id: str
    schema: Dict
    metadata: Dict
    lineage: List[str]
    quality_score: float

class DataCatalog:
    def __init__(self):
        self.datasets = {}
        
    async def register_dataset(self, dataset_info: DatasetInfo) -> bool:
        """데이터셋 등록"""
        try:
            self.datasets[dataset_info.dataset_id] = {
                'info': dataset_info,
                'last_updated': pd.Timestamp.now(),
                'status': 'active'
            }
            await self._update_catalog_index()
            return True
        except Exception as e:
            await self._handle_registration_error(e)
            return False
