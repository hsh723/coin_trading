from typing import Dict, List
from dataclasses import dataclass

@dataclass
class TransformedData:
    exchange_id: str
    symbol: str
    data_type: str
    transformed: Dict
    metadata: Dict

class ApiDataTransformer:
    def __init__(self, schema_config: Dict = None):
        self.config = schema_config or {
            'standardize_fields': True,
            'include_metadata': True
        }
        
    async def transform_data(self, raw_data: Dict, 
                           exchange_id: str) -> TransformedData:
        """API 응답 데이터 변환"""
        try:
            standardized = self._standardize_fields(raw_data)
            enriched = self._enrich_data(standardized)
            validated = self._validate_schema(enriched)
            
            return TransformedData(
                exchange_id=exchange_id,
                symbol=raw_data.get('symbol'),
                data_type=raw_data.get('type'),
                transformed=validated,
                metadata=self._extract_metadata(raw_data)
            )
        except Exception as e:
            await self._handle_transform_error(e, raw_data)
