from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class ResponseInfo:
    success: bool
    data: Optional[Dict]
    error: Optional[str]
    response_time: float
    metadata: Dict

class ApiResponseHandler:
    def __init__(self, response_config: Dict = None):
        self.config = response_config or {
            'validate_response': True,
            'max_response_size': 1024 * 1024  # 1MB
        }
        
    async def handle_response(self, response: Dict) -> ResponseInfo:
        """API 응답 처리"""
        try:
            if self._is_valid_response(response):
                processed_data = self._process_response_data(response)
                return ResponseInfo(
                    success=True,
                    data=processed_data,
                    error=None,
                    response_time=self._get_response_time(response),
                    metadata=self._extract_metadata(response)
                )
            else:
                return self._handle_invalid_response(response)
        except Exception as e:
            await self._handle_processing_error(e, response)
