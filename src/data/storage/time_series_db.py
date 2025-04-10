import pandas as pd
from typing import Dict, List
from influxdb_client import InfluxDBClient

class TimeSeriesDB:
    def __init__(self, connection_config: Dict):
        self.client = InfluxDBClient(**connection_config)
        self.write_api = self.client.write_api()
        
    async def store_data(self, data: pd.DataFrame, measurement: str) -> bool:
        """시계열 데이터 저장"""
        try:
            self.write_api.write(
                bucket=self.config['bucket'],
                org=self.config['org'],
                record=data,
                data_frame_measurement_name=measurement
            )
            return True
        except Exception as e:
            self._handle_storage_error(e)
            return False
