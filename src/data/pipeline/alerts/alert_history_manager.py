from typing import Dict, List
from dataclasses import dataclass
import sqlite3
import json

@dataclass
class AlertHistory:
    alert_id: str
    created_at: float
    resolved_at: float
    resolution: str
    metadata: Dict

class AlertHistoryManager:
    def __init__(self, db_path: str = "alerts.db"):
        self.db_path = db_path
        self._init_database()
        
    async def store_alert(self, alert: Dict) -> str:
        """알림 히스토리 저장"""
        alert_id = alert.get('id')
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alert_history 
                (alert_id, alert_data, created_at) 
                VALUES (?, ?, ?)
            """, (alert_id, json.dumps(alert), time.time()))
        return alert_id
