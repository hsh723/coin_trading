from typing import Dict, List
import sqlite3
import json

class AlertPersistenceManager:
    def __init__(self, db_path: str = "alerts.db"):
        self.db_path = db_path
        self._initialize_db()
        
    def _initialize_db(self):
        """알림 데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    created_at FLOAT,
                    severity TEXT,
                    message TEXT,
                    metadata TEXT,
                    status TEXT
                )
            """)
            
    async def store_alert(self, alert: Dict) -> bool:
        """알림 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO alerts 
                    (alert_id, created_at, severity, message, metadata, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    alert['id'],
                    alert['created_at'],
                    alert['severity'],
                    alert['message'],
                    json.dumps(alert.get('metadata', {})),
                    'active'
                ))
            return True
        except Exception as e:
            await self._handle_storage_error(e)
            return False
