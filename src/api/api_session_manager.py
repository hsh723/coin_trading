from typing import Dict, Optional
from dataclasses import dataclass
import aiohttp
import time

@dataclass
class SessionInfo:
    session_id: str
    created_at: float
    last_used: float
    total_requests: int
    active: bool

class ApiSessionManager:
    def __init__(self, session_config: Dict = None):
        self.config = session_config or {
            'session_timeout': 3600,
            'max_sessions': 5
        }
        self.sessions = {}
        
    async def create_session(self, exchange_id: str) -> SessionInfo:
        """API 세션 생성"""
        if len(self.sessions) >= self.config['max_sessions']:
            await self._cleanup_old_sessions()
            
        session = aiohttp.ClientSession()
        session_info = SessionInfo(
            session_id=f"{exchange_id}_{int(time.time())}",
            created_at=time.time(),
            last_used=time.time(),
            total_requests=0,
            active=True
        )
        
        self.sessions[session_info.session_id] = {
            'session': session,
            'info': session_info
        }
        
        return session_info
