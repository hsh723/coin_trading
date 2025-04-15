import os
import json
import logging
import threading
import queue
import time
import pickle
import asyncio
import websockets
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
from pathlib import Path

class WebSocketManager:
    """WebSocket 클래스"""
    
    def __init__(self,
                 config_dir: str = "./config",
                 data_dir: str = "./data"):
        """
        WebSocket 초기화
        
        Args:
            config_dir: 설정 디렉토리
            data_dir: 데이터 디렉토리
        """
        self.config_dir = config_dir
        self.data_dir = data_dir
        
        # 로거 설정
        self.logger = logging.getLogger("websocket")
        
        # 메시지 큐
        self.message_queue = queue.Queue()
        
        # 디렉토리 생성
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # 설정 로드
        self.config = self._load_config()
        
        # WebSocket 서버
        self.server: Optional[websockets.WebSocketServer] = None
        
        # WebSocket 관리자
        self.is_running = False
        
        # 연결 관리
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        
        # 핸들러 관리
        self.handlers: Dict[str, List[Callable]] = {}
        
        # 통계
        self.stats = {
            "connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            config_path = os.path.join(self.config_dir, "websocket_config.json")
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"설정 파일 로드 중 오류 발생: {e}")
            return {
                "host": "localhost",
                "port": 8765,
                "ping_interval": 30,
                "ping_timeout": 10
            }
            
    def start(self) -> None:
        """WebSocket 서버 시작"""
        try:
            self.is_running = True
            
            # 메시지 처리 시작
            threading.Thread(target=self._process_messages, daemon=True).start()
            
            # 서버 시작
            asyncio.run(self._start_server())
            
            self.logger.info("WebSocket 서버가 시작되었습니다")
            
        except Exception as e:
            self.logger.error(f"WebSocket 서버 시작 중 오류 발생: {e}")
            raise
            
    def stop(self) -> None:
        """WebSocket 서버 중지"""
        try:
            self.is_running = False
            
            # 서버 종료
            if self.server:
                asyncio.run(self.server.close())
                
            # 연결 종료
            for connection in self.connections.values():
                asyncio.run(connection.close())
                
            self.logger.info("WebSocket 서버가 중지되었습니다")
            
        except Exception as e:
            self.logger.error(f"WebSocket 서버 중지 중 오류 발생: {e}")
            raise
            
    async def _start_server(self) -> None:
        """WebSocket 서버 시작"""
        try:
            self.server = await websockets.serve(
                self._handle_connection,
                self.config["host"],
                self.config["port"],
                ping_interval=self.config["ping_interval"],
                ping_timeout=self.config["ping_timeout"]
            )
            
        except Exception as e:
            self.logger.error(f"WebSocket 서버 시작 중 오류 발생: {e}")
            raise
            
    async def _handle_connection(self,
                               websocket: websockets.WebSocketServerProtocol,
                               path: str) -> None:
        """
        WebSocket 연결 처리
        
        Args:
            websocket: WebSocket 연결
            path: 연결 경로
        """
        try:
            # 연결 ID 생성
            connection_id = str(id(websocket))
            
            # 연결 관리
            self.connections[connection_id] = websocket
            self.stats["connections"] += 1
            
            self.logger.info(f"새로운 연결: {connection_id}")
            
            # 메시지 처리
            async for message in websocket:
                await self._process_message(connection_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"연결 종료: {connection_id}")
            
        except Exception as e:
            self.logger.error(f"연결 처리 중 오류 발생: {e}")
            self.stats["errors"] += 1
            
        finally:
            # 연결 정리
            if connection_id in self.connections:
                del self.connections[connection_id]
                self.stats["connections"] -= 1
                
    async def _process_message(self,
                             connection_id: str,
                             message: str) -> None:
        """
        메시지 처리
        
        Args:
            connection_id: 연결 ID
            message: 메시지
        """
        try:
            # 메시지 파싱
            data = json.loads(message)
            event = data.get("event")
            payload = data.get("payload")
            
            # 핸들러 호출
            if event in self.handlers:
                for handler in self.handlers[event]:
                    try:
                        handler(connection_id, payload)
                    except Exception as e:
                        self.logger.error(f"핸들러 실행 중 오류 발생: {e}")
                        self.stats["errors"] += 1
                        
            self.stats["messages_received"] += 1
            
        except Exception as e:
            self.logger.error(f"메시지 처리 중 오류 발생: {e}")
            self.stats["errors"] += 1
            
    def _process_messages(self) -> None:
        """메시지 처리 루프"""
        try:
            while self.is_running:
                if not self.message_queue.empty():
                    message = self.message_queue.get()
                    asyncio.run(self._handle_message(
                        message["connection_id"],
                        message["event"],
                        message["payload"]
                    ))
                    
                time.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"메시지 처리 중 오류 발생: {e}")
            self.stats["errors"] += 1
            
    async def _handle_message(self,
                            connection_id: str,
                            event: str,
                            payload: Any) -> None:
        """
        메시지 작업 처리
        
        Args:
            connection_id: 연결 ID
            event: 이벤트
            payload: 데이터
        """
        try:
            if connection_id in self.connections:
                websocket = self.connections[connection_id]
                message = json.dumps({
                    "event": event,
                    "payload": payload
                })
                await websocket.send(message)
                self.stats["messages_sent"] += 1
                
        except Exception as e:
            self.logger.error(f"메시지 작업 처리 중 오류 발생: {e}")
            self.stats["errors"] += 1
            
    def send_message(self,
                    connection_id: str,
                    event: str,
                    payload: Any) -> None:
        """
        메시지 전송 요청
        
        Args:
            connection_id: 연결 ID
            event: 이벤트
            payload: 데이터
        """
        try:
            self.message_queue.put({
                "connection_id": connection_id,
                "event": event,
                "payload": payload
            })
            
        except Exception as e:
            self.logger.error(f"메시지 전송 요청 중 오류 발생: {e}")
            self.stats["errors"] += 1
            
    def broadcast_message(self,
                         event: str,
                         payload: Any) -> None:
        """
        메시지 브로드캐스트 요청
        
        Args:
            event: 이벤트
            payload: 데이터
        """
        try:
            for connection_id in self.connections:
                self.send_message(connection_id, event, payload)
                
        except Exception as e:
            self.logger.error(f"메시지 브로드캐스트 요청 중 오류 발생: {e}")
            self.stats["errors"] += 1
            
    def register_handler(self,
                        event: str,
                        handler: Callable) -> None:
        """
        이벤트 핸들러 등록
        
        Args:
            event: 이벤트
            handler: 핸들러 함수
        """
        try:
            if event not in self.handlers:
                self.handlers[event] = []
                
            self.handlers[event].append(handler)
            
        except Exception as e:
            self.logger.error(f"핸들러 등록 중 오류 발생: {e}")
            self.stats["errors"] += 1
            
    def unregister_handler(self,
                          event: str,
                          handler: Callable) -> None:
        """
        이벤트 핸들러 해제
        
        Args:
            event: 이벤트
            handler: 핸들러 함수
        """
        try:
            if event in self.handlers:
                self.handlers[event].remove(handler)
                
        except Exception as e:
            self.logger.error(f"핸들러 해제 중 오류 발생: {e}")
            self.stats["errors"] += 1
            
    def get_connection_count(self) -> int:
        """
        연결 수 조회
        
        Returns:
            연결 수
        """
        return self.stats["connections"]
        
    def get_messages_sent_count(self) -> int:
        """
        전송된 메시지 수 조회
        
        Returns:
            전송된 메시지 수
        """
        return self.stats["messages_sent"]
        
    def get_messages_received_count(self) -> int:
        """
        수신된 메시지 수 조회
        
        Returns:
            수신된 메시지 수
        """
        return self.stats["messages_received"]
        
    def get_stats(self) -> Dict[str, int]:
        """
        WebSocket 통계 조회
        
        Returns:
            WebSocket 통계
        """
        return self.stats.copy()
        
    def reset_stats(self) -> None:
        """WebSocket 통계 초기화"""
        self.stats = {
            "connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0
        } 