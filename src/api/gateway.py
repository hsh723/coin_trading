import os
import json
import logging
import threading
import queue
import time
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
import aiohttp
import asyncio
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import jwt
from src.auth.auth_manager import AuthManager

class APIGateway:
    """API 게이트웨이 클래스"""
    
    def __init__(self,
                 auth_manager: AuthManager,
                 config_dir: str = "./config"):
        """
        API 게이트웨이 초기화
        
        Args:
            auth_manager: 인증 관리자
            config_dir: 설정 디렉토리
        """
        self.auth_manager = auth_manager
        self.config_dir = config_dir
        
        # 로거 설정
        self.logger = logging.getLogger("api_gateway")
        
        # 요청 큐
        self.request_queue = queue.Queue()
        
        # FastAPI 앱
        self.app = FastAPI()
        
        # CORS 설정
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 미들웨어 설정
        self.app.middleware("http")(self._auth_middleware)
        
        # 라우트 설정
        self._setup_routes()
        
        # 설정 로드
        self.config = self._load_config()
        
        # 서비스 상태
        self.services: Dict[str, Dict[str, Any]] = {}
        
        # 요청 처리 스레드
        self.request_thread = None
        self.is_running = False
        
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            config_path = os.path.join(self.config_dir, "gateway_config.json")
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"설정 파일 로드 중 오류 발생: {e}")
            return {
                "rate_limit": 100,
                "timeout": 30,
                "retry_count": 3,
                "circuit_breaker": {
                    "failure_threshold": 5,
                    "reset_timeout": 60
                }
            }
            
    def _setup_routes(self) -> None:
        """라우트 설정"""
        # 헬스 체크
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy"}
            
        # 서비스 등록
        @self.app.post("/services")
        async def register_service(service: Dict[str, Any]):
            return await self._register_service(service)
            
        # 서비스 해제
        @self.app.delete("/services/{service_name}")
        async def unregister_service(service_name: str):
            return await self._unregister_service(service_name)
            
        # 동적 라우트 생성
        @self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
        async def handle_request(request: Request, path: str):
            return await self._handle_request(request, path)
            
    async def _auth_middleware(self,
                             request: Request,
                             call_next: Callable) -> Response:
        """
        인증 미들웨어
        
        Args:
            request: 요청 객체
            call_next: 다음 미들웨어 호출 함수
            
        Returns:
            응답 객체
        """
        try:
            # 인증이 필요한 경로 확인
            if not self._is_auth_required(request.url.path):
                return await call_next(request)
                
            # 토큰 검증
            token = request.headers.get("Authorization")
            if not token:
                raise HTTPException(status_code=401, detail="인증 토큰이 필요합니다")
                
            if not token.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="잘못된 토큰 형식입니다")
                
            token = token[7:]
            payload = self.auth_manager.validate_token(token)
            
            if not payload:
                raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다")
                
            # 권한 확인
            if not self._check_permission(payload, request):
                raise HTTPException(status_code=403, detail="권한이 없습니다")
                
            # 요청 처리
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail}
            )
        except Exception as e:
            self.logger.error(f"인증 미들웨어 오류: {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": "서버 오류가 발생했습니다"}
            )
            
    def _is_auth_required(self, path: str) -> bool:
        """
        인증이 필요한 경로 확인
        
        Args:
            path: 요청 경로
            
        Returns:
            인증 필요 여부
        """
        # 인증이 필요하지 않은 경로 목록
        public_paths = ["/health", "/services"]
        
        return not any(path.startswith(p) for p in public_paths)
        
    def _check_permission(self,
                         payload: Dict[str, Any],
                         request: Request) -> bool:
        """
        권한 확인
        
        Args:
            payload: 토큰 페이로드
            request: 요청 객체
            
        Returns:
            권한 여부
        """
        try:
            # 서비스 권한 확인
            service_name = self._get_service_name(request.url.path)
            if not service_name:
                return False
                
            # 서비스 정보 조회
            service = self.services.get(service_name)
            if not service:
                return False
                
            # 필요한 권한 확인
            required_permission = self._get_required_permission(request.method)
            if not required_permission:
                return True
                
            # 사용자 권한 확인
            return self.auth_manager.check_permission(
                request.headers.get("Authorization")[7:],
                required_permission
            )
            
        except Exception as e:
            self.logger.error(f"권한 확인 중 오류 발생: {e}")
            return False
            
    def _get_service_name(self, path: str) -> Optional[str]:
        """
        서비스 이름 추출
        
        Args:
            path: 요청 경로
            
        Returns:
            서비스 이름
        """
        try:
            parts = path.strip("/").split("/")
            return parts[0] if parts else None
        except Exception:
            return None
            
    def _get_required_permission(self, method: str) -> Optional[str]:
        """
        필요한 권한 확인
        
        Args:
            method: HTTP 메서드
            
        Returns:
            필요한 권한
        """
        permissions = {
            "GET": "read",
            "POST": "write",
            "PUT": "write",
            "DELETE": "delete"
        }
        return permissions.get(method)
        
    async def _register_service(self,
                              service: Dict[str, Any]) -> Dict[str, Any]:
        """
        서비스 등록
        
        Args:
            service: 서비스 정보
            
        Returns:
            등록된 서비스 정보
        """
        try:
            service_name = service.get("name")
            if not service_name:
                raise HTTPException(status_code=400, detail="서비스 이름이 필요합니다")
                
            if service_name in self.services:
                raise HTTPException(status_code=400, detail="이미 등록된 서비스입니다")
                
            self.services[service_name] = {
                "url": service.get("url"),
                "health_check": service.get("health_check", "/health"),
                "timeout": service.get("timeout", self.config["timeout"]),
                "retry_count": service.get("retry_count", self.config["retry_count"]),
                "circuit_breaker": {
                    "failure_count": 0,
                    "last_failure": None,
                    "is_open": False
                }
            }
            
            return {
                "name": service_name,
                "status": "registered"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"서비스 등록 중 오류 발생: {e}")
            raise HTTPException(status_code=500, detail="서비스 등록에 실패했습니다")
            
    async def _unregister_service(self,
                                service_name: str) -> Dict[str, Any]:
        """
        서비스 해제
        
        Args:
            service_name: 서비스 이름
            
        Returns:
            해제된 서비스 정보
        """
        try:
            if service_name not in self.services:
                raise HTTPException(status_code=404, detail="등록되지 않은 서비스입니다")
                
            del self.services[service_name]
            
            return {
                "name": service_name,
                "status": "unregistered"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"서비스 해제 중 오류 발생: {e}")
            raise HTTPException(status_code=500, detail="서비스 해제에 실패했습니다")
            
    async def _handle_request(self,
                            request: Request,
                            path: str) -> Response:
        """
        요청 처리
        
        Args:
            request: 요청 객체
            path: 요청 경로
            
        Returns:
            응답 객체
        """
        try:
            # 서비스 확인
            service_name = self._get_service_name(path)
            if not service_name:
                raise HTTPException(status_code=404, detail="서비스가 존재하지 않습니다")
                
            service = self.services.get(service_name)
            if not service:
                raise HTTPException(status_code=404, detail="서비스가 존재하지 않습니다")
                
            # 서킷 브레이커 확인
            if service["circuit_breaker"]["is_open"]:
                if self._should_reset_circuit_breaker(service):
                    service["circuit_breaker"]["is_open"] = False
                    service["circuit_breaker"]["failure_count"] = 0
                else:
                    raise HTTPException(status_code=503, detail="서비스가 일시적으로 사용할 수 없습니다")
                    
            # 요청 전달
            async with aiohttp.ClientSession() as session:
                for attempt in range(service["retry_count"]):
                    try:
                        response = await session.request(
                            method=request.method,
                            url=f"{service['url']}{path}",
                            headers=dict(request.headers),
                            params=dict(request.query_params),
                            data=await request.body(),
                            timeout=service["timeout"]
                        )
                        
                        # 성공 시 서킷 브레이커 리셋
                        service["circuit_breaker"]["failure_count"] = 0
                        
                        return Response(
                            content=await response.read(),
                            status_code=response.status,
                            headers=dict(response.headers)
                        )
                        
                    except Exception as e:
                        self.logger.error(f"요청 처리 중 오류 발생: {e}")
                        
                        # 실패 시 서킷 브레이커 업데이트
                        service["circuit_breaker"]["failure_count"] += 1
                        service["circuit_breaker"]["last_failure"] = datetime.now()
                        
                        if service["circuit_breaker"]["failure_count"] >= self.config["circuit_breaker"]["failure_threshold"]:
                            service["circuit_breaker"]["is_open"] = True
                            
                        if attempt == service["retry_count"] - 1:
                            raise HTTPException(status_code=500, detail="서비스 요청에 실패했습니다")
                            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"요청 처리 중 오류 발생: {e}")
            raise HTTPException(status_code=500, detail="서버 오류가 발생했습니다")
            
    def _should_reset_circuit_breaker(self,
                                    service: Dict[str, Any]) -> bool:
        """
        서킷 브레이커 리셋 여부 확인
        
        Args:
            service: 서비스 정보
            
        Returns:
            리셋 여부
        """
        if not service["circuit_breaker"]["last_failure"]:
            return False
            
        last_failure = datetime.strptime(
            service["circuit_breaker"]["last_failure"],
            "%Y-%m-%d %H:%M:%S"
        )
        
        return (datetime.now() - last_failure).total_seconds() >= self.config["circuit_breaker"]["reset_timeout"]
        
    def start(self) -> None:
        """API 게이트웨이 시작"""
        try:
            self.is_running = True
            self.request_thread = threading.Thread(target=self._process_requests)
            self.request_thread.start()
            
        except Exception as e:
            self.logger.error(f"API 게이트웨이 시작 중 오류 발생: {e}")
            raise
            
    def stop(self) -> None:
        """API 게이트웨이 중지"""
        try:
            self.is_running = False
            if self.request_thread:
                self.request_thread.join()
                
        except Exception as e:
            self.logger.error(f"API 게이트웨이 중지 중 오류 발생: {e}")
            raise
            
    def _process_requests(self) -> None:
        """요청 처리 루프"""
        try:
            while self.is_running:
                if not self.request_queue.empty():
                    request = self.request_queue.get()
                    asyncio.run(self._handle_request(*request))
                    
                time.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"요청 처리 중 오류 발생: {e}") 