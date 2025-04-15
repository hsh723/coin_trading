import os
import json
import logging
import threading
import queue
import time
import pickle
import pika
import redis
import zmq
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
from pathlib import Path

class MessageQueue:
    """메시지 큐 클래스"""
    
    def __init__(self,
                 config_dir: str = "./config",
                 data_dir: str = "./data"):
        """
        메시지 큐 초기화
        
        Args:
            config_dir: 설정 디렉토리
            data_dir: 데이터 디렉토리
        """
        self.config_dir = config_dir
        self.data_dir = data_dir
        
        # 로거 설정
        self.logger = logging.getLogger("mq")
        
        # 메시지 큐
        self.message_queue = queue.Queue()
        
        # 디렉토리 생성
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # 설정 로드
        self.config = self._load_config()
        
        # 메시지 큐 클라이언트
        self.clients: Dict[str, Any] = {}
        
        # 메시지 큐 관리자
        self.is_running = False
        
        # 구독 관리
        self.subscriptions: Dict[str, Dict[str, Any]] = {}
        
        # 통계
        self.stats = {
            "published": 0,
            "consumed": 0,
            "subscribers": 0,
            "errors": 0
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            config_path = os.path.join(self.config_dir, "mq_config.json")
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"설정 파일 로드 중 오류 발생: {e}")
            return {
                "default": {
                    "type": "rabbitmq",
                    "host": "localhost",
                    "port": 5672,
                    "username": "guest",
                    "password": "guest"
                }
            }
            
    def start(self) -> None:
        """메시지 큐 시작"""
        try:
            self.is_running = True
            
            # 메시지 처리 시작
            threading.Thread(target=self._process_messages, daemon=True).start()
            
            self.logger.info("메시지 큐가 시작되었습니다")
            
        except Exception as e:
            self.logger.error(f"메시지 큐 시작 중 오류 발생: {e}")
            raise
            
    def stop(self) -> None:
        """메시지 큐 중지"""
        try:
            self.is_running = False
            
            # 클라이언트 종료
            for client in self.clients.values():
                self._close_client(client)
                
            self.logger.info("메시지 큐가 중지되었습니다")
            
        except Exception as e:
            self.logger.error(f"메시지 큐 중지 중 오류 발생: {e}")
            raise
            
    def _create_client(self,
                      mq_type: str,
                      config: Dict[str, Any]) -> Any:
        """
        메시지 큐 클라이언트 생성
        
        Args:
            mq_type: 메시지 큐 타입
            config: 클라이언트 설정
            
        Returns:
            메시지 큐 클라이언트
        """
        try:
            if mq_type == "rabbitmq":
                # RabbitMQ 클라이언트
                credentials = pika.PlainCredentials(
                    config["username"],
                    config["password"]
                )
                parameters = pika.ConnectionParameters(
                    host=config["host"],
                    port=config["port"],
                    credentials=credentials
                )
                return pika.BlockingConnection(parameters)
                
            elif mq_type == "redis":
                # Redis 클라이언트
                return redis.Redis(
                    host=config["host"],
                    port=config["port"],
                    password=config["password"],
                    decode_responses=False
                )
                
            elif mq_type == "zeromq":
                # ZeroMQ 클라이언트
                context = zmq.Context()
                return context.socket(zmq.PUB)
                
            else:
                raise ValueError(f"지원하지 않는 메시지 큐 타입: {mq_type}")
                
        except Exception as e:
            self.logger.error(f"메시지 큐 클라이언트 생성 중 오류 발생: {e}")
            raise
            
    def _close_client(self, client: Any) -> None:
        """
        메시지 큐 클라이언트 종료
        
        Args:
            client: 메시지 큐 클라이언트
        """
        try:
            if isinstance(client, pika.BlockingConnection):
                client.close()
            elif isinstance(client, redis.Redis):
                client.close()
            elif isinstance(client, zmq.Socket):
                client.close()
                
        except Exception as e:
            self.logger.error(f"메시지 큐 클라이언트 종료 중 오류 발생: {e}")
            
    def get_client(self,
                   mq_type: str,
                   config: Dict[str, Any]) -> Any:
        """
        메시지 큐 클라이언트 가져오기
        
        Args:
            mq_type: 메시지 큐 타입
            config: 클라이언트 설정
            
        Returns:
            메시지 큐 클라이언트
        """
        try:
            # 클라이언트 키 생성
            client_key = f"{mq_type}_{json.dumps(config, sort_keys=True)}"
            
            # 클라이언트 확인
            if client_key not in self.clients:
                self.clients[client_key] = self._create_client(mq_type, config)
                
            return self.clients[client_key]
            
        except Exception as e:
            self.logger.error(f"메시지 큐 클라이언트 가져오기 중 오류 발생: {e}")
            raise
            
    def _process_messages(self) -> None:
        """메시지 처리 루프"""
        try:
            while self.is_running:
                if not self.message_queue.empty():
                    message = self.message_queue.get()
                    self._handle_message(
                        message["mq_type"],
                        message["config"],
                        message["operation"],
                        message["queue"],
                        message.get("message"),
                        message.get("callback")
                    )
                    
                time.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"메시지 처리 중 오류 발생: {e}")
            self.stats["errors"] += 1
            
    def _handle_message(self,
                       mq_type: str,
                       config: Dict[str, Any],
                       operation: str,
                       queue: str,
                       message: Optional[Any] = None,
                       callback: Optional[Callable] = None) -> None:
        """
        메시지 작업 처리
        
        Args:
            mq_type: 메시지 큐 타입
            config: 클라이언트 설정
            operation: 작업 타입
            queue: 큐 이름
            message: 메시지
            callback: 콜백 함수
        """
        try:
            if operation == "publish":
                # 메시지 발행
                self._publish_message(mq_type, config, queue, message)
                
            elif operation == "subscribe":
                # 메시지 구독
                self._subscribe_message(mq_type, config, queue, callback)
                
            elif operation == "unsubscribe":
                # 메시지 구독 해제
                self._unsubscribe_message(mq_type, config, queue)
                
        except Exception as e:
            self.logger.error(f"메시지 작업 처리 중 오류 발생: {e}")
            self.stats["errors"] += 1
            raise
            
    def _publish_message(self,
                        mq_type: str,
                        config: Dict[str, Any],
                        queue: str,
                        message: Any) -> None:
        """
        메시지 발행
        
        Args:
            mq_type: 메시지 큐 타입
            config: 클라이언트 설정
            queue: 큐 이름
            message: 메시지
        """
        try:
            if mq_type == "rabbitmq":
                # RabbitMQ 메시지 발행
                client = self.get_client(mq_type, config)
                channel = client.channel()
                channel.queue_declare(queue=queue)
                channel.basic_publish(
                    exchange="",
                    routing_key=queue,
                    body=pickle.dumps(message)
                )
                
            elif mq_type == "redis":
                # Redis 메시지 발행
                client = self.get_client(mq_type, config)
                client.publish(queue, pickle.dumps(message))
                
            elif mq_type == "zeromq":
                # ZeroMQ 메시지 발행
                client = self.get_client(mq_type, config)
                client.bind(f"tcp://*:{config['port']}")
                client.send_multipart([queue.encode(), pickle.dumps(message)])
                
            self.stats["published"] += 1
            
        except Exception as e:
            self.logger.error(f"메시지 발행 중 오류 발생: {e}")
            self.stats["errors"] += 1
            raise
            
    def _subscribe_message(self,
                         mq_type: str,
                         config: Dict[str, Any],
                         queue: str,
                         callback: Callable) -> None:
        """
        메시지 구독
        
        Args:
            mq_type: 메시지 큐 타입
            config: 클라이언트 설정
            queue: 큐 이름
            callback: 콜백 함수
        """
        try:
            if mq_type == "rabbitmq":
                # RabbitMQ 메시지 구독
                client = self.get_client(mq_type, config)
                channel = client.channel()
                channel.queue_declare(queue=queue)
                
                def on_message(ch, method, properties, body):
                    try:
                        message = pickle.loads(body)
                        callback(message)
                        self.stats["consumed"] += 1
                    except Exception as e:
                        self.logger.error(f"메시지 처리 중 오류 발생: {e}")
                        self.stats["errors"] += 1
                        
                channel.basic_consume(
                    queue=queue,
                    on_message_callback=on_message,
                    auto_ack=True
                )
                
                # 구독 관리
                self.subscriptions[queue] = {
                    "mq_type": mq_type,
                    "config": config,
                    "channel": channel
                }
                
                # 구독 시작
                threading.Thread(
                    target=channel.start_consuming,
                    daemon=True
                ).start()
                
            elif mq_type == "redis":
                # Redis 메시지 구독
                client = self.get_client(mq_type, config)
                pubsub = client.pubsub()
                pubsub.subscribe(queue)
                
                def on_message(message):
                    try:
                        if message["type"] == "message":
                            data = pickle.loads(message["data"])
                            callback(data)
                            self.stats["consumed"] += 1
                    except Exception as e:
                        self.logger.error(f"메시지 처리 중 오류 발생: {e}")
                        self.stats["errors"] += 1
                        
                # 구독 관리
                self.subscriptions[queue] = {
                    "mq_type": mq_type,
                    "config": config,
                    "pubsub": pubsub
                }
                
                # 구독 시작
                threading.Thread(
                    target=pubsub.run_in_thread,
                    daemon=True
                ).start()
                
            elif mq_type == "zeromq":
                # ZeroMQ 메시지 구독
                context = zmq.Context()
                socket = context.socket(zmq.SUB)
                socket.connect(f"tcp://{config['host']}:{config['port']}")
                socket.setsockopt_string(zmq.SUBSCRIBE, queue)
                
                def on_message():
                    try:
                        while True:
                            topic, message = socket.recv_multipart()
                            if topic.decode() == queue:
                                data = pickle.loads(message)
                                callback(data)
                                self.stats["consumed"] += 1
                    except Exception as e:
                        self.logger.error(f"메시지 처리 중 오류 발생: {e}")
                        self.stats["errors"] += 1
                        
                # 구독 관리
                self.subscriptions[queue] = {
                    "mq_type": mq_type,
                    "config": config,
                    "socket": socket
                }
                
                # 구독 시작
                threading.Thread(target=on_message, daemon=True).start()
                
            self.stats["subscribers"] += 1
            
        except Exception as e:
            self.logger.error(f"메시지 구독 중 오류 발생: {e}")
            self.stats["errors"] += 1
            raise
            
    def _unsubscribe_message(self,
                           mq_type: str,
                           config: Dict[str, Any],
                           queue: str) -> None:
        """
        메시지 구독 해제
        
        Args:
            mq_type: 메시지 큐 타입
            config: 클라이언트 설정
            queue: 큐 이름
        """
        try:
            if queue in self.subscriptions:
                subscription = self.subscriptions[queue]
                
                if mq_type == "rabbitmq":
                    # RabbitMQ 구독 해제
                    subscription["channel"].stop_consuming()
                    
                elif mq_type == "redis":
                    # Redis 구독 해제
                    subscription["pubsub"].unsubscribe(queue)
                    
                elif mq_type == "zeromq":
                    # ZeroMQ 구독 해제
                    subscription["socket"].close()
                    
                del self.subscriptions[queue]
                self.stats["subscribers"] -= 1
                
        except Exception as e:
            self.logger.error(f"메시지 구독 해제 중 오류 발생: {e}")
            self.stats["errors"] += 1
            raise
            
    def publish(self,
                mq_type: str,
                config: Dict[str, Any],
                queue: str,
                message: Any) -> None:
        """
        메시지 발행 요청
        
        Args:
            mq_type: 메시지 큐 타입
            config: 클라이언트 설정
            queue: 큐 이름
            message: 메시지
        """
        try:
            self.message_queue.put({
                "mq_type": mq_type,
                "config": config,
                "operation": "publish",
                "queue": queue,
                "message": message
            })
            
        except Exception as e:
            self.logger.error(f"메시지 발행 요청 중 오류 발생: {e}")
            self.stats["errors"] += 1
            
    def subscribe(self,
                 mq_type: str,
                 config: Dict[str, Any],
                 queue: str,
                 callback: Callable) -> None:
        """
        메시지 구독 요청
        
        Args:
            mq_type: 메시지 큐 타입
            config: 클라이언트 설정
            queue: 큐 이름
            callback: 콜백 함수
        """
        try:
            self.message_queue.put({
                "mq_type": mq_type,
                "config": config,
                "operation": "subscribe",
                "queue": queue,
                "callback": callback
            })
            
        except Exception as e:
            self.logger.error(f"메시지 구독 요청 중 오류 발생: {e}")
            self.stats["errors"] += 1
            
    def unsubscribe(self,
                   mq_type: str,
                   config: Dict[str, Any],
                   queue: str) -> None:
        """
        메시지 구독 해제 요청
        
        Args:
            mq_type: 메시지 큐 타입
            config: 클라이언트 설정
            queue: 큐 이름
        """
        try:
            self.message_queue.put({
                "mq_type": mq_type,
                "config": config,
                "operation": "unsubscribe",
                "queue": queue
            })
            
        except Exception as e:
            self.logger.error(f"메시지 구독 해제 요청 중 오류 발생: {e}")
            self.stats["errors"] += 1
            
    def get_published_count(self) -> int:
        """
        발행된 메시지 수 조회
        
        Returns:
            발행된 메시지 수
        """
        return self.stats["published"]
        
    def get_consumed_count(self) -> int:
        """
        소비된 메시지 수 조회
        
        Returns:
            소비된 메시지 수
        """
        return self.stats["consumed"]
        
    def get_subscriber_count(self) -> int:
        """
        구독자 수 조회
        
        Returns:
            구독자 수
        """
        return self.stats["subscribers"]
        
    def get_stats(self) -> Dict[str, int]:
        """
        메시지 큐 통계 조회
        
        Returns:
            메시지 큐 통계
        """
        return self.stats.copy()
        
    def reset_stats(self) -> None:
        """메시지 큐 통계 초기화"""
        self.stats = {
            "published": 0,
            "consumed": 0,
            "subscribers": 0,
            "errors": 0
        } 