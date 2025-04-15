import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import psutil
import platform
import time
import threading
import queue
import sqlite3
from pathlib import Path
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import asyncio

class SystemMonitor:
    """시스템 모니터링 클래스"""
    
    def __init__(self,
                 config_path: str = "./config/monitor_config.json",
                 data_dir: str = "./data",
                 log_dir: str = "./logs"):
        """
        시스템 모니터 초기화
        
        Args:
            config_path: 설정 파일 경로
            data_dir: 데이터 디렉토리
            log_dir: 로그 디렉토리
        """
        self.config_path = config_path
        self.data_dir = data_dir
        self.log_dir = log_dir
        
        # 로거 설정
        self.logger = logging.getLogger("system_monitor")
        
        # 설정 로드
        self.config = self._load_config()
        
        # 알림 큐
        self.notification_queue = queue.Queue()
        
        # 데이터베이스 연결
        self.db_path = os.path.join(data_dir, "monitoring.db")
        self._init_database()
        
        # 디렉토리 생성
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # 모니터링 스레드
        self.monitoring_thread = None
        self.is_running = False
        
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"설정 파일 로드 중 오류 발생: {e}")
            return {}
            
    def _init_database(self) -> None:
        """데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 시스템 메트릭 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    timestamp DATETIME,
                    cpu_usage REAL,
                    memory_usage REAL,
                    disk_usage REAL,
                    network_traffic REAL,
                    PRIMARY KEY (timestamp)
                )
            ''')
            
            # 성능 메트릭 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    timestamp DATETIME,
                    processing_speed REAL,
                    latency REAL,
                    error_rate REAL,
                    PRIMARY KEY (timestamp)
                )
            ''')
            
            # 알림 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notifications (
                    timestamp DATETIME,
                    level TEXT,
                    message TEXT,
                    status TEXT,
                    PRIMARY KEY (timestamp, level, message)
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"데이터베이스 초기화 중 오류 발생: {e}")
            raise
            
    def start(self) -> None:
        """모니터링 시작"""
        try:
            self.is_running = True
            self.monitoring_thread = threading.Thread(target=self._monitor)
            self.monitoring_thread.start()
            
        except Exception as e:
            self.logger.error(f"모니터링 시작 중 오류 발생: {e}")
            raise
            
    def stop(self) -> None:
        """모니터링 중지"""
        try:
            self.is_running = False
            if self.monitoring_thread:
                self.monitoring_thread.join()
                
        except Exception as e:
            self.logger.error(f"모니터링 중지 중 오류 발생: {e}")
            raise
            
    def _monitor(self) -> None:
        """모니터링 루프"""
        try:
            while self.is_running:
                # 시스템 메트릭 수집
                system_metrics = self._collect_system_metrics()
                self._save_system_metrics(system_metrics)
                
                # 성능 메트릭 수집
                performance_metrics = self._collect_performance_metrics()
                self._save_performance_metrics(performance_metrics)
                
                # 임계값 체크
                self._check_thresholds(system_metrics, performance_metrics)
                
                # 알림 처리
                self._process_notifications()
                
                # 대기
                time.sleep(self.config.get("monitoring_interval", 60))
                
        except Exception as e:
            self.logger.error(f"모니터링 중 오류 발생: {e}")
            
    def _collect_system_metrics(self) -> Dict[str, float]:
        """
        시스템 메트릭 수집
        
        Returns:
            시스템 메트릭
        """
        try:
            metrics = {}
            
            # CPU 사용량
            metrics["cpu_usage"] = psutil.cpu_percent(interval=1)
            
            # 메모리 사용량
            memory = psutil.virtual_memory()
            metrics["memory_usage"] = memory.percent
            
            # 디스크 사용량
            disk = psutil.disk_usage('/')
            metrics["disk_usage"] = disk.percent
            
            # 네트워크 트래픽
            net_io = psutil.net_io_counters()
            metrics["network_traffic"] = net_io.bytes_sent + net_io.bytes_recv
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"시스템 메트릭 수집 중 오류 발생: {e}")
            return {}
            
    def _collect_performance_metrics(self) -> Dict[str, float]:
        """
        성능 메트릭 수집
        
        Returns:
            성능 메트릭
        """
        try:
            metrics = {}
            
            # 처리 속도
            metrics["processing_speed"] = self._calculate_processing_speed()
            
            # 대기 시간
            metrics["latency"] = self._calculate_latency()
            
            # 오류율
            metrics["error_rate"] = self._calculate_error_rate()
        
        return metrics

        except Exception as e:
            self.logger.error(f"성능 메트릭 수집 중 오류 발생: {e}")
            return {}
            
    def _save_system_metrics(self, metrics: Dict[str, float]) -> None:
        """
        시스템 메트릭 저장
        
        Args:
            metrics: 시스템 메트릭
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_metrics
                (timestamp, cpu_usage, memory_usage, disk_usage, network_traffic)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                metrics.get("cpu_usage", 0),
                metrics.get("memory_usage", 0),
                metrics.get("disk_usage", 0),
                metrics.get("network_traffic", 0)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"시스템 메트릭 저장 중 오류 발생: {e}")
            
    def _save_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """
        성능 메트릭 저장
        
        Args:
            metrics: 성능 메트릭
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_metrics
                (timestamp, processing_speed, latency, error_rate)
                VALUES (?, ?, ?, ?)
            ''', (
                datetime.now(),
                metrics.get("processing_speed", 0),
                metrics.get("latency", 0),
                metrics.get("error_rate", 0)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"성능 메트릭 저장 중 오류 발생: {e}")
            
    def _check_thresholds(self,
                         system_metrics: Dict[str, float],
                         performance_metrics: Dict[str, float]) -> None:
        """
        임계값 체크
        
        Args:
            system_metrics: 시스템 메트릭
            performance_metrics: 성능 메트릭
        """
        try:
            # CPU 사용량 체크
            if system_metrics.get("cpu_usage", 0) > self.config.get("cpu_threshold", 80):
                self._add_notification(
                    "WARNING",
                    f"CPU 사용량이 높습니다: {system_metrics['cpu_usage']}%"
                )
                
            # 메모리 사용량 체크
            if system_metrics.get("memory_usage", 0) > self.config.get("memory_threshold", 80):
                self._add_notification(
                    "WARNING",
                    f"메모리 사용량이 높습니다: {system_metrics['memory_usage']}%"
                )
                
            # 디스크 사용량 체크
            if system_metrics.get("disk_usage", 0) > self.config.get("disk_threshold", 80):
                self._add_notification(
                    "WARNING",
                    f"디스크 사용량이 높습니다: {system_metrics['disk_usage']}%"
                )
                
            # 처리 속도 체크
            if performance_metrics.get("processing_speed", 0) < self.config.get("speed_threshold", 100):
                self._add_notification(
                    "WARNING",
                    f"처리 속도가 느립니다: {performance_metrics['processing_speed']} ops/s"
                )
                
            # 대기 시간 체크
            if performance_metrics.get("latency", 0) > self.config.get("latency_threshold", 1000):
                self._add_notification(
                    "WARNING",
                    f"대기 시간이 길어집니다: {performance_metrics['latency']} ms"
                )
                
            # 오류율 체크
            if performance_metrics.get("error_rate", 0) > self.config.get("error_threshold", 1):
                self._add_notification(
                    "ERROR",
                    f"오류율이 높습니다: {performance_metrics['error_rate']}%"
                )
                
        except Exception as e:
            self.logger.error(f"임계값 체크 중 오류 발생: {e}")
            
    def _add_notification(self, level: str, message: str) -> None:
        """
        알림 추가
        
        Args:
            level: 알림 레벨
            message: 알림 메시지
        """
        try:
            # 알림 큐에 추가
            self.notification_queue.put({
                "timestamp": datetime.now(),
                "level": level,
                "message": message,
                "status": "PENDING"
            })
            
            # 데이터베이스에 저장
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO notifications
                (timestamp, level, message, status)
                VALUES (?, ?, ?, ?)
            ''', (datetime.now(), level, message, "PENDING"))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"알림 추가 중 오류 발생: {e}")
            
    def _process_notifications(self) -> None:
        """알림 처리"""
        try:
            while not self.notification_queue.empty():
                notification = self.notification_queue.get()
                
                # 알림 전송
                self._send_notification(notification)
                
                # 상태 업데이트
                self._update_notification_status(notification)
                
        except Exception as e:
            self.logger.error(f"알림 처리 중 오류 발생: {e}")
            
    def _send_notification(self, notification: Dict[str, Any]) -> None:
        """
        알림 전송
        
        Args:
            notification: 알림 정보
        """
        try:
            # 이메일 전송
            if "email" in self.config.get("notification_channels", []):
                self._send_email_notification(notification)
                
            # 슬랙 전송
            if "slack" in self.config.get("notification_channels", []):
                self._send_slack_notification(notification)
                
            # 텔레그램 전송
            if "telegram" in self.config.get("notification_channels", []):
                self._send_telegram_notification(notification)
                
        except Exception as e:
            self.logger.error(f"알림 전송 중 오류 발생: {e}")
            
    def _send_email_notification(self, notification: Dict[str, Any]) -> None:
        """
        이메일 알림 전송
        
        Args:
            notification: 알림 정보
        """
        try:
            # 이메일 설정
            smtp_server = self.config.get("smtp_server")
            smtp_port = self.config.get("smtp_port")
            sender_email = self.config.get("sender_email")
            sender_password = self.config.get("sender_password")
            receiver_email = self.config.get("receiver_email")
            
            # 이메일 메시지 생성
            msg = MIMEMultipart()
            msg["From"] = sender_email
            msg["To"] = receiver_email
            msg["Subject"] = f"[{notification['level']}] 시스템 알림"
            
            body = f"""
            시간: {notification['timestamp']}
            레벨: {notification['level']}
            메시지: {notification['message']}
            """
            
            msg.attach(MIMEText(body, "plain"))
            
            # 이메일 전송
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
                
        except Exception as e:
            self.logger.error(f"이메일 알림 전송 중 오류 발생: {e}")
            
    def _send_slack_notification(self, notification: Dict[str, Any]) -> None:
        """
        슬랙 알림 전송
        
        Args:
            notification: 알림 정보
        """
        try:
            # 슬랙 설정
            webhook_url = self.config.get("slack_webhook_url")
            
            # 메시지 생성
            message = {
                "text": f"""
                *[{notification['level']}] 시스템 알림*
                시간: {notification['timestamp']}
                메시지: {notification['message']}
                """
            }
            
            # 슬랙 전송
            requests.post(webhook_url, json=message)
            
        except Exception as e:
            self.logger.error(f"슬랙 알림 전송 중 오류 발생: {e}")
            
    def _send_telegram_notification(self, notification: Dict[str, Any]) -> None:
        """
        텔레그램 알림 전송
        
        Args:
            notification: 알림 정보
        """
        try:
            # 텔레그램 설정
            bot_token = self.config.get("telegram_bot_token")
            chat_id = self.config.get("telegram_chat_id")
            
            # 메시지 생성
            message = f"""
            [{notification['level']}] 시스템 알림
            시간: {notification['timestamp']}
            메시지: {notification['message']}
            """
            
            # 텔레그램 전송
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": message
            }
            
            requests.post(url, json=data)
            
        except Exception as e:
            self.logger.error(f"텔레그램 알림 전송 중 오류 발생: {e}")
            
    def _update_notification_status(self, notification: Dict[str, Any]) -> None:
        """
        알림 상태 업데이트
        
        Args:
            notification: 알림 정보
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE notifications
                SET status = ?
                WHERE timestamp = ? AND level = ? AND message = ?
            ''', ("SENT", notification["timestamp"], notification["level"], notification["message"]))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"알림 상태 업데이트 중 오류 발생: {e}")
            
    def _calculate_processing_speed(self) -> float:
        """
        처리 속도 계산
        
        Returns:
            처리 속도 (ops/s)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 최근 1분간 처리된 데이터 수 조회
            cursor.execute('''
                SELECT COUNT(*) FROM system_metrics
                WHERE timestamp >= ?
            ''', (datetime.now() - timedelta(minutes=1),))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] / 60 if result[0] else 0
            
        except Exception as e:
            self.logger.error(f"처리 속도 계산 중 오류 발생: {e}")
            return 0
            
    def _calculate_latency(self) -> float:
        """
        대기 시간 계산
        
        Returns:
            대기 시간 (ms)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 최근 1분간 평균 대기 시간 조회
            cursor.execute('''
                SELECT AVG(latency) FROM performance_metrics
                WHERE timestamp >= ?
            ''', (datetime.now() - timedelta(minutes=1),))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result[0] else 0
            
        except Exception as e:
            self.logger.error(f"대기 시간 계산 중 오류 발생: {e}")
            return 0
            
    def _calculate_error_rate(self) -> float:
        """
        오류율 계산
        
        Returns:
            오류율 (%)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 최근 1분간 오류 수 조회
            cursor.execute('''
                SELECT COUNT(*) FROM notifications
                WHERE level = 'ERROR' AND timestamp >= ?
            ''', (datetime.now() - timedelta(minutes=1),))
            
            error_count = cursor.fetchone()[0]
            
            # 총 처리 수 조회
            cursor.execute('''
                SELECT COUNT(*) FROM system_metrics
                WHERE timestamp >= ?
            ''', (datetime.now() - timedelta(minutes=1),))
            
            total_count = cursor.fetchone()[0]
            conn.close()
            
            return (error_count / total_count * 100) if total_count > 0 else 0
            
        except Exception as e:
            self.logger.error(f"오류율 계산 중 오류 발생: {e}")
            return 0
            
    def get_metrics(self,
                   start_time: datetime,
                   end_time: datetime) -> Dict[str, pd.DataFrame]:
        """
        메트릭 조회
        
        Args:
            start_time: 시작 시간
            end_time: 종료 시간
            
        Returns:
            메트릭 데이터프레임
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 시스템 메트릭 조회
            system_metrics = pd.read_sql_query('''
                SELECT * FROM system_metrics
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            ''', conn, params=(start_time, end_time))
            
            # 성능 메트릭 조회
            performance_metrics = pd.read_sql_query('''
                SELECT * FROM performance_metrics
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            ''', conn, params=(start_time, end_time))
            
            conn.close()
            
            return {
                "system": system_metrics,
                "performance": performance_metrics
            }
            
        except Exception as e:
            self.logger.error(f"메트릭 조회 중 오류 발생: {e}")
            raise
            
    def get_notifications(self,
                         start_time: datetime,
                         end_time: datetime) -> pd.DataFrame:
        """
        알림 조회
        
        Args:
            start_time: 시작 시간
            end_time: 종료 시간
            
        Returns:
            알림 데이터프레임
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            notifications = pd.read_sql_query('''
                SELECT * FROM notifications
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
            ''', conn, params=(start_time, end_time))
            
            conn.close()
            
            return notifications
            
        except Exception as e:
            self.logger.error(f"알림 조회 중 오류 발생: {e}")
            raise
