"""
설정 관리 페이지
"""

import streamlit as st
import json
import os
from datetime import datetime

from src.utils.logger import setup_logger

# 로거 설정
logger = setup_logger()

def load_settings():
    """설정 파일 로드"""
    try:
        with open('config/settings.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"설정 파일 로드 오류: {str(e)}")
        return {}

def save_settings(settings):
    """설정 파일 저장"""
    try:
        with open('config/settings.json', 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
        logger.info("설정이 저장되었습니다.")
    except Exception as e:
        logger.error(f"설정 파일 저장 오류: {str(e)}")
        st.error("설정 저장 중 오류가 발생했습니다.")

def main():
    st.title("설정")
    
    # 설정 로드
    settings = load_settings()
    
    # 거래 설정
    st.header("거래 설정")
    
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input(
            "거래 페어",
            value=settings.get('symbol', 'BTC/USDT'),
            help="거래할 암호화폐 페어를 입력하세요."
        )
    with col2:
        timeframe = st.selectbox(
            "시간 프레임",
            options=['1m', '5m', '15m', '1h', '4h', '1d'],
            index=['1m', '5m', '15m', '1h', '4h', '1d'].index(settings.get('timeframe', '1h')),
            help="거래에 사용할 시간 프레임을 선택하세요."
        )
    
    # 전략 설정
    st.header("전략 설정")
    
    strategy = st.selectbox(
        "거래 전략",
        options=['Momentum', 'Mean Reversion', 'Breakout'],
        index=['Momentum', 'Mean Reversion', 'Breakout'].index(settings.get('strategy', 'Momentum')),
        help="사용할 거래 전략을 선택하세요."
    )
    
    if strategy == 'Momentum':
        col1, col2 = st.columns(2)
        with col1:
            rsi_period = st.number_input(
                "RSI 기간",
                min_value=1,
                max_value=100,
                value=settings.get('rsi_period', 14),
                help="RSI 계산에 사용할 기간을 입력하세요."
            )
        with col2:
            rsi_upper = st.number_input(
                "RSI 상단",
                min_value=0,
                max_value=100,
                value=settings.get('rsi_upper', 70),
                help="RSI 상단 임계값을 입력하세요."
            )
    
    # 리스크 관리 설정
    st.header("리스크 관리 설정")
    
    col1, col2 = st.columns(2)
    with col1:
        position_size = st.number_input(
            "포지션 크기 (%)",
            min_value=0.0,
            max_value=100.0,
            value=settings.get('position_size', 1.0),
            step=0.1,
            help="총 자본 대비 포지션 크기 비율을 입력하세요."
        )
    with col2:
        stop_loss = st.number_input(
            "손절 비율 (%)",
            min_value=0.0,
            max_value=100.0,
            value=settings.get('stop_loss', 2.0),
            step=0.1,
            help="손절할 가격 변동 비율을 입력하세요."
        )
    
    col1, col2 = st.columns(2)
    with col1:
        take_profit = st.number_input(
            "익절 비율 (%)",
            min_value=0.0,
            max_value=100.0,
            value=settings.get('take_profit', 4.0),
            step=0.1,
            help="익절할 가격 변동 비율을 입력하세요."
        )
    with col2:
        max_positions = st.number_input(
            "최대 포지션 수",
            min_value=1,
            max_value=10,
            value=settings.get('max_positions', 3),
            help="동시에 보유할 수 있는 최대 포지션 수를 입력하세요."
        )
    
    # 알림 설정
    st.header("알림 설정")
    
    col1, col2 = st.columns(2)
    with col1:
        telegram_enabled = st.checkbox(
            "텔레그램 알림",
            value=settings.get('telegram_enabled', False),
            help="텔레그램 알림을 활성화하세요."
        )
    with col2:
        email_enabled = st.checkbox(
            "이메일 알림",
            value=settings.get('email_enabled', False),
            help="이메일 알림을 활성화하세요."
        )
    
    if telegram_enabled:
        telegram_token = st.text_input(
            "텔레그램 봇 토큰",
            value=settings.get('telegram_token', ''),
            type='password',
            help="텔레그램 봇 토큰을 입력하세요."
        )
        telegram_chat_id = st.text_input(
            "텔레그램 채팅 ID",
            value=settings.get('telegram_chat_id', ''),
            help="텔레그램 채팅 ID를 입력하세요."
        )
    
    if email_enabled:
        email_address = st.text_input(
            "이메일 주소",
            value=settings.get('email_address', ''),
            help="알림을 받을 이메일 주소를 입력하세요."
        )
    
    # 설정 저장
    if st.button("설정 저장"):
        new_settings = {
            'symbol': symbol,
            'timeframe': timeframe,
            'strategy': strategy,
            'rsi_period': rsi_period,
            'rsi_upper': rsi_upper,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'max_positions': max_positions,
            'telegram_enabled': telegram_enabled,
            'email_enabled': email_enabled,
            'telegram_token': telegram_token if telegram_enabled else '',
            'telegram_chat_id': telegram_chat_id if telegram_enabled else '',
            'email_address': email_address if email_enabled else '',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        save_settings(new_settings)
        st.success("설정이 저장되었습니다.")
    
    # 설정 백업/복원
    st.header("설정 백업/복원")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("설정 백업"):
            backup_file = f"config/backup/settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            save_settings(settings)
            st.success(f"설정이 {backup_file}에 백업되었습니다.")
    
    with col2:
        backup_files = [f for f in os.listdir('config/backup') if f.endswith('.json')]
        if backup_files:
            selected_backup = st.selectbox(
                "복원할 백업 파일 선택",
                options=backup_files,
                help="복원할 백업 파일을 선택하세요."
            )
            if st.button("설정 복원"):
                try:
                    with open(f"config/backup/{selected_backup}", 'r', encoding='utf-8') as f:
                        backup_settings = json.load(f)
                    save_settings(backup_settings)
                    st.success("설정이 복원되었습니다.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"설정 복원 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main() 