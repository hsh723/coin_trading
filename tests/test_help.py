import pytest
from src.help.help_manager import HelpManager
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def help_manager():
    config_dir = "./config"
    help_dir = "./help"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(help_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "help": {
                "language": "ko",
                "topics": {
                    "trading": {
                        "title": "트레이딩 가이드",
                        "description": "트레이딩 기본 개념과 전략",
                        "sections": ["기본", "전략", "리스크 관리"]
                    },
                    "api": {
                        "title": "API 가이드",
                        "description": "API 사용 방법과 예제",
                        "sections": ["인증", "요청", "응답"]
                    },
                    "analysis": {
                        "title": "분석 가이드",
                        "description": "시장 분석과 지표",
                        "sections": ["기술적 분석", "기본적 분석", "감성 분석"]
                    }
                }
            }
        }
    }
    with open(os.path.join(config_dir, "help.json"), "w") as f:
        json.dump(config, f)
    
    # 도움말 파일 생성
    help_files = {
        "trading_basic.md": "# 트레이딩 기본\n\n기본적인 트레이딩 개념 설명",
        "trading_strategy.md": "# 트레이딩 전략\n\n다양한 트레이딩 전략 설명",
        "trading_risk.md": "# 리스크 관리\n\n리스크 관리 방법 설명",
        "api_auth.md": "# API 인증\n\nAPI 인증 방법 설명",
        "api_request.md": "# API 요청\n\nAPI 요청 방법 설명",
        "api_response.md": "# API 응답\n\nAPI 응답 처리 방법 설명",
        "analysis_technical.md": "# 기술적 분석\n\n기술적 분석 방법 설명",
        "analysis_fundamental.md": "# 기본적 분석\n\n기본적 분석 방법 설명",
        "analysis_sentiment.md": "# 감성 분석\n\n감성 분석 방법 설명"
    }
    
    for filename, content in help_files.items():
        with open(os.path.join(help_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)
    
    return HelpManager(config_dir=config_dir, help_dir=help_dir)

def test_help_manager_initialization(help_manager):
    assert help_manager is not None
    assert help_manager.config_dir == "./config"
    assert help_manager.help_dir == "./help"

def test_help_topics(help_manager):
    # 도움말 주제 테스트
    topics = help_manager.get_topics()
    assert topics is not None
    assert "trading" in topics
    assert "api" in topics
    assert "analysis" in topics
    
    # 주제 상세 정보
    trading_info = help_manager.get_topic_info("trading")
    assert trading_info["title"] == "트레이딩 가이드"
    assert trading_info["description"] == "트레이딩 기본 개념과 전략"
    assert "sections" in trading_info

def test_help_content(help_manager):
    # 도움말 내용 테스트
    # 기본 섹션 내용
    content = help_manager.get_content("trading", "기본")
    assert content is not None
    assert "트레이딩 기본" in content
    
    # 전략 섹션 내용
    content = help_manager.get_content("trading", "전략")
    assert content is not None
    assert "트레이딩 전략" in content
    
    # 리스크 관리 섹션 내용
    content = help_manager.get_content("trading", "리스크 관리")
    assert content is not None
    assert "리스크 관리" in content

def test_help_search(help_manager):
    # 도움말 검색 테스트
    # 키워드 검색
    results = help_manager.search("트레이딩")
    assert len(results) > 0
    assert any("트레이딩" in result["content"] for result in results)
    
    # 섹션 검색
    results = help_manager.search("기본")
    assert len(results) > 0
    assert any("기본" in result["section"] for result in results)

def test_help_formatting(help_manager):
    # 도움말 포맷팅 테스트
    content = help_manager.get_content("trading", "기본")
    formatted_content = help_manager.format_content(content)
    
    # 마크다운 포맷팅 확인
    assert "<h1>" in formatted_content
    assert "<p>" in formatted_content

def test_help_navigation(help_manager):
    # 도움말 네비게이션 테스트
    # 다음 섹션
    next_section = help_manager.get_next_section("trading", "기본")
    assert next_section == "전략"
    
    # 이전 섹션
    prev_section = help_manager.get_prev_section("trading", "전략")
    assert prev_section == "기본"

def test_help_language(help_manager):
    # 도움말 언어 테스트
    # 언어 설정
    help_manager.set_language("en")
    assert help_manager.get_language() == "en"
    
    # 언어 변경
    help_manager.set_language("ko")
    assert help_manager.get_language() == "ko"

def test_help_error_handling(help_manager):
    # 도움말 에러 처리 테스트
    # 존재하지 않는 주제
    with pytest.raises(ValueError):
        help_manager.get_topic_info("nonexistent")
    
    # 존재하지 않는 섹션
    with pytest.raises(ValueError):
        help_manager.get_content("trading", "nonexistent")
    
    # 잘못된 언어
    with pytest.raises(ValueError):
        help_manager.set_language("invalid")

def test_help_performance(help_manager):
    # 도움말 성능 테스트
    # 대량의 검색
    start_time = datetime.now()
    
    for i in range(100):
        help_manager.search("트레이딩")
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 100개의 검색을 1초 이내에 처리
    assert processing_time < 1.0

def test_help_configuration(help_manager):
    # 도움말 설정 테스트
    config = help_manager.get_configuration()
    
    assert config is not None
    assert "help" in config
    assert "language" in config["help"]
    assert "topics" in config["help"]
    assert "trading" in config["help"]["topics"]
    assert "api" in config["help"]["topics"]
    assert "analysis" in config["help"]["topics"] 