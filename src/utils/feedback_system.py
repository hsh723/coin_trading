"""
피드백 시스템 모듈
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
from pathlib import Path
import json

class FeedbackSystem:
    """피드백 시스템 클래스"""
    
    def __init__(self):
        """피드백 시스템 초기화"""
        self.logger = logging.getLogger(__name__)
        self.feedback_path = Path(__file__).parent.parent.parent / 'data' / 'feedback'
        self.feedback_path.mkdir(exist_ok=True)
        
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """기본 감정 분석"""
        return {
            'positive': 0.5,
            'negative': 0.5,
            'neutral': 0.0
        }
        
    def process_feedback(self, feedback: Dict) -> Dict:
        """피드백 처리"""
        self.logger.info(f"피드백 처리: {feedback}")
        return {
            'timestamp': datetime.now().isoformat(),
            'type': feedback.get('type', 'unknown'),
            'sentiment': self.analyze_sentiment(feedback.get('text', '')),
            'rating': feedback.get('rating', 0),
            'metadata': feedback.get('metadata', {})
        }
        
    def save_feedback(self, feedback: Dict) -> None:
        """피드백 저장"""
        feedback_file = self.feedback_path / f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback, f, ensure_ascii=False, indent=2)
            
    def get_feedback_summary(self) -> Dict:
        """피드백 요약"""
        feedback_files = list(self.feedback_path.glob('feedback_*.json'))
        if not feedback_files:
            return {
                'total_feedback': 0,
                'average_rating': 0,
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0}
            }
            
        feedbacks = []
        for file in feedback_files:
            with open(file, 'r', encoding='utf-8') as f:
                feedbacks.append(json.load(f))
                
        return {
            'total_feedback': len(feedbacks),
            'average_rating': np.mean([f.get('rating', 0) for f in feedbacks]),
            'sentiment_distribution': {
                'positive': len([f for f in feedbacks if f.get('sentiment', {}).get('positive', 0) > 0.5]),
                'negative': len([f for f in feedbacks if f.get('sentiment', {}).get('negative', 0) > 0.5]),
                'neutral': len([f for f in feedbacks if f.get('sentiment', {}).get('neutral', 0) > 0.5])
            }
        } 