"""
사용자 피드백 수집 모듈
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

class FeedbackSystem:
    """사용자 피드백 수집 클래스"""
    
    def __init__(self, db_manager):
        """
        초기화
        
        Args:
            db_manager: 데이터베이스 관리자
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.feedback_clusters = None
        
    def submit_feedback(self,
                       user_id: str,
                       feedback_type: str,
                       content: str,
                       rating: int,
                       metadata: Optional[Dict] = None) -> bool:
        """
        피드백 제출
        
        Args:
            user_id (str): 사용자 ID
            feedback_type (str): 피드백 유형
            content (str): 피드백 내용
            rating (int): 평점 (1-5)
            metadata (Optional[Dict]): 추가 메타데이터
            
        Returns:
            bool: 제출 성공 여부
        """
        try:
            # 감정 분석
            sentiment = self.sentiment_analyzer.polarity_scores(content)
            
            # 피드백 데이터 저장
            feedback_data = {
                'user_id': user_id,
                'feedback_type': feedback_type,
                'content': content,
                'rating': rating,
                'sentiment': sentiment,
                'metadata': metadata or {},
                'timestamp': datetime.now()
            }
            
            self.db_manager.save_feedback(feedback_data)
            
            # 피드백 클러스터링 업데이트
            self._update_feedback_clusters()
            
            return True
            
        except Exception as e:
            self.logger.error(f"피드백 제출 실패: {str(e)}")
            return False
            
    def _update_feedback_clusters(self):
        """피드백 클러스터링 업데이트"""
        try:
            # 피드백 데이터 로드
            feedback_data = self.db_manager.get_feedback_data()
            
            if not feedback_data:
                return
                
            # 텍스트 벡터화
            texts = [f['content'] for f in feedback_data]
            X = self.vectorizer.fit_transform(texts)
            
            # 클러스터링
            n_clusters = min(5, len(feedback_data))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X)
            
            # 클러스터 결과 저장
            self.feedback_clusters = {
                'clusters': clusters,
                'centers': kmeans.cluster_centers_,
                'labels': self.vectorizer.get_feature_names_out()
            }
            
        except Exception as e:
            self.logger.error(f"피드백 클러스터링 업데이트 실패: {str(e)}")
            
    def get_feedback_summary(self) -> Dict:
        """
        피드백 요약 조회
        
        Returns:
            Dict: 피드백 요약
        """
        try:
            # 피드백 데이터 로드
            feedback_data = self.db_manager.get_feedback_data()
            
            if not feedback_data:
                return {}
                
            # 기본 통계 계산
            total_feedback = len(feedback_data)
            avg_rating = np.mean([f['rating'] for f in feedback_data])
            
            # 감정 분석 통계
            sentiments = [f['sentiment'] for f in feedback_data]
            avg_sentiment = {
                'positive': np.mean([s['pos'] for s in sentiments]),
                'negative': np.mean([s['neg'] for s in sentiments]),
                'neutral': np.mean([s['neu'] for s in sentiments])
            }
            
            # 피드백 유형별 통계
            feedback_types = {}
            for f in feedback_data:
                if f['feedback_type'] not in feedback_types:
                    feedback_types[f['feedback_type']] = {
                        'count': 0,
                        'avg_rating': 0,
                        'total_rating': 0
                    }
                feedback_types[f['feedback_type']]['count'] += 1
                feedback_types[f['feedback_type']]['total_rating'] += f['rating']
                
            for f_type in feedback_types:
                feedback_types[f_type]['avg_rating'] = (
                    feedback_types[f_type]['total_rating'] /
                    feedback_types[f_type]['count']
                )
                
            return {
                'total_feedback': total_feedback,
                'average_rating': avg_rating,
                'average_sentiment': avg_sentiment,
                'feedback_types': feedback_types,
                'clusters': self.feedback_clusters
            }
            
        except Exception as e:
            self.logger.error(f"피드백 요약 조회 실패: {str(e)}")
            return {}
            
    def get_feedback_trends(self,
                          start_time: datetime,
                          end_time: datetime) -> Dict:
        """
        피드백 트렌드 조회
        
        Args:
            start_time (datetime): 시작 시간
            end_time (datetime): 종료 시간
            
        Returns:
            Dict: 피드백 트렌드
        """
        try:
            # 기간별 피드백 데이터 로드
            feedback_data = self.db_manager.get_feedback_data(
                start_time=start_time,
                end_time=end_time
            )
            
            if not feedback_data:
                return {}
                
            # 시간대별 통계 계산
            feedback_by_time = {}
            for f in feedback_data:
                time_key = f['timestamp'].strftime('%Y-%m-%d')
                if time_key not in feedback_by_time:
                    feedback_by_time[time_key] = {
                        'count': 0,
                        'total_rating': 0,
                        'sentiments': []
                    }
                feedback_by_time[time_key]['count'] += 1
                feedback_by_time[time_key]['total_rating'] += f['rating']
                feedback_by_time[time_key]['sentiments'].append(f['sentiment'])
                
            # 시간대별 평균 계산
            for time_key in feedback_by_time:
                data = feedback_by_time[time_key]
                data['average_rating'] = data['total_rating'] / data['count']
                data['average_sentiment'] = {
                    'positive': np.mean([s['pos'] for s in data['sentiments']]),
                    'negative': np.mean([s['neg'] for s in data['sentiments']]),
                    'neutral': np.mean([s['neu'] for s in data['sentiments']])
                }
                del data['sentiments']
                
            return feedback_by_time
            
        except Exception as e:
            self.logger.error(f"피드백 트렌드 조회 실패: {str(e)}")
            return {}
            
    def get_improvement_suggestions(self) -> List[Dict]:
        """
        개선 제안 조회
        
        Returns:
            List[Dict]: 개선 제안 목록
        """
        try:
            # 부정적인 피드백 필터링
            negative_feedback = self.db_manager.get_feedback_data(
                min_rating=1,
                max_rating=3
            )
            
            if not negative_feedback:
                return []
                
            # 클러스터링 결과가 있는 경우
            if self.feedback_clusters:
                # 각 클러스터의 주요 키워드 추출
                suggestions = []
                for cluster_id in range(len(self.feedback_clusters['centers'])):
                    center = self.feedback_clusters['centers'][cluster_id]
                    top_keywords = [
                        self.feedback_clusters['labels'][i]
                        for i in center.argsort()[-5:][::-1]
                    ]
                    
                    # 해당 클러스터의 피드백 수집
                    cluster_feedback = [
                        f for i, f in enumerate(negative_feedback)
                        if self.feedback_clusters['clusters'][i] == cluster_id
                    ]
                    
                    suggestions.append({
                        'keywords': top_keywords,
                        'feedback_count': len(cluster_feedback),
                        'average_rating': np.mean([f['rating'] for f in cluster_feedback]),
                        'examples': [f['content'] for f in cluster_feedback[:3]]
                    })
                    
                return suggestions
                
            return []
            
        except Exception as e:
            self.logger.error(f"개선 제안 조회 실패: {str(e)}")
            return [] 