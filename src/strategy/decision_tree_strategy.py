from typing import Dict, List
from dataclasses import dataclass
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

@dataclass
class TreeDecision:
    signal: str
    confidence: float
    decision_path: List[str]
    feature_importance: Dict[str, float]

class DecisionTreeStrategy:
    def __init__(self, tree_config: Dict = None):
        self.config = tree_config or {
            'max_depth': 5,
            'min_samples_split': 10,
            'features': ['rsi', 'macd', 'volume', 'trend']
        }
        self.model = DecisionTreeClassifier(
            max_depth=self.config['max_depth'],
            min_samples_split=self.config['min_samples_split']
        )
        
    async def make_decision(self, market_data: pd.DataFrame) -> TreeDecision:
        """의사결정 트리 기반 매매 결정"""
        features = self._extract_features(market_data)
        prediction_proba = self.model.predict_proba(features)
        
        return TreeDecision(
            signal=self._get_signal(prediction_proba),
            confidence=self._get_confidence(prediction_proba),
            decision_path=self._get_decision_path(features),
            feature_importance=dict(zip(
                self.config['features'],
                self.model.feature_importances_
            ))
        )
