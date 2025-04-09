from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class ReconciliationResult:
    matched: List[Dict]
    unmatched: List[Dict]
    discrepancies: List[Dict]
    reconciliation_status: str

class TradeReconciliation:
    def __init__(self):
        self.tolerance = 0.0001  # 금액 차이 허용 범위
        
    async def reconcile_trades(self, 
                             internal_trades: List[Dict],
                             exchange_trades: List[Dict]) -> ReconciliationResult:
        """거래 내역 조정"""
        matched_trades = []
        unmatched_trades = []
        discrepancies = []
        
        for internal in internal_trades:
            match = self._find_matching_trade(internal, exchange_trades)
            if match:
                if self._check_discrepancy(internal, match):
                    discrepancies.append({
                        'internal': internal,
                        'exchange': match
                    })
                else:
                    matched_trades.append(internal)
            else:
                unmatched_trades.append(internal)
                
        return ReconciliationResult(
            matched=matched_trades,
            unmatched=unmatched_trades,
            discrepancies=discrepancies,
            reconciliation_status=self._determine_status(len(discrepancies))
        )
