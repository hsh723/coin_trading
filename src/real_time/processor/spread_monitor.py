import asyncio
from typing import Dict, List

class SpreadMonitor:
    def __init__(self, monitor_config: Dict = None):
        self.config = monitor_config or {
            'spread_threshold': 0.001,
            'monitoring_interval': 0.1
        }
        
    async def monitor_spreads(self, order_book_stream: asyncio.Queue) -> Dict:
        """실시간 스프레드 모니터링"""
        while True:
            order_book = await order_book_stream.get()
            spread_analysis = {
                'current_spread': self._calculate_current_spread(order_book),
                'spread_metrics': self._analyze_spread_metrics(order_book),
                'liquidity_impact': self._calculate_liquidity_impact(order_book),
                'warning_signals': self._check_warning_signals(order_book)
            }
            
            await self._process_spread_alerts(spread_analysis)
            await asyncio.sleep(self.config['monitoring_interval'])
