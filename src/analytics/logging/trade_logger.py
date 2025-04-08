from typing import Dict
import pandas as pd
from datetime import datetime
import logging

class TradeLogger:
    def __init__(self, log_file: str = "trades.log"):
        self.logger = logging.getLogger("trade_logger")
        self.logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
        
    async def log_trade(self, trade: Dict):
        """거래 정보 로깅"""
        self.logger.info(
            f"Trade executed: {trade['symbol']} | "
            f"Side: {trade['side']} | "
            f"Price: {trade['price']} | "
            f"Amount: {trade['amount']} | "
            f"Strategy: {trade.get('strategy', 'unknown')}"
        )
