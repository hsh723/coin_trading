from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import List, Dict
import pandas as pd
from .models import Base, MarketData, Trade

class DatabaseManager:
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def save_market_data(self, data: pd.DataFrame) -> None:
        """시장 데이터 저장"""
        session = self.Session()
        try:
            for _, row in data.iterrows():
                market_data = MarketData(
                    timestamp=row['timestamp'],
                    symbol=row['symbol'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume']
                )
                session.add(market_data)
            session.commit()
        finally:
            session.close()
