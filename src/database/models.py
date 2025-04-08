from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, JSON, Enum
from sqlalchemy.ext.declarative import declarative_base
import enum

Base = declarative_base()

class MarketData(Base):
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    strategy_id = Column(Integer, ForeignKey('strategies.id'))

class OrderType(enum.Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class Order(Base):
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String, nullable=False)
    order_type = Column(Enum(OrderType))
    side = Column(String)
    amount = Column(Float)
    price = Column(Float)
    status = Column(String)
    trade_id = Column(Integer, ForeignKey('trades.id'))

class Strategy(Base):
    __tablename__ = 'strategies'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    parameters = Column(JSON)
    status = Column(String)
    last_updated = Column(DateTime)
