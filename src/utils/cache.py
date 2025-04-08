from collections import OrderedDict
from typing import Any, Optional
import numpy as np
import pandas as pd

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

class DataFrameCache:
    def __init__(self, max_size_mb: int = 1000):
        self.max_size = max_size_mb * 1024 * 1024  # Convert to bytes
        self.cache = {}
        self.size = 0

    def _get_size(self, df: pd.DataFrame) -> int:
        return df.memory_usage(deep=True).sum()

    def add(self, key: str, df: pd.DataFrame) -> None:
        size = self._get_size(df)
        if size > self.max_size:
            return
        
        while self.size + size > self.max_size and self.cache:
            _, oldest_df = self.cache.popitem(last=False)
            self.size -= self._get_size(oldest_df)
            
        self.cache[key] = df
        self.size += size
