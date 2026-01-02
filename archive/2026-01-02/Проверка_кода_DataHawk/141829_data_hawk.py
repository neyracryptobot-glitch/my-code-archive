import asyncio
import pandas as pd
import aiohttp
from typing import Dict, List, Optional
from datetime import datetime
from abc import ABC, abstractmethod

class DataSource(ABC):
    @abstractmethod
    async def fetch(self) -> pd.DataFrame:
        pass

class DataHawk:
    def __init__(self, sources: List[DataSource]):
        self.sources = sources
        self._cache: Dict[str, pd.DataFrame] = {}

    async def collect_all(self) -> pd.DataFrame:
        tasks = [source.fetch() for source in self.sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_frames = [
            df for df in results 
            if isinstance(df, pd.DataFrame) and not df.empty
        ]
        
        if not valid_frames:
            return pd.DataFrame()
            
        return self._merge_and_clean(valid_frames)

    def _merge_and_clean(self, frames: List[pd.DataFrame]) -> pd.DataFrame:
        combined = pd.concat(frames, axis=0).drop_duplicates()
        combined = combined.sort_index()
        
        # Векторизованная очистка
        if 'timestamp' in combined.columns:
            combined['timestamp'] = pd.to_datetime(combined['timestamp'])
            combined.set_index('timestamp', inplace=True)
            
        return combined.fillna(method='ffill').dropna()

class BinanceSource(DataSource):
    def __init__(self, symbol: str, interval: str = '1m'):
        self.symbol = symbol
        self.interval = interval
        self.endpoint = "https://api.binance.com/api/v3/klines"

    async def fetch(self) -> pd.DataFrame:
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": 1000
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.endpoint, params=params, timeout=10) as resp:
                    if resp.status != 200:
                        return pd.DataFrame()
                    data = await resp.json()
                    return self._parse(data)
            except Exception:
                return pd.DataFrame()

    def _parse(self, raw_data: List[List]) -> pd.DataFrame:
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'
        ]
        df = pd.DataFrame(raw_data, columns=columns)
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df[['timestamp'] + numeric_cols].set_index('timestamp')

async def get_market_snapshot(symbols: List[str]):
    sources = [BinanceSource(symbol) for symbol in symbols]
    hawk = DataHawk(sources)
    return await hawk.collect_all()

# Пример инициализации для тестов
# if __name__ == "__main__":
#     data = asyncio.run(get_market_snapshot(['BTCUSDT', 'ETHUSDT']))