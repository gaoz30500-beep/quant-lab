"""
数据加载模块
支持 CSV 格式日线数据，列名: date, open, high, low, close, volume
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union


class DataLoader:
    """行情数据加载器"""

    REQUIRED_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume']

    @staticmethod
    def load_csv(
        filepath: Union[str, Path],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        加载CSV行情数据

        参数:
            filepath: CSV文件路径
            start_date: 起始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'

        返回:
            以 date 为索引的 DataFrame，列: open, high, low, close, volume
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"数据文件不存在: {filepath}")

        df = pd.read_csv(filepath)
        df.columns = [c.strip().lower() for c in df.columns]

        missing = set(DataLoader.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"CSV缺少必要列: {missing}")

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = df.set_index('date')

        # 数据清洗
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['close'])

        # 日期过滤
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        if df.empty:
            raise ValueError("过滤后数据为空，请检查日期范围")

        return df

    @staticmethod
    def load_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """从已有DataFrame加载，做基础校验"""
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        missing = set(DataLoader.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame缺少必要列: {missing}")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        return df

    @staticmethod
    def generate_sample_data(
        days: int = 500,
        start_price: float = 100.0,
        seed: int = 42,
        volatility: float = 0.02,
        drift: float = 0.0003,
    ) -> pd.DataFrame:
        """
        生成模拟行情数据（用于测试）
        几何布朗运动 + 波动率聚集
        """
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)
        n = len(dates)

        # 波动率聚集 (GARCH-like)
        returns = np.zeros(n)
        sigma = volatility
        for i in range(n):
            sigma = np.sqrt(0.9 * sigma**2 + 0.1 * volatility**2)
            shock = rng.normal(0, 1)
            returns[i] = drift + sigma * shock

        prices = start_price * np.cumprod(1 + returns)
        high = prices * (1 + np.abs(rng.normal(0, volatility, n)) * 0.5)
        low = prices * (1 - np.abs(rng.normal(0, volatility, n)) * 0.5)
        volume = rng.integers(1_000_000, 10_000_000, n)
        opens = np.concatenate([[start_price], prices[:-1]])

        df = pd.DataFrame({
            'date': dates,
            'open': opens,
            'high': high,
            'low': low,
            'close': prices,
            'volume': volume,
        })
        return df.set_index('date')
