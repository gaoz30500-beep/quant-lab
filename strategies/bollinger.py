"""
布林带均值回归策略
价格跌破下轨 -> 买入 (超卖回归)
价格突破上轨 -> 卖出 (超买回归)
"""
import pandas as pd
import numpy as np
from .base import Strategy, SignalType


class BollingerStrategy(Strategy):
    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__(
            name='布林带均值回归',
            params={'window': window, 'num_std': num_std},
        )
        self.window = window
        self.num_std = num_std

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        ma = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        upper = ma + self.num_std * std
        lower = ma - self.num_std * std

        signals = pd.Series(SignalType.HOLD, index=data.index, dtype=object)

        # 价格跌破下轨 -> 买入
        buy_signal = (close < lower) & (close.shift(1) >= lower.shift(1))
        # 价格突破上轨 -> 卖出
        sell_signal = (close > upper) & (close.shift(1) <= upper.shift(1))

        signals[buy_signal] = SignalType.BUY
        signals[sell_signal] = SignalType.SELL

        return signals

    def get_indicator_data(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data['close']
        ma = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        df = pd.DataFrame(index=data.index)
        df['MA'] = ma
        df['Upper'] = ma + self.num_std * std
        df['Lower'] = ma - self.num_std * std
        return df
