"""
双均线策略 (趋势跟踪)
快均线上穿慢均线 -> 买入
快均线下穿慢均线 -> 卖出
"""
import pandas as pd
from .base import Strategy, SignalType


class MACrossStrategy(Strategy):
    def __init__(self, fast_window: int = 5, slow_window: int = 20):
        super().__init__(
            name='双均线策略',
            params={'fast_window': fast_window, 'slow_window': slow_window},
        )
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        fast_ma = close.rolling(self.fast_window).mean()
        slow_ma = close.rolling(self.slow_window).mean()

        signals = pd.Series(SignalType.HOLD, index=data.index, dtype=object)

        # 金叉: 快均线从下方上穿慢均线
        golden_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        # 死叉: 快均线从上方下穿慢均线
        death_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

        signals[golden_cross] = SignalType.BUY
        signals[death_cross] = SignalType.SELL

        return signals

    def get_indicator_data(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data['close']
        df = pd.DataFrame(index=data.index)
        df[f'MA{self.fast_window}'] = close.rolling(self.fast_window).mean()
        df[f'MA{self.slow_window}'] = close.rolling(self.slow_window).mean()
        return df
