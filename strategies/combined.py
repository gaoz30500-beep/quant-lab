"""
组合策略：趋势市用双均线，震荡市用布林带

设计原则 (来自经验):
- 信号生成收敛到唯一函数 generate_signals()
- 市场状态判断只调用 MarketStateDetector.classify()，不在策略内复算
- 子策略只负责各自的信号逻辑，不做状态判断
"""
import pandas as pd
from .base import Strategy, SignalType
from .ma_cross import MACrossStrategy
from .bollinger import BollingerStrategy
from market_state import MarketStateDetector, MarketState


class CombinedStrategy(Strategy):
    """
    市场状态自适应组合策略

    决策流程 (单一入口):
      1. MarketStateDetector.classify() → 判断趋势/震荡
      2. 趋势市 → MACrossStrategy 信号
      3. 震荡市 → BollingerStrategy 信号
    """

    def __init__(
        self,
        fast_window: int = 5,
        slow_window: int = 30,
        boll_window: int = 20,
        boll_std: float = 2.0,
        adx_window: int = 14,
        adx_threshold: float = 25.0,
    ):
        super().__init__(
            name='市场状态自适应组合',
            params={
                'fast_window': fast_window,
                'slow_window': slow_window,
                'boll_window': boll_window,
                'boll_std': boll_std,
                'adx_window': adx_window,
                'adx_threshold': adx_threshold,
            },
        )
        # 子策略
        self.ma_cross = MACrossStrategy(fast_window=fast_window, slow_window=slow_window)
        self.bollinger = BollingerStrategy(window=boll_window, num_std=boll_std)
        # 状态检测器 (唯一决策入口)
        self.state_detector = MarketStateDetector(
            adx_window=adx_window,
            adx_threshold=adx_threshold,
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # 1. 唯一状态分类入口
        states = self.state_detector.classify(data)

        # 2. 各子策略生成信号
        ma_signals = self.ma_cross.generate_signals(data)
        boll_signals = self.bollinger.generate_signals(data)

        # 3. 根据状态选择信号 (唯一信号合并入口)
        signals = pd.Series(SignalType.HOLD, index=data.index, dtype=object)

        trending_mask = states == MarketState.TRENDING
        ranging_mask = states == MarketState.RANGING

        # 趋势市用双均线，震荡市用布林带
        signals[trending_mask] = ma_signals[trending_mask]
        signals[ranging_mask] = boll_signals[ranging_mask]

        return signals

    def get_indicator_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """返回绘图指标：均线 + 布林带 + ADX + 市场状态"""
        close = data['close']

        # 均线
        ma_fast = close.rolling(self.params['fast_window']).mean()
        ma_slow = close.rolling(self.params['slow_window']).mean()

        # 布林带
        boll_ma = close.rolling(self.params['boll_window']).mean()
        boll_std = close.rolling(self.params['boll_window']).std()
        boll_upper = boll_ma + self.params['boll_std'] * boll_std
        boll_lower = boll_ma - self.params['boll_std'] * boll_std

        # ADX
        adx = self.state_detector._calc_adx(data)

        # 市场状态 (数值化便于绘图)
        states = self.state_detector.classify(data)
        state_code = states.map({
            MarketState.TRENDING: 1,
            MarketState.RANGING: 0,
            MarketState.UNKNOWN: -1,
        })

        df = pd.DataFrame({
            'MA_fast': ma_fast,
            'MA_slow': ma_slow,
            'Boll_upper': boll_upper,
            'Boll_mid': boll_ma,
            'Boll_lower': boll_lower,
            'ADX': adx,
            'market_state': state_code,
        }, index=data.index)
        return df
