"""
闸门策略包装器
两层过滤:
  1. 大盘情绪闸门 — risk-off 时不买
  2. 个股趋势过滤 — 个股在自身均线下方时不买

设计原则: 两层过滤收敛到 generate_signals 单一入口
"""
import pandas as pd
from .base import Strategy, SignalType
from market_gate import MarketGate, MarketRegime


class GatedStrategy(Strategy):
    """
    带双重闸门的策略包装器

    信号流程 (唯一入口):
      1. inner_strategy.generate_signals() → 原始信号
      2. market_gate.filter_signals()     → 大盘 risk-off 屏蔽 BUY
      3. stock_trend_filter()             → 个股下跌趋势屏蔽 BUY
    """

    def __init__(
        self,
        inner_strategy: Strategy,
        gate: MarketGate,
        stock_ma_window: int = 20,      # 个股趋势均线周期
        stock_max_drop: float = -0.08,  # 个股近N日跌幅超过此值则屏蔽
        enable_stock_filter: bool = True,
    ):
        self.inner = inner_strategy
        self.gate = gate
        self.stock_ma_window = stock_ma_window
        self.stock_max_drop = stock_max_drop
        self.enable_stock_filter = enable_stock_filter
        super().__init__(
            name=f'[双闸门] {inner_strategy.name}',
            params={
                **inner_strategy.params,
                'stock_ma_window': stock_ma_window,
                'stock_max_drop': stock_max_drop,
            },
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # 1. 内层策略生成信号
        raw_signals = self.inner.generate_signals(data)

        # 2. 大盘闸门过滤
        filtered = self.gate.filter_signals(raw_signals, data.index)

        # 3. 个股趋势过滤 (只在均线上方才允许买入)
        if self.enable_stock_filter:
            filtered = self._stock_trend_filter(filtered, data)

        return filtered

    def _stock_trend_filter(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        个股暴跌过滤: 近 stock_ma_window 日跌幅超过 stock_max_drop 时屏蔽买入

        不屏蔽普通下跌，只屏蔽"短期暴跌"的股票
        (例如近20日跌超8%，说明个股正在加速下跌，不抄底)
        """
        close = data['close']
        recent_return = close.pct_change(self.stock_ma_window)
        not_crashing = recent_return > self.stock_max_drop

        filtered = signals.copy()
        buy_mask = filtered == SignalType.BUY
        filtered[buy_mask & ~not_crashing] = SignalType.HOLD

        return filtered

    def get_indicator_data(self, data: pd.DataFrame) -> pd.DataFrame:
        inner_indicators = self.inner.get_indicator_data(data)

        # 大盘闸门状态
        regimes = self.gate.classify(data.index)
        gate_state = regimes.map({MarketRegime.RISK_ON: 1, MarketRegime.RISK_OFF: 0})
        inner_indicators['market_gate'] = gate_state

        # 个股趋势状态
        if self.enable_stock_filter:
            close = data['close']
            stock_ma = close.rolling(self.stock_ma_window).mean()
            inner_indicators['stock_trend'] = (close > stock_ma).astype(int)

        return inner_indicators
