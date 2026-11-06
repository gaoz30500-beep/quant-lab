"""
大盘情绪闸门 (Market Sentiment Gate)

第一层风控：判断大盘环境是 risk-on 还是 risk-off
  risk-on  → 允许策略开仓
  risk-off → 禁止开仓（但允许平仓）

设计原则:
- 闸门判断收敛到唯一函数 is_risk_on()
- 用显式阈值表，不用隐式 if 碰运气
- 闸门只过滤 BUY 信号，不影响 SELL（该止损就止损）
"""
from enum import Enum
import pandas as pd
import numpy as np

from data_source import DataSource


class MarketRegime(Enum):
    RISK_ON = 'RISK_ON'      # 大盘向好，可以开仓
    RISK_OFF = 'RISK_OFF'    # 大盘走弱，禁止开仓


class MarketGate:
    """
    大盘情绪闸门

    判断规则 (唯一决策入口):
      指数收盘价 > 指数MA(ma_window)  → RISK_ON
      指数收盘价 <= 指数MA(ma_window) → RISK_OFF

    可选增强条件:
      - 指数近 lookback 日涨幅 > min_return (动量确认)
    """

    def __init__(
        self,
        index_symbol: str = '000001',   # 上证指数代码
        ma_window: int = 60,             # 大盘均线周期
        lookback_days: int = 20,         # 动量回看天数
        min_return: float = -0.03,       # 近N日跌幅超过此值则 risk-off
        use_momentum_filter: bool = True,
    ):
        self.index_symbol = index_symbol
        self.ma_window = ma_window
        self.lookback_days = lookback_days
        self.min_return = min_return
        self.use_momentum_filter = use_momentum_filter
        self._index_data: pd.DataFrame = None

    def load_index_data(self, start_date: str = '20100101', end_date: str = None):
        """加载指数日线数据"""
        ds = DataSource(use_cache=True)
        try:
            self._index_data = ds.get_index_daily(
                self.index_symbol, start_date=start_date, end_date=end_date,
            )
        except Exception:
            self._index_data = None
        return self._index_data

    def set_index_data(self, data: pd.DataFrame):
        """直接设置预加载的指数数据 (优化时复用)"""
        self._index_data = data

    def classify(self, data_index: pd.DatetimeIndex) -> pd.Series:
        """
        唯一决策入口：对每个交易日判断大盘状态

        参数:
            data_index: 个股交易日索引 (用于对齐)
        返回: pd.Series，值为 MarketRegime
        """
        if self._index_data is None:
            # 无指数数据时，默认全部 risk-on (不做过滤)
            return pd.Series(MarketRegime.RISK_ON, index=data_index)

        idx_close = self._index_data['close']
        idx_ma = idx_close.rolling(self.ma_window).mean()

        # 规则1: 指数在均线之上
        above_ma = idx_close > idx_ma

        # 规则2: 近 lookback_days 日跌幅不超过 min_return
        if self.use_momentum_filter:
            idx_return = idx_close.pct_change(self.lookback_days)
            not_oversold = idx_return > self.min_return
        else:
            not_oversold = pd.Series(True, index=idx_close.index)

        # 综合判断
        risk_on = above_ma & not_oversold

        # 对齐到交易日索引 (前向填充，非交易日沿用最近状态)
        regime = pd.Series(MarketRegime.RISK_OFF, index=idx_close.index, dtype=object)
        regime[risk_on] = MarketRegime.RISK_ON

        # 对齐到个股交易日
        regime = regime.reindex(data_index, method='ffill')
        regime = regime.fillna(MarketRegime.RISK_OFF)

        return regime

    def filter_signals(
        self,
        signals: pd.Series,
        data_index: pd.DatetimeIndex,
    ) -> pd.Series:
        """
        闸门过滤：risk-off 时屏蔽 BUY 信号，保留 SELL 和 HOLD

        这是闸门的唯一执行入口
        """
        # 延迟导入，避免与 strategies 包形成循环依赖
        from strategies.base import SignalType

        regimes = self.classify(data_index)
        filtered = signals.copy()

        # risk-off 时，BUY → HOLD (禁止开仓)
        risk_off_mask = regimes == MarketRegime.RISK_OFF
        buy_mask = filtered == SignalType.BUY
        filtered[risk_off_mask & buy_mask] = SignalType.HOLD

        # SELL 信号不受影响 (该止损就止损，该止盈就止盈)

        return filtered

    def get_gate_stats(self, regimes: pd.Series) -> dict:
        """统计闸门开关占比"""
        counts = regimes.value_counts()
        total = len(regimes)
        return {
            'risk_on_pct': f"{counts.get(MarketRegime.RISK_ON, 0) / total:.1%}",
            'risk_off_pct': f"{counts.get(MarketRegime.RISK_OFF, 0) / total:.1%}",
        }
