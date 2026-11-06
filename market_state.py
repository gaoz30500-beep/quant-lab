"""
市场状态检测器
唯一决策入口：判断当前是趋势市还是震荡市

设计原则 (来自经验):
- 状态分类收敛到唯一函数 classify()，其它位置只调用不复写
- 用显式阈值表表达判断逻辑，不用隐式 if 顺序碰运气
"""
from enum import Enum
import pandas as pd
import numpy as np


class MarketState(Enum):
    TRENDING = 'TRENDING'      # 趋势市 → 用趋势策略
    RANGING = 'RANGING'        # 震荡市 → 用均值回归策略
    UNKNOWN = 'UNKNOWN'        # 数据不足


class MarketStateDetector:
    """
    市场状态检测器

    判断逻辑 (单一决策入口):
      ADX > adx_threshold  → 趋势市 (TRENDING)
      ADX <= adx_threshold → 震荡市 (RANGING)

    ADX 衡量趋势强度，不区分方向，只衡量"有没有趋势"
    """

    def __init__(self, adx_window: int = 14, adx_threshold: float = 25.0):
        """
        参数:
            adx_window: ADX 计算周期 (默认14，标准值)
            adx_threshold: ADX 阈值，>此值为趋势市
                          (ADX>25 通常认为有明确趋势，<20 为震荡)
        """
        self.adx_window = adx_window
        self.adx_threshold = adx_threshold

    def classify(self, data: pd.DataFrame) -> pd.Series:
        """
        唯一状态分类入口

        返回: pd.Series，值为 MarketState，索引与 data 一致
        """
        adx = self._calc_adx(data)

        states = pd.Series(MarketState.UNKNOWN, index=data.index, dtype=object)

        # 显式阈值判断 (唯一决策点)
        states[adx > self.adx_threshold] = MarketState.TRENDING
        states[adx <= self.adx_threshold] = MarketState.RANGING

        # 前 adx_window 根数据不足，标记为 UNKNOWN
        states.iloc[:self.adx_window * 2] = MarketState.UNKNOWN

        return states

    def _calc_adx(self, data: pd.DataFrame) -> pd.Series:
        """计算 ADX (Average Directional Index)"""
        high = data['high']
        low = data['low']
        close = data['close']

        # +DM / -DM
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0),
                            index=data.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0),
                             index=data.index)

        # TR (True Range)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 平滑 (Wilder's smoothing = EMA with alpha=1/window)
        window = self.adx_window
        atr = tr.ewm(alpha=1/window, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1/window, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=1/window, adjust=False).mean() / atr.replace(0, np.nan)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1/window, adjust=False).mean()

        return adx

    def get_state_distribution(self, states: pd.Series) -> dict:
        """统计各状态占比 (用于分析)"""
        counts = states.value_counts()
        total = len(states)
        return {state.value: f"{counts.get(state, 0)/total:.1%}"
                for state in MarketState}
