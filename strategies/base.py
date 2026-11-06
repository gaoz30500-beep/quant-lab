"""
策略基类
所有策略需继承此类并实现 generate_signals 方法
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any
import pandas as pd


class SignalType(Enum):
    BUY = 'BUY'
    SELL = 'SELL'
    HOLD = 'HOLD'


@dataclass
class Signal:
    """交易信号"""
    symbol: str
    signal_type: SignalType
    price: float
    date: pd.Timestamp
    reason: str = ''

    def __repr__(self) -> str:
        return (f"Signal({self.symbol}, {self.signal_type.value}, "
                f"price={self.price:.2f}, {self.date.date()}, "
                f"reason={self.reason})")


class Strategy(ABC):
    """策略基类"""

    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        根据行情数据生成信号序列

        参数:
            data: 行情数据 DataFrame，索引为日期

        返回:
            pd.Series，值为 SignalType，索引与 data 一致
        """
        pass

    @abstractmethod
    def get_indicator_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        返回用于绘图的指标数据 (与 data 同索引)
        """
        pass

    def describe(self) -> str:
        params_str = ', '.join(f'{k}={v}' for k, v in self.params.items())
        return f"[{self.name}] {params_str}"
