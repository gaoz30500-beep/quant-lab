"""
策略模块
"""
from .base import Strategy, SignalType
from .ma_cross import MACrossStrategy
from .bollinger import BollingerStrategy
from .combined import CombinedStrategy
from .gated import GatedStrategy

__all__ = ['Strategy', 'SignalType', 'MACrossStrategy', 'BollingerStrategy',
           'CombinedStrategy', 'GatedStrategy']
