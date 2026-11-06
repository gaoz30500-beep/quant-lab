"""
风控模块
- 仓位管理
- 止损/止盈
- 移动止损
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class Position:
    """持仓"""
    symbol: str
    shares: int
    entry_price: float
    entry_date: pd.Timestamp
    highest_price: float           # 持仓期间最高价 (用于移动止损)
    stop_loss_price: float
    take_profit_price: float

    def update(self, price: float) -> None:
        """更新持仓最高价"""
        if price > self.highest_price:
            self.highest_price = price

    def unrealized_pnl(self, current_price: float) -> float:
        """浮动盈亏"""
        return (current_price - self.entry_price) * self.shares


class RiskManager:
    """风控管理器"""

    def __init__(
        self,
        stop_loss_pct: float = 0.08,
        take_profit_pct: float = 0.15,
        trailing_stop_pct: float = 0.05,
        trailing_activation_pct: float = 0.0,
        position_size_pct: float = 0.95,
        max_positions: int = 5,
    ):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions

    def calc_position_size(
        self,
        cash: float,
        price: float,
        commission_rate: float,
        min_commission: float,
        stamp_tax: float = 0.0,
    ) -> int:
        """
        计算可买入股数 (A股100股整数倍)
        考虑手续费和印花税
        """
        # 可用资金上限
        max_amount = cash * self.position_size_pct
        # 每股成本 = 价格 + 手续费 + 印花税
        per_share_cost = price * (1 + commission_rate + stamp_tax)
        # 考虑最低手续费
        raw_shares = max_amount / per_share_cost
        # 向下取整到100的倍数
        shares = int(raw_shares // 100) * 100
        return max(0, shares)

    def check_exit(
        self,
        position: Position,
        current_price: float,
    ) -> Optional[str]:
        """
        检查是否需要平仓
        返回: 'stop_loss' / 'take_profit' / 'trailing_stop' / None
        """
        # 固定止损 (stop_loss_pct<=0 表示关闭)
        if self.stop_loss_pct > 0 and current_price <= position.stop_loss_price:
            return 'stop_loss'

        # 固定止盈 (take_profit_pct<=0 表示关闭)
        if self.take_profit_pct > 0 and current_price >= position.take_profit_price:
            return 'take_profit'

        # 移动止损 (从最高点回撤超过阈值)
        # 仅在浮盈达到 trailing_activation_pct 后才启用, 避免早期正常波动被打断
        if self.trailing_stop_pct > 0:
            activated = (
                self.trailing_activation_pct <= 0
                or position.highest_price >= position.entry_price * (1 + self.trailing_activation_pct)
            )
            if activated:
                trailing_price = position.highest_price * (1 - self.trailing_stop_pct)
                if current_price <= trailing_price and current_price < position.highest_price:
                    return 'trailing_stop'

        return None

    def create_position(
        self,
        symbol: str,
        shares: int,
        entry_price: float,
        entry_date: pd.Timestamp,
    ) -> Position:
        """创建持仓，设置止损止盈价"""
        return Position(
            symbol=symbol,
            shares=shares,
            entry_price=entry_price,
            entry_date=entry_date,
            highest_price=entry_price,
            stop_loss_price=entry_price * (1 - self.stop_loss_pct),
            take_profit_price=entry_price * (1 + self.take_profit_pct),
        )

    def calc_commission(
        self,
        amount: float,
        commission_rate: float,
        min_commission: float,
    ) -> float:
        """计算手续费"""
        return max(amount * commission_rate, min_commission)

    def calc_stamp_tax(self, amount: float, stamp_tax_rate: float) -> float:
        """计算印花税 (卖出)"""
        return amount * stamp_tax_rate
