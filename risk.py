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
    buy_commission: float = 0.0    # 建仓时实际支付的手续费 (计入该笔交易成本)

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
    ) -> int:
        """
        计算可买入股数 (A股100股整数倍)

        买入不收印花税, 所以每股成本只含价格与佣金;
        最低佣金按"按比例算出的佣金不足 5 元时取 5 元"处理:
        先用比例估算, 若触发最低佣金再回退一次修正股数。
        """
        max_amount = cash * self.position_size_pct
        per_share_cost = price * (1 + commission_rate)
        shares = int((max_amount / per_share_cost) // 100) * 100

        # 最低佣金修正: 保证 现金 >= 成交额 + max(成交额*费率, 最低佣金)
        while shares > 0:
            amount = shares * price
            fee = max(amount * commission_rate, min_commission)
            if amount + fee <= max_amount:
                break
            shares -= 100
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
        buy_commission: float = 0.0,
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
            buy_commission=buy_commission,
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
