"""
回测引擎
核心逻辑: 逐K线驱动，处理信号、持仓、风控、资金
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from config import BacktestConfig
from strategies.base import Strategy, SignalType, Signal
from risk import RiskManager, Position
from metrics import Metrics


# ---------------- 交易规则 ----------------

def price_limit_pct(symbol: str) -> Optional[float]:
    """涨跌停幅度: 创业板/科创板 20%, 主板 10%; ETF/基金/指数无涨跌停限制"""
    code = str(symbol).strip().lower()
    if code.startswith(('sh', 'sz', 'bj')):
        code = code[2:]
    code = code.split('.')[0]
    if code.startswith(('300', '301', '688', '689')):
        return 0.20
    if code.startswith(('000', '001', '002', '003', '600', '601', '603', '605')):
        return 0.10
    return None


def is_limit_locked(symbol: str, side: str, open_price: float, prev_close: float) -> bool:
    """
    开盘价即触及涨跌停 → 当日无法成交

    买入: 开盘价 >= 涨停价 (一字板买不进)
    卖出: 开盘价 <= 跌停价 (一字板卖不出)
    """
    pct = price_limit_pct(symbol)
    if pct is None or not np.isfinite(prev_close) or not np.isfinite(open_price):
        return False
    limit_up = round(float(prev_close) * (1 + pct), 2)
    limit_down = round(float(prev_close) * (1 - pct), 2)
    if side == 'BUY':
        return open_price >= limit_up - 1e-6
    return open_price <= limit_down + 1e-6


@dataclass
class Trade:
    """一笔完整的交易记录"""
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    exit_reason: str
    holding_days: int
    commission: float
    stamp_tax: float


@dataclass
class BacktestResult:
    """回测结果"""
    equity_curve: pd.Series                # 净值曲线
    positions_log: pd.DataFrame            # 每日持仓
    trades: List[Trade]                    # 交易记录
    metrics: Dict                          # 绩效指标
    signals: pd.Series                     # 信号序列
    indicator_data: pd.DataFrame           # 指标数据
    config: BacktestConfig
    price_data: pd.DataFrame = None        # 原始行情数据
    meta: Dict = None                      # 实验快照 (数据版本/配置哈希)

    def summary(self) -> str:
        return Metrics.format_metrics(self.metrics)


class BacktestEngine:
    """回测引擎"""

    def __init__(self, data: pd.DataFrame, strategy: Strategy, config: BacktestConfig,
                 symbol: str = 'STOCK'):
        """
        参数:
            data: 行情数据 (索引为日期)
            strategy: 策略实例
            config: 回测配置
            symbol: 标的代码 (用于涨跌停判断; 非个股代码无涨跌停限制)
        """
        self.data = data.copy()
        self.strategy = strategy
        self.config = config
        self.symbol = symbol
        self.risk_manager = RiskManager(
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
            trailing_stop_pct=config.trailing_stop_pct,
            trailing_activation_pct=config.trailing_activation_pct,
            position_size_pct=config.position_size_pct,
            max_positions=config.max_positions,
        )

        # 状态
        self.cash: float = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_history: List[Tuple[pd.Timestamp, float]] = []
        self.daily_positions: List[Dict] = []
        # T 日收盘产生的挂单, 下一交易日开盘执行
        self.pending: Optional[Dict] = None

    def run(self) -> BacktestResult:
        """运行回测"""
        # 1. 生成信号
        signals = self.strategy.generate_signals(self.data)
        indicator_data = self.strategy.get_indicator_data(self.data)

        # 2. 逐K线回测
        dates = self.data.index
        prev_closes = self.data['close'].shift(1)
        self.pending = None

        for i, date in enumerate(dates):
            row = self.data.loc[date]
            close_price = float(row['close'])
            open_price = float(row['open']) if 'open' in row.index else close_price

            # 2.1 先执行上一交易日收盘时挂出的单 (T+1 开盘成交)
            self._execute_pending(date, open_price, float(prev_closes.iloc[i]))

            # 2.2 更新持仓最高价 (收盘价)
            for pos in self.positions.values():
                pos.update(close_price)

            # 2.3 收盘检查风控 (触发后顺延到下一交易日开盘卖出)
            self._check_risk_exit(date, close_price)

            # 2.4 收盘处理策略信号 (下一交易日开盘执行)
            self._queue_signal(signals.iloc[i])

            # 记录净值
            total_value = self.cash + sum(
                p.shares * close_price for p in self.positions.values()
            )
            self.equity_history.append((date, total_value))

            # 记录每日持仓
            self.daily_positions.append({
                'date': date,
                'cash': self.cash,
                'position_value': sum(p.shares * close_price for p in self.positions.values()),
                'total_value': total_value,
                'shares_held': sum(p.shares for p in self.positions.values()),
            })

        # 3. 强制平仓 (回测结束)
        self.pending = None
        self._force_close_all(dates[-1], float(self.data['close'].iloc[-1]))

        # 4. 计算绩效
        equity_curve = pd.Series(
            [v for _, v in self.equity_history],
            index=[d for d, _ in self.equity_history],
            name='equity',
        )
        metrics = Metrics.calc_all(
            equity_curve,
            [t.__dict__ for t in self.trades],
            risk_free_rate=self.config.risk_free_rate,
        )

        positions_log = pd.DataFrame(self.daily_positions).set_index('date')

        return BacktestResult(
            equity_curve=equity_curve,
            positions_log=positions_log,
            trades=self.trades,
            metrics=metrics,
            signals=signals,
            indicator_data=indicator_data,
            config=self.config,
            price_data=self.data,
        )

    def _execute_pending(self, date: pd.Timestamp, open_price: float, prev_close: float) -> None:
        """执行上一交易日收盘挂单 (T+1 开盘价成交)"""
        if self.pending is None:
            return

        side = self.pending['side']
        reason = self.pending.get('reason', '')

        # 无有效开盘价 (如停牌) → 顺延
        if not np.isfinite(open_price) or open_price <= 0:
            return

        # 开盘一字涨停买不进 / 一字跌停卖不出
        if is_limit_locked(self.symbol, side, open_price, prev_close):
            if side == 'BUY':
                self.pending = None   # 买不进, 放弃这次机会
            return                    # 卖不出, 持仓与挂单都保留

        self.pending = None
        if side == 'BUY':
            self._open_position(self.symbol, date, open_price)
        else:
            self._close_position(self.symbol, date, open_price, reason)

    def _queue_signal(self, signal: SignalType) -> None:
        """收盘信号 → 下一交易日开盘挂单"""
        if self.pending is not None:
            return
        if signal == SignalType.BUY and self.symbol not in self.positions:
            self.pending = {'side': 'BUY', 'reason': '策略信号买入'}
        elif signal == SignalType.SELL and self.symbol in self.positions:
            self.pending = {'side': 'SELL', 'reason': '策略信号卖出'}

    def _open_position(self, symbol: str, date: pd.Timestamp, price: float) -> None:
        """开仓"""
        if symbol in self.positions:
            return  # 已持仓，不加仓

        if len(self.positions) >= self.config.max_positions:
            return  # 超过最大持仓数

        # 计算滑点后的买入价
        buy_price = price * (1 + self.config.slippage)

        # 计算可买股数
        shares = self.risk_manager.calc_position_size(
            cash=self.cash,
            price=buy_price,
            commission_rate=self.config.commission_rate,
            min_commission=self.config.min_commission,
        )

        if shares <= 0:
            return

        # 计算费用
        amount = shares * buy_price
        commission = self.risk_manager.calc_commission(
            amount, self.config.commission_rate, self.config.min_commission
        )
        total_cost = amount + commission

        if total_cost > self.cash:
            return

        # 执行买入
        self.cash -= total_cost
        position = self.risk_manager.create_position(symbol, shares, buy_price, date)
        self.positions[symbol] = position

    def _close_position(
        self,
        symbol: str,
        date: pd.Timestamp,
        price: float,
        reason: str = '',
    ) -> None:
        """平仓"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        # 卖出价 (滑点)
        sell_price = price * (1 - self.config.slippage)

        amount = position.shares * sell_price
        commission = self.risk_manager.calc_commission(
            amount, self.config.commission_rate, self.config.min_commission
        )
        stamp_tax = (
            self.risk_manager.calc_stamp_tax(amount, self.config.stamp_tax)
            if self.config.enable_stamp_tax else 0.0
        )

        net_amount = amount - commission - stamp_tax
        pnl = net_amount - (position.shares * position.entry_price)
        pnl_pct = pnl / (position.shares * position.entry_price)

        holding_days = (date - position.entry_date).days

        trade = Trade(
            symbol=symbol,
            entry_date=position.entry_date,
            entry_price=position.entry_price,
            exit_date=date,
            exit_price=sell_price,
            shares=position.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            holding_days=holding_days,
            commission=commission,
            stamp_tax=stamp_tax,
        )
        self.trades.append(trade)
        self.cash += net_amount
        del self.positions[symbol]

    def _check_risk_exit(self, date: pd.Timestamp, price: float) -> None:
        """收盘检查风控平仓 → 挂单到下一交易日开盘卖出"""
        if self.pending is not None:
            return
        if self.symbol not in self.positions:
            return

        position = self.positions[self.symbol]
        reason = self.risk_manager.check_exit(position, price)
        if reason:
            reason_map = {
                'stop_loss': '止损',
                'take_profit': '止盈',
                'trailing_stop': '移动止损',
            }
            self.pending = {'side': 'SELL', 'reason': reason_map.get(reason, reason)}

    def _force_close_all(self, date: pd.Timestamp, price: float) -> None:
        """回测结束，强制平仓"""
        symbols = list(self.positions.keys())
        for symbol in symbols:
            self._close_position(symbol, date, price, reason='回测结束平仓')
