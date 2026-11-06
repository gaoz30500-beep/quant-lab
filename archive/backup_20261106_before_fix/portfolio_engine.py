"""
组合回测引擎
多股票共享资金池，等额仓位分配

与单标的 BacktestEngine 的差异:
- 多个股票共享一份现金
- 每只股票占用 equity/max_positions 的仓位额度
- 交易日历优先用指数日历, 个股当日无 K 线视为停牌, 挂单顺延到复牌
- 信号/T 日收盘产生, T+1 开盘价成交, 开盘触及涨跌停则不成交
- 净值按各持仓最新收盘价计算 (停牌沿用前收)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable

from config import BacktestConfig
from strategies.base import Strategy, SignalType
from risk import RiskManager, Position
from metrics import Metrics
from engine import Trade, BacktestResult, is_limit_locked


class PortfolioEngine:
    """多股票组合回测引擎"""

    def __init__(
        self,
        data_dict: Dict[str, pd.DataFrame],
        strategy_factory: Callable[[str], Strategy],
        config: BacktestConfig,
        trading_days=None,
    ):
        """
        参数:
            data_dict: {股票代码: 行情DataFrame}
            strategy_factory: 为每只股票创建策略实例的工厂函数
            config: 回测配置
            trading_days: 交易日历 (通常为指数日线的索引); 缺省用个股交易日并集
        """
        self.data_dict = {s: df.copy() for s, df in data_dict.items()}
        self.strategy_factory = strategy_factory
        self.config = config
        self.risk_manager = RiskManager(
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
            trailing_stop_pct=config.trailing_stop_pct,
            trailing_activation_pct=config.trailing_activation_pct,
            position_size_pct=1.0,   # 组合内按槽位分配，不使用整体比例
            max_positions=config.max_positions,
        )

        # 交易日历: 优先用指数日历 (个股缺 bar 即视为停牌), 否则退化为并集
        stock_union = sorted(set().union(*[set(df.index) for df in self.data_dict.values()]))
        lo = min(df.index[0] for df in self.data_dict.values())
        hi = max(df.index[-1] for df in self.data_dict.values())
        if trading_days is not None and len(trading_days) > 0:
            cal = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(trading_days)).unique()))
            cal = cal[(cal >= lo) & (cal <= hi)]
            self.dates = cal if len(cal) > 0 else pd.DatetimeIndex(stock_union)
        else:
            self.dates = pd.DatetimeIndex(stock_union)

        # 价格面板 (净值按收盘价, 停牌日沿用前收; 开盘价仅用于成交)
        self.price_panel = pd.DataFrame({
            s: df['close'] for s, df in self.data_dict.items()
        }).reindex(self.dates).ffill()
        self.open_panel = pd.DataFrame({
            s: df['open'] for s, df in self.data_dict.items()
        }).reindex(self.dates)
        # 涨跌停基准: 最近一次真实成交的收盘价 (停牌日不能取前向填充的当日值)
        self.prev_close_panel = pd.DataFrame({
            s: df['close'] for s, df in self.data_dict.items()
        }).reindex(self.dates).ffill().shift(1)
        # 当日是否有真实 K 线 (False = 停牌)
        self.has_bar = pd.DataFrame({
            s: pd.Series(1.0, index=df.index) for s, df in self.data_dict.items()
        }).reindex(self.dates).fillna(0.0).astype(bool)

        # 状态
        self.cash: float = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_history: List[tuple] = []
        self.daily_positions: List[dict] = []
        # T 日收盘产生的挂单, 下一交易日开盘执行
        self.pending: Dict[str, dict] = {}

    def run(self) -> BacktestResult:
        # 1. 各股票分别生成信号
        signals_dict: Dict[str, pd.Series] = {}
        for symbol, df in self.data_dict.items():
            strategy = self.strategy_factory(symbol)
            signals_dict[symbol] = strategy.generate_signals(df)

        # 2. 逐日驱动
        self.pending = {}
        for date in self.dates:
            # 当日有真实 K 线的股票 (停牌股票不在其中)
            prices = {}
            for symbol, df in self.data_dict.items():
                if date in df.index:
                    prices[symbol] = float(df.loc[date, 'close'])

            # 2.1 先执行上一交易日的挂单 (T+1 开盘价成交)
            self._execute_pending(date)

            # 2.2 更新持仓最高价 (仅在有行情的股票上)
            for symbol, pos in self.positions.items():
                if symbol in prices:
                    pos.update(prices[symbol])

            # 2.3 收盘检查风控 (触发后顺延到下一交易日开盘卖出)
            self._check_risk_exit(date, prices)

            # 2.4 收盘收集策略信号 (下一交易日开盘执行)
            for symbol in self.data_dict.keys():
                if symbol not in prices:
                    continue
                series = signals_dict[symbol]
                if date not in series.index:
                    continue
                self._queue_signal(symbol, series.loc[date])

            # 空仓资金计息 (货币基金/逆回购收益)
            if self.config.cash_yield > 0 and self.cash > 0:
                self.cash *= 1 + self.config.cash_yield / 252

            # 记录净值
            equity = self._mark_to_market(date)
            self.equity_history.append((date, equity))
            self.daily_positions.append({
                'date': date,
                'cash': self.cash,
                'position_value': equity - self.cash,
                'total_value': equity,
                'n_positions': len(self.positions),
                'shares_held': sum(p.shares for p in self.positions.values()),
            })

        # 3. 回测结束强制平仓 (未成交挂单作废)
        self.pending = {}
        last_date = self.dates[-1]
        for symbol in list(self.positions.keys()):
            last_price = float(self.price_panel.loc[last_date, symbol])
            if not np.isnan(last_price):
                self._close_position(symbol, last_date, last_price, '回测结束平仓')

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
            signals=pd.Series(dtype=object),
            indicator_data=pd.DataFrame(),
            config=self.config,
            price_data=self.price_panel,
        )

    # ---------- 内部方法 ----------

    def _mark_to_market(self, date: pd.Timestamp, at_open: bool = False) -> float:
        """组合权益; at_open=True 时持仓按当日开盘价计价 (用于开盘入场定量)"""
        value = self.cash
        for symbol, pos in self.positions.items():
            price = np.nan
            if at_open and bool(self.has_bar.at[date, symbol]):
                price = float(self.open_panel.at[date, symbol])
            if not np.isfinite(price):
                price = float(self.price_panel.loc[date, symbol])
            if not np.isnan(price):
                value += pos.shares * price
        return value

    def _execute_pending(self, date: pd.Timestamp) -> None:
        """执行上一交易日收盘挂单 (T+1 开盘价成交); 停牌/涨跌停则顺延"""
        for symbol in list(self.pending.keys()):
            order = self.pending[symbol]
            side = order['side']

            # 停牌: 当日无真实 K 线, 挂单保留到复牌
            if not bool(self.has_bar.at[date, symbol]):
                continue

            open_price = float(self.open_panel.at[date, symbol])
            prev_close = float(self.prev_close_panel.at[date, symbol])
            if not np.isfinite(open_price) or open_price <= 0:
                continue

            # 持仓状态已与挂单矛盾 (如已强制平仓), 作废
            if side == 'SELL' and symbol not in self.positions:
                del self.pending[symbol]
                continue
            if side == 'BUY' and symbol in self.positions:
                del self.pending[symbol]
                continue

            # 开盘一字板: 涨停买不进 (放弃), 跌停卖不出 (保留挂单)
            if is_limit_locked(symbol, side, open_price, prev_close):
                if side == 'BUY':
                    del self.pending[symbol]
                continue

            del self.pending[symbol]
            if side == 'BUY':
                self._open_position(symbol, date, open_price)
            else:
                self._close_position(symbol, date, open_price, order.get('reason', ''))

    def _queue_signal(self, symbol: str, signal) -> None:
        """收盘信号 → 下一交易日开盘挂单 (同一标的只保留一张单)"""
        if symbol in self.pending:
            return
        if signal == SignalType.BUY and symbol not in self.positions:
            self.pending[symbol] = {'side': 'BUY', 'reason': '策略信号买入'}
        elif signal == SignalType.SELL and symbol in self.positions:
            self.pending[symbol] = {'side': 'SELL', 'reason': '策略信号卖出'}

    def _open_position(self, symbol: str, date, price: float) -> None:
        if symbol in self.positions:
            return
        if len(self.positions) >= self.config.max_positions:
            return

        buy_price = price * (1 + self.config.slippage)

        # 槽位资金 = 总权益 / 最大持仓数，且不超过可用现金
        # 用开盘价计算权益 (成交发生在开盘, 不能用当日收盘价)
        equity = self._mark_to_market(date, at_open=True)
        slot_capital = equity / self.config.max_positions
        budget = min(self.cash, slot_capital)

        shares = self.risk_manager.calc_position_size(
            cash=budget,
            price=buy_price,
            commission_rate=self.config.commission_rate,
            min_commission=self.config.min_commission,
        )
        if shares <= 0:
            return

        amount = shares * buy_price
        commission = self.risk_manager.calc_commission(
            amount, self.config.commission_rate, self.config.min_commission
        )
        total_cost = amount + commission
        if total_cost > self.cash:
            return

        self.cash -= total_cost
        self.positions[symbol] = self.risk_manager.create_position(
            symbol, shares, buy_price, date
        )

    def _close_position(self, symbol: str, date, price: float, reason: str = '') -> None:
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        sell_price = price * (1 - self.config.slippage)

        amount = pos.shares * sell_price
        commission = self.risk_manager.calc_commission(
            amount, self.config.commission_rate, self.config.min_commission
        )
        stamp_tax = (
            self.risk_manager.calc_stamp_tax(amount, self.config.stamp_tax)
            if self.config.enable_stamp_tax else 0.0
        )

        net_amount = amount - commission - stamp_tax
        cost_basis = pos.shares * pos.entry_price
        pnl = net_amount - cost_basis
        pnl_pct = pnl / cost_basis if cost_basis else 0.0

        self.trades.append(Trade(
            symbol=symbol,
            entry_date=pos.entry_date,
            entry_price=pos.entry_price,
            exit_date=date,
            exit_price=sell_price,
            shares=pos.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            holding_days=(date - pos.entry_date).days,
            commission=commission,
            stamp_tax=stamp_tax,
        ))
        self.cash += net_amount
        del self.positions[symbol]

    def _check_risk_exit(self, date, prices: Dict[str, float]) -> None:
        """收盘检查风控 → 挂单到下一交易日开盘卖出"""
        for symbol, pos in self.positions.items():
            if symbol in self.pending:
                continue
            if symbol not in prices:
                continue
            reason = self.risk_manager.check_exit(pos, prices[symbol])
            if reason:
                reason_map = {
                    'stop_loss': '止损',
                    'take_profit': '止盈',
                    'trailing_stop': '移动止损',
                }
                self.pending[symbol] = {
                    'side': 'SELL', 'reason': reason_map.get(reason, reason),
                }
