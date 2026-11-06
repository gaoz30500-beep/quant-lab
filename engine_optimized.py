"""
优化版回测引擎
- 并行计算支持
- 内存优化
- 批量处理
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from config import BacktestConfig
from strategies.base import Strategy, SignalType, Signal
from risk import RiskManager, Position
from metrics import Metrics


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
    execution_time: float = 0.0           # 执行时间

    def summary(self) -> str:
        return Metrics.format_metrics(self.metrics)


class OptimizedBacktestEngine:
    """优化版回测引擎 - 支持并行计算和内存优化"""

    def __init__(self, data: pd.DataFrame, strategy: Strategy, config: BacktestConfig):
        """
        参数:
            data: 行情数据 (索引为日期)
            strategy: 策略实例
            config: 回测配置
        """
        self.data = data.copy()
        self.strategy = strategy
        self.config = config
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

    def run(self, parallel: bool = True, chunk_size: int = 100) -> BacktestResult:
        """运行回测 - 支持并行计算"""
        start_time = time.perf_counter()
        
        # 1. 生成信号
        signals = self.strategy.generate_signals(self.data)
        indicator_data = self.strategy.get_indicator_data(self.data)

        # 2. 优化处理：分块并行回测
        if parallel and len(self.data) > chunk_size:
            result = self._run_parallel_backtest(signals, chunk_size)
        else:
            result = self._run_sequential_backtest(signals)

        # 3. 强制平仓
        self._force_close_all(self.data.index[-1], self.data['close'].iloc[-1])

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
        execution_time = time.perf_counter() - start_time

        return BacktestResult(
            equity_curve=equity_curve,
            positions_log=positions_log,
            trades=self.trades,
            metrics=metrics,
            signals=signals,
            indicator_data=indicator_data,
            config=self.config,
            price_data=self.data,
            execution_time=execution_time,
        )

    def _run_sequential_backtest(self, signals: pd.Series) -> None:
        """顺序回测（原始方法）"""
        dates = self.data.index
        for i, date in enumerate(dates):
            row = self.data.loc[date]
            close_price = row['close']
            symbol = 'STOCK'

            # 更新持仓最高价
            for pos in self.positions.values():
                pos.update(close_price)

            # 检查风控平仓
            self._check_risk_exit(date, close_price)

            # 处理策略信号
            signal = signals.iloc[i]
            self._process_signal(signal, date, close_price)

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

    def _run_parallel_backtest(self, signals: pd.Series, chunk_size: int) -> None:
        """并行回测 - 分块处理"""
        dates = self.data.index
        n_chunks = len(dates) // chunk_size + (1 if len(dates) % chunk_size else 0)
        
        # 准备分块数据
        chunks = []
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(dates))
            chunk_dates = dates[start_idx:end_idx]
            chunk_signals = signals.iloc[start_idx:end_idx]
            chunk_data = self.data.iloc[start_idx:end_idx]
            chunks.append((chunk_dates, chunk_signals, chunk_data, i))

        # 使用进程池并行处理
        with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), n_chunks)) as executor:
            futures = []
            for chunk in chunks:
                future = executor.submit(self._process_chunk, *chunk)
                futures.append(future)

            # 收集结果
            for future in as_completed(futures):
                chunk_result = future.result()
                self._merge_chunk_result(chunk_result)

    def _process_chunk(self, chunk_dates: pd.DatetimeIndex, 
                      chunk_signals: pd.Series, chunk_data: pd.DataFrame,
                      chunk_id: int) -> Dict:
        """处理一个数据块"""
        chunk_result = {
            'equity_history': [],
            'daily_positions': [],
            'trades': [],
            'positions': {},
            'cash': self.cash if chunk_id == 0 else 0.0,
        }

        # 复制当前状态（第一个块从初始状态开始）
        if chunk_id == 0:
            chunk_result['cash'] = self.cash
            chunk_result['positions'] = {}
        else:
            # 后续块从前一个块的最终状态开始
            chunk_result['cash'] = 0.0  # 将在merge时设置
            chunk_result['positions'] = {}

        # 处理块内数据
        for i, date in enumerate(chunk_dates):
            row = chunk_data.iloc[i]
            close_price = row['close']
            symbol = 'STOCK'

            # 更新持仓最高价
            for pos in chunk_result['positions'].values():
                pos.update(close_price)

            # 检查风控平仓
            to_close = []
            for symbol, position in chunk_result['positions'].items():
                reason = self.risk_manager.check_exit(position, close_price)
                if reason:
                    reason_map = {
                        'stop_loss': '止损',
                        'take_profit': '止盈',
                        'trailing_stop': '移动止损',
                    }
                    to_close.append((symbol, reason_map.get(reason, reason)))

            for symbol, reason in to_close:
                trade = self._close_position_chunk(
                    chunk_result, symbol, date, close_price, reason
                )
                if trade:
                    chunk_result['trades'].append(trade)

            # 处理策略信号
            signal = chunk_signals.iloc[i]
            if signal == SignalType.BUY:
                self._open_position_chunk(chunk_result, symbol, date, close_price)
            elif signal == SignalType.SELL:
                trade = self._close_position_chunk(
                    chunk_result, symbol, date, close_price, '策略信号卖出'
                )
                if trade:
                    chunk_result['trades'].append(trade)

            # 记录净值
            total_value = chunk_result['cash'] + sum(
                p.shares * close_price for p in chunk_result['positions'].values()
            )
            chunk_result['equity_history'].append((date, total_value))

            # 记录每日持仓
            chunk_result['daily_positions'].append({
                'date': date,
                'cash': chunk_result['cash'],
                'position_value': sum(p.shares * close_price for p in chunk_result['positions'].values()),
                'total_value': total_value,
                'shares_held': sum(p.shares for p in chunk_result['positions'].values()),
            })

        return chunk_result

    def _merge_chunk_result(self, chunk_result: Dict) -> None:
        """合并块结果到主结果"""
        # 更新现金
        self.cash = chunk_result['cash']
        
        # 更新持仓
        self.positions.update(chunk_result['positions'])
        
        # 更新交易记录
        self.trades.extend(chunk_result['trades'])
        
        # 更新净值历史
        self.equity_history.extend(chunk_result['equity_history'])
        
        # 更新持仓记录
        self.daily_positions.extend(chunk_result['daily_positions'])

    def _open_position_chunk(self, chunk_result: Dict, symbol: str, 
                           date: pd.Timestamp, price: float) -> None:
        """在块处理中开仓"""
        if symbol in chunk_result['positions']:
            return  # 已持仓，不加仓

        if len(chunk_result['positions']) >= self.config.max_positions:
            return  # 超过最大持仓数

        # 计算滑点后的买入价
        buy_price = price * (1 + self.config.slippage)

        # 计算可买股数
        shares = self.risk_manager.calc_position_size(
            cash=chunk_result['cash'],
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

        if total_cost > chunk_result['cash']:
            return

        # 执行买入
        chunk_result['cash'] -= total_cost
        position = self.risk_manager.create_position(symbol, shares, buy_price, date)
        chunk_result['positions'][symbol] = position

    def _close_position_chunk(self, chunk_result: Dict, symbol: str,
                            date: pd.Timestamp, price: float, reason: str) -> Optional[Trade]:
        """在块处理中平仓"""
        if symbol not in chunk_result['positions']:
            return None

        position = chunk_result['positions'][symbol]
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
        
        chunk_result['cash'] += net_amount
        del chunk_result['positions'][symbol]
        return trade

    def _process_signal(self, signal: SignalType, date: pd.Timestamp, price: float) -> None:
        """处理交易信号 - 顺序处理使用"""
        symbol = 'STOCK'

        if signal == SignalType.BUY:
            self._open_position(symbol, date, price)
        elif signal == SignalType.SELL:
            self._close_position(symbol, date, price, reason='策略信号卖出')

    def _open_position(self, symbol: str, date: pd.Timestamp, price: float) -> None:
        """开仓 - 顺序处理使用"""
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

    def _close_position(self, symbol: str, date: pd.Timestamp, 
                      price: float, reason: str = '') -> None:
        """平仓 - 顺序处理使用"""
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
        """检查风控平仓"""
        to_close = []
        for symbol, position in self.positions.items():
            reason = self.risk_manager.check_exit(position, price)
            if reason:
                reason_map = {
                    'stop_loss': '止损',
                    'take_profit': '止盈',
                    'trailing_stop': '移动止损',
                }
                to_close.append((symbol, reason_map.get(reason, reason)))

        for symbol, reason in to_close:
            self._close_position(symbol, date, price, reason)

    def _force_close_all(self, date: pd.Timestamp, price: float) -> None:
        """回测结束，强制平仓"""
        symbols = list(self.positions.keys())
        for symbol in symbols:
            self._close_position(symbol, date, price, reason='回测结束平仓')