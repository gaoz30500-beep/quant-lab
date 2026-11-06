"""
长期趋势策略 vs 均值回归策略 对比
验证: 在蓝筹池上, 长期趋势持有能否兼顾收益与回撤
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from config import BacktestConfig
from strategies.ma_cross import MACrossStrategy
from strategies.combined import CombinedStrategy
from strategies.gated import GatedStrategy
from market_gate import MarketGate
from portfolio_engine import PortfolioEngine
from portfolio import DEFAULT_POOL, load_pool, warmup_start, equal_weight_benchmark


def run(data, idx, kind, start=None, end=None):
    """评估区间由 eval_start 控制, 之前的数据只用于指标预热"""
    eval_start = pd.Timestamp(start) if start else None
    eval_end = pd.Timestamp(end) if end else None

    sub = {}
    for s, df in data.items():
        d = df if eval_end is None else df[df.index <= eval_end]
        if len(d) > 0:
            sub[s] = d

    config = BacktestConfig(
        initial_capital=100000,
        max_positions=len(sub),
        stop_loss_pct=0.08,
        take_profit_pct=0.0,      # 0 = 关闭固定止盈, 让趋势跑完
        trailing_stop_pct=0.0,
    )

    def factory(symbol):
        if kind == '组合均值回归':
            core = CombinedStrategy(fast_window=7, slow_window=20,
                                    boll_window=25, boll_std=2.0,
                                    adx_window=14, adx_threshold=35.0)
            sf = True
        else:
            name, fast, slow, gate_only = kind
            core = MACrossStrategy(fast_window=fast, slow_window=slow)
            sf = not gate_only
        g = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
        g.set_index_data(idx)
        return GatedStrategy(core, g, stock_ma_window=20,
                             stock_max_drop=-0.06, enable_stock_filter=sf)

    return PortfolioEngine(sub, factory, config,
                           trading_days=idx.index if idx is not None else None,
                           eval_start=eval_start).run().metrics


def main():
    history_start = warmup_start('20180101')
    data = load_pool(DEFAULT_POOL, start_date=history_start)
    g = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
    idx = g.load_index_data(start_date=history_start)

    variants = [
        ('组合均值回归(基线)', '组合均值回归'),
        ('MA20/120+双闸门', ('MA20/120+双闸门', 20, 120, False)),
        ('MA20/120+仅大盘', ('MA20/120+仅大盘', 20, 120, True)),
        ('MA60/200+双闸门', ('MA60/200+双闸门', 60, 200, False)),
        ('MA60/200+仅大盘', ('MA60/200+仅大盘', 60, 200, True)),
        ('MA5/30+双闸门(旧)', ('MA5/30+双闸门', 5, 30, False)),
    ]

    print("=" * 104)
    print("长期趋势 vs 均值回归 (12只蓝筹, 2018-2026)")
    print("=" * 104)
    print(f"{'策略':<20} {'全样本收益':>11} {'全样本回撤':>11} {'卡玛':>7} "
          f"{'样本内':>10} {'样本外':>10} {'样本外回撤':>11} {'胜率':>7} {'笔数':>5}")
    print("-" * 104)

    for label, kind in variants:
        m_all = run(data, idx, kind)
        m_in = run(data, idx, kind, '2018-01-01', '2022-12-31')
        m_out = run(data, idx, kind, '2023-01-01', '2026-10-31')
        calmar = m_all['calmar_ratio']
        print(f"{label:<20} {m_all['total_return']:>10.2%} {m_all['max_drawdown']:>10.2%} "
              f"{calmar:>7.3f} {m_in['total_return']:>9.2%} {m_out['total_return']:>9.2%} "
              f"{m_out['max_drawdown']:>10.2%} {m_all['win_rate']:>6.1%} {m_all['total_trades']:>5}")

    print("-" * 104)
    # 基准
    # 基准: 与策略同一评估窗口
    bench = equal_weight_benchmark(data, start='2018-01-01')
    print(f"{'等权买入持有':<20} {bench.iloc[-1]-1:>10.2%} {(bench/bench.cummax()-1).min():>10.2%} "
          f"{'':>7} {'':>9} {'':>9} {'':>10}")


if __name__ == '__main__':
    main()
