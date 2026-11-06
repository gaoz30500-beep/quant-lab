"""
组合参数样本外验证
检验 ADX=25 是否只是全样本过拟合
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from config import BacktestConfig
from strategies.combined import CombinedStrategy
from strategies.gated import GatedStrategy
from market_gate import MarketGate
from portfolio_engine import PortfolioEngine
from portfolio import DEFAULT_POOL, load_pool


def run(data, idx, start=None, end=None, adx=25, use_gate=True, use_stock_filter=True):
    sub = {}
    for s, df in data.items():
        d = df
        if start:
            d = d[d.index >= pd.Timestamp(start)]
        if end:
            d = d[d.index <= pd.Timestamp(end)]
        if len(d) > 0:
            sub[s] = d

    config = BacktestConfig(
        initial_capital=100000,
        max_positions=len(sub),
        stop_loss_pct=0.08,
        take_profit_pct=0.20,
        trailing_stop_pct=0.0,
    )

    def factory(symbol):
        combined = CombinedStrategy(fast_window=7, slow_window=20,
                                    boll_window=25, boll_std=2.0,
                                    adx_window=14, adx_threshold=adx)
        if not use_gate:
            return combined
        g = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
        g.set_index_data(idx)
        return GatedStrategy(combined, g, stock_ma_window=20,
                             stock_max_drop=-0.06, enable_stock_filter=use_stock_filter)

    return PortfolioEngine(sub, factory, config).run().metrics


def main():
    data = load_pool(DEFAULT_POOL)
    gate = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
    idx = gate.load_index_data(start_date='20220101')

    print("=" * 78)
    print("样本内/样本外验证")
    print("=" * 78)
    print(f"{'配置':<26} {'区间':<8} {'收益':>9} {'回撤':>8} {'夏普':>8} {'胜率':>7} {'笔数':>5}")
    print("-" * 78)

    splits = [('样本内', '2018-01-01', '2022-12-31'),
              ('样本外', '2023-01-01', '2026-10-31')]

    for label, adx, use_gate, sf in [
        ('ADX25+双闸门', 25, True, True),
        ('ADX35+双闸门', 35, True, True),
        ('ADX25+无闸门', 25, False, False),
    ]:
        for seg, s, e in splits:
            m = run(data, idx, s, e, adx=adx, use_gate=use_gate, use_stock_filter=sf)
            print(f"{label:<26} {seg:<8} {m['total_return']:>8.2%} {m['max_drawdown']:>7.2%} "
                  f"{m['sharpe_ratio']:>8.3f} {m['win_rate']:>6.1%} {m['total_trades']:>5}")
        print("-" * 78)

    # 滚动验证: 4 段
    print("\n" + "=" * 78)
    print("滚动验证 (4段, 检验稳定性)")
    print("=" * 78)
    dates = sorted(set().union(*[set(df.index) for df in data.values()]))
    n = len(dates)
    folds = [(dates[i * n // 4], dates[min((i + 1) * n // 4 - 1, n - 1)]) for i in range(4)]

    print(f"{'段':<6} {'区间':<26} {'收益':>9} {'回撤':>8} {'胜率':>7} {'笔数':>5}")
    print("-" * 78)
    returns = []
    for i, (s, e) in enumerate(folds, 1):
        m = run(data, idx, s, e, adx=25, use_gate=True, use_stock_filter=True)
        returns.append(m['total_return'])
        print(f"{i:<6} {str(pd.Timestamp(s).date()) + ' ~ ' + str(pd.Timestamp(e).date()):<26} "
              f"{m['total_return']:>8.2%} {m['max_drawdown']:>7.2%} "
              f"{m['win_rate']:>6.1%} {m['total_trades']:>5}")
    print("-" * 78)
    pos = sum(1 for r in returns if r > 0)
    print(f"盈利段数: {pos}/4, 平均收益: {sum(returns)/4:.2%}")


if __name__ == '__main__':
    main()
