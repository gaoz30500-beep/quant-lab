"""
组合层消融实验
拆解每层风控对组合绩效的真实贡献
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from config import BacktestConfig
from strategies.combined import CombinedStrategy
from strategies.gated import GatedStrategy
from market_gate import MarketGate
from portfolio_engine import PortfolioEngine
from portfolio import DEFAULT_POOL, load_pool, equal_weight_benchmark


def build_variant(kind, idx):
    def factory(symbol):
        combined = CombinedStrategy(
            fast_window=7, slow_window=20,
            boll_window=25, boll_std=2.0,
            adx_window=14, adx_threshold=35.0,
        )
        if kind == 'A_无闸门':
            return combined
        gate = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
        gate.set_index_data(idx)
        if kind == 'B_仅大盘闸门':
            return GatedStrategy(combined, gate, enable_stock_filter=False)
        if kind == 'C_双闸门':
            return GatedStrategy(combined, gate, stock_ma_window=20,
                                 stock_max_drop=-0.06, enable_stock_filter=True)
        raise ValueError(kind)
    return factory


def main():
    data = load_pool(DEFAULT_POOL)
    gate = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
    idx = gate.load_index_data(start_date='20220101')

    config = BacktestConfig(
        initial_capital=100000,
        max_positions=len(DEFAULT_POOL),
        stop_loss_pct=0.08,
        take_profit_pct=0.20,
        trailing_stop_pct=0.0,   # 关闭移动止损 (已验证是亏损源)
    )

    bench = equal_weight_benchmark(data)
    bench_ret = bench.iloc[-1] - 1
    bench_dd = (bench / bench.cummax() - 1).min()

    print("=" * 84)
    print("组合层消融实验 (移动止损已关闭)")
    print("=" * 84)
    print(f"{'版本':<16} {'收益':>10} {'年化':>9} {'夏普':>8} {'回撤':>9} {'胜率':>7} {'笔数':>5} {'盈亏比':>7}")
    print("-" * 84)

    rows = []
    for kind in ['A_无闸门', 'B_仅大盘闸门', 'C_双闸门']:
        r = PortfolioEngine(data, build_variant(kind, idx), config).run()
        m = r.metrics
        print(f"{kind:<16} {m['total_return']:>9.2%} {m['annual_return']:>8.2%} "
              f"{m['sharpe_ratio']:>8.3f} {m['max_drawdown']:>8.2%} "
              f"{m['win_rate']:>6.1%} {m['total_trades']:>5} {m['profit_factor']:>7.2f}")
        rows.append({'version': kind, **{k: m[k] for k in
                     ['total_return', 'annual_return', 'sharpe_ratio',
                      'max_drawdown', 'win_rate', 'total_trades', 'profit_factor']}})

    print("-" * 84)
    print(f"{'等权基准':<16} {bench_ret:>9.2%} {'':>8} {'':>8} {bench_dd:>8.2%}")

    df = pd.DataFrame(rows)
    out = Path(__file__).parent / 'results' / 'portfolio_ablation.csv'
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\n已保存: {out}")

    # ADX 阈值敏感性 (在双闸门版本上)
    print("\n" + "=" * 84)
    print("ADX 阈值敏感性")
    print("=" * 84)
    print(f"{'ADX阈值':>8} {'收益':>10} {'回撤':>9} {'胜率':>7} {'笔数':>5}")
    print("-" * 44)
    for adx_th in [20, 25, 30, 35, 40]:
        def factory(symbol, _th=adx_th):
            combined = CombinedStrategy(fast_window=7, slow_window=20,
                                        boll_window=25, boll_std=2.0,
                                        adx_window=14, adx_threshold=_th)
            g = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
            g.set_index_data(idx)
            return GatedStrategy(combined, g, stock_ma_window=20,
                                 stock_max_drop=-0.06, enable_stock_filter=True)
        m = PortfolioEngine(data, factory, config).run().metrics
        print(f"{adx_th:>8} {m['total_return']:>9.2%} {m['max_drawdown']:>8.2%} "
              f"{m['win_rate']:>6.1%} {m['total_trades']:>5}")


if __name__ == '__main__':
    main()
