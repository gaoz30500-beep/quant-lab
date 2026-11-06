"""
趋势参数滚动验证
把 2018-2026 均分为 4 段, 逐段检验稳定性
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from optimize_trend import run
from market_gate import MarketGate
from portfolio import DEFAULT_POOL, load_pool


def main():
    data = load_pool(DEFAULT_POOL, start_date='20180101')
    g = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
    idx = g.load_index_data(start_date='20180101')

    dates = sorted(set().union(*[set(df.index) for df in data.values()]))
    n = len(dates)
    folds = [(dates[i * n // 4], dates[min((i + 1) * n // 4 - 1, n - 1)]) for i in range(4)]

    print("=" * 92)
    print("滚动验证 (4段)")
    print("=" * 92)
    print(f"{'参数':<12} {'段':<4} {'区间':<26} {'收益':>9} {'回撤':>8} {'笔数':>5}")
    print("-" * 92)

    summary = []
    for fast, slow in [(20, 200), (30, 200), (20, 150)]:
        rets = []
        for i, (s, e) in enumerate(folds, 1):
            m = run(data, idx, fast, slow, s, e)
            rets.append(m['total_return'])
            print(f"{f'MA{fast}/{slow}':<12} {i:<4} "
                  f"{str(pd.Timestamp(s).date()) + ' ~ ' + str(pd.Timestamp(e).date()):<26} "
                  f"{m['total_return']:>8.2%} {m['max_drawdown']:>7.2%} {m['total_trades']:>5}")
        pos = sum(1 for r in rets if r > 0)
        summary.append((f'MA{fast}/{slow}', pos, sum(rets) / 4, min(rets), rets))
        print("-" * 92)

    print("\n" + "=" * 92)
    print("汇总")
    print("=" * 92)
    print(f"{'参数':<12} {'盈利段数':>8} {'平均收益':>10} {'最差一段':>10}")
    print("-" * 46)
    for name, pos, avg, worst, _ in summary:
        print(f"{name:<12} {pos}/4{'':<5} {avg:>9.2%} {worst:>9.2%}")


if __name__ == '__main__':
    main()
