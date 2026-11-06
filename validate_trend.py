"""
趋势参数滚动验证
把 2018-2026 均分为 4 段, 逐段检验稳定性

口径约定:
- 数据从 HISTORY_START 加载, 每段评估时指标已在段外预热 (不会段首空仓数月)
- 段间不可直接相加总收益, 汇总用"四段复合收益"与"各段年化"
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from optimize_trend import run, HISTORY_START
from market_gate import MarketGate
from portfolio import DEFAULT_POOL, load_pool


def main():
    print(f"加载历史行情 ({HISTORY_START} 起, 供各段预热)...")
    data = load_pool(DEFAULT_POOL, start_date=HISTORY_START)
    g = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
    idx = g.load_index_data(start_date=HISTORY_START)
    if idx is None:
        print("指数数据不可用, 退出")
        return

    # 段边界取自评估区间 (2018-01 起), 与 HISTORY_START 的预热数据无关
    eval_dates = [d for d in idx.index if d >= pd.Timestamp('2018-01-01')]
    n = len(eval_dates)
    folds = [(eval_dates[i * n // 4], eval_dates[min((i + 1) * n // 4 - 1, n - 1)])
             for i in range(4)]

    print("=" * 100)
    print("滚动验证 (4段, 预热在段外)")
    print("=" * 100)
    print(f"{'参数':<12} {'段':<4} {'区间':<26} {'收益':>9} {'年化':>8} {'回撤':>8} {'笔数':>5}")
    print("-" * 100)

    summary = []
    for fast, slow in [(20, 200), (30, 200), (20, 150)]:
        rets = []
        anns = []
        for i, (s, e) in enumerate(folds, 1):
            m = run(data, idx, fast, slow, s, e)
            rets.append(m['total_return'])
            anns.append(m['annual_return'])
            label = f'MA{fast}/{slow}'
            span = f"{pd.Timestamp(s).date()} ~ {pd.Timestamp(e).date()}"
            print(f"{label:<12} {i:<4} {span:<26} "
                  f"{m['total_return']:>8.2%} {m['annual_return']:>7.2%} "
                  f"{m['max_drawdown']:>7.2%} {m['total_trades']:>5}")
        pos = sum(1 for r in rets if r > 0)
        compound = float(np.prod([1 + r for r in rets]) - 1)
        summary.append((f'MA{fast}/{slow}', pos, compound, min(rets), min(anns)))
        print("-" * 100)

    print("\n" + "=" * 100)
    print("汇总")
    print("=" * 100)
    print(f"{'参数':<12} {'盈利段数':>8} {'四段复合收益':>12} {'最差一段':>10} {'最差段年化':>10}")
    print("-" * 56)
    for name, pos, compound, worst, worst_ann in summary:
        print(f"{name:<12} {pos}/4{'':<5} {compound:>11.2%} {worst:>9.2%} {worst_ann:>9.2%}")

    # 段收益序列本身也是样本, 单看"全部盈利"不足以说明稳健
    print("\n注: 4 段复合收益不等于 4 段总收益之和; 段内年化已按各自区间长度折算。")


if __name__ == '__main__':
    main()
