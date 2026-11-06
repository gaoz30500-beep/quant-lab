"""
组合策略参数扫描
重点排查: 移动止损/固定止损/止盈 对组合绩效的影响
并用 样本内/样本外 切分验证
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import itertools
import numpy as np
import pandas as pd

from config import BacktestConfig
from data_source import DataSource
from market_gate import MarketGate
from portfolio_engine import PortfolioEngine
from portfolio import build_strategy, DEFAULT_POOL, equal_weight_benchmark


def load_all(pool, start_date='20220101'):
    ds = DataSource(use_cache=True)
    data = {s: ds.get_stock_daily(s, start_date=start_date, adjust='qfq') for s in pool}
    gate = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
    idx = gate.load_index_data(start_date=start_date)
    return data, idx


def run_one(data, idx, start=None, end=None, **kw):
    """用给定风控参数跑一次组合回测, 可限定时间段"""
    sub = {}
    for s, df in data.items():
        d = df
        if start is not None:
            d = d[d.index >= pd.Timestamp(start)]
        if end is not None:
            d = d[d.index <= pd.Timestamp(end)]
        sub[s] = d

    config = BacktestConfig(
        initial_capital=100000,
        max_positions=len(sub),
        **kw,
    )

    def factory(symbol):
        st = build_strategy(symbol)
        st.gate.set_index_data(idx)
        return st

    result = PortfolioEngine(sub, factory, config).run()
    return result


def score_of(m):
    return m['annual_return'] - 0.5 * m['max_drawdown']


def sweep():
    pool = DEFAULT_POOL
    data, idx = load_all(pool)

    grid = {
        'stop_loss_pct': [0.08, 0.10, 0.12],
        'take_profit_pct': [0.20, 0.30],
        'trailing_stop_pct': [0.0, 0.10, 0.15],
    }
    keys = list(grid)
    rows = []
    total = int(np.prod([len(v) for v in grid.values()]))
    print(f"参数组合: {total} 组\n")

    for i, combo in enumerate(itertools.product(*grid.values()), 1):
        p = dict(zip(keys, combo))
        r = run_one(data, idx, **p)
        m = r.metrics
        trades = pd.DataFrame([t.__dict__ for t in r.trades]) if r.trades else pd.DataFrame()
        trail_pnl = (
            trades.loc[trades['exit_reason'] == '移动止损', 'pnl'].sum()
            if not trades.empty else 0.0
        )
        rows.append({
            **p,
            'ret': m['total_return'],
            'annual': m['annual_return'],
            'sharpe': m['sharpe_ratio'],
            'dd': m['max_drawdown'],
            'win': m['win_rate'],
            'trades': m['total_trades'],
            'trail_pnl': trail_pnl,
            'score': score_of(m),
        })

    df = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)

    print("=" * 92)
    print("参数扫描结果 (按 年化-0.5×回撤 排序)")
    print("=" * 92)
    print(f"{'止损':>7} {'止盈':>7} {'移动止损':>9} {'收益':>10} {'年化':>9} {'夏普':>8} {'回撤':>9} {'胜率':>7} {'笔数':>5}")
    print("-" * 92)
    for _, r in df.head(10).iterrows():
        trail = '关闭' if r['trailing_stop_pct'] == 0 else f"{r['trailing_stop_pct']:.0%}"
        print(f"{r['stop_loss_pct']:>6.0%} {r['take_profit_pct']:>7.0%} {trail:>9} "
              f"{r['ret']:>9.2%} {r['annual']:>8.2%} {r['sharpe']:>8.3f} "
              f"{r['dd']:>8.2%} {r['win']:>6.1%} {r['trades']:>5.0f}")

    out = Path(__file__).parent / 'results' / 'opt_portfolio.csv'
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\n已保存: {out}")

    best = df.iloc[0]
    print("\n" + "=" * 92)
    print("最优参数: " + ", ".join(f"{k}={best[k]}" for k in keys))
    print("=" * 92)

    # 样本内/样本外验证
    print("\n样本内外验证 (2022-01 ~ 2024-06 内 / 2024-07 ~ 2026-10 外)")
    print("-" * 92)
    params = {k: best[k] for k in keys}
    for label, s, e in [
        ('样本内', '2022-01-01', '2024-06-30'),
        ('样本外', '2024-07-01', '2026-10-31'),
    ]:
        r = run_one(data, idx, start=s, end=e, **params)
        m = r.metrics
        print(f"  {label}: 收益 {m['total_return']:>8.2%}  回撤 {m['max_drawdown']:>7.2%}  "
              f"夏普 {m['sharpe_ratio']:>6.3f}  胜率 {m['win_rate']:>5.1%}  笔数 {m['total_trades']}")

    return df


if __name__ == '__main__':
    sweep()
