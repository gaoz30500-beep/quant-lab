"""
趋势策略参数网格
筛选标准: 样本内与样本外同时为正, 再比卡玛比率
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from config import BacktestConfig
from strategies.ma_cross import MACrossStrategy
from strategies.gated import GatedStrategy
from market_gate import MarketGate
from portfolio_engine import PortfolioEngine
from portfolio import DEFAULT_POOL, load_pool


def run(data, idx, fast, slow, start=None, end=None):
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
        take_profit_pct=9.99,
        trailing_stop_pct=0.0,
    )

    def factory(symbol):
        core = MACrossStrategy(fast_window=fast, slow_window=slow)
        g = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
        g.set_index_data(idx)
        # 趋势策略下个股暴跌过滤实测无差异, 统一关闭以减少参数
        return GatedStrategy(core, g, enable_stock_filter=False)

    return PortfolioEngine(sub, factory, config,
                           trading_days=idx.index if idx is not None else None).run().metrics


def main():
    data = load_pool(DEFAULT_POOL, start_date='20180101')
    g = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
    idx = g.load_index_data(start_date='20180101')

    print("=" * 104)
    print("趋势参数网格 (双段一致性优先)")
    print("=" * 104)
    print(f"{'参数':<14} {'全样本':>10} {'全回撤':>9} {'样本内':>9} {'样本外':>9} {'外回撤':>9} {'卡玛':>7} {'双段为正':>8} {'笔数':>5}")
    print("-" * 104)

    rows = []
    for fast in [20, 30, 40, 60]:
        for slow in [100, 120, 150, 200]:
            if fast >= slow:
                continue
            m_all = run(data, idx, fast, slow)
            m_in = run(data, idx, fast, slow, '2018-01-01', '2022-12-31')
            m_out = run(data, idx, fast, slow, '2023-01-01', '2026-10-31')
            both_pos = m_in['total_return'] > 0 and m_out['total_return'] > 0
            calmar = m_all['annual_return'] / (abs(m_all['max_drawdown']) + 0.02)
            rows.append({
                'fast': fast, 'slow': slow,
                'ret': m_all['total_return'], 'dd': m_all['max_drawdown'],
                'in': m_in['total_return'], 'out': m_out['total_return'],
                'out_dd': m_out['max_drawdown'], 'calmar': calmar,
                'both_pos': both_pos, 'trades': m_all['total_trades'],
            })
            print(f"{f'MA{fast}/{slow}':<14} {m_all['total_return']:>9.2%} "
                  f"{m_all['max_drawdown']:>8.2%} {m_in['total_return']:>8.2%} "
                  f"{m_out['total_return']:>8.2%} {m_out['max_drawdown']:>8.2%} "
                  f"{calmar:>7.3f} {'是' if both_pos else '否':>8} {m_all['total_trades']:>5}")

    df = pd.DataFrame(rows)
    out = Path(__file__).parent / 'results' / 'opt_trend.csv'
    df.to_csv(out, index=False, encoding='utf-8-sig')

    # 稳健候选: 双段为正, 按最差段收益排序
    cand = df[df['both_pos']].copy()
    cand['min_seg'] = cand[['in', 'out']].min(axis=1)
    cand = cand.sort_values(['min_seg', 'calmar'], ascending=False)
    print("\n" + "=" * 104)
    print("双段均为正的候选 (按较差一段收益排序)")
    print("=" * 104)
    if cand.empty:
        print("无候选: 所有参数都无法在样本内/样本外同时盈利")
    else:
        for _, r in cand.head(5).iterrows():
            print(f"  MA{int(r['fast'])}/{int(r['slow'])}: 样本内 {r['in']:.2%}, "
                  f"样本外 {r['out']:.2%}, 样本外回撤 {r['out_dd']:.2%}, 卡玛 {r['calmar']:.3f}")
    print(f"\n已保存: {out}")


if __name__ == '__main__':
    main()
