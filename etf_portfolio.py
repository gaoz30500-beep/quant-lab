"""
ETF 组合回测
ETF 免印花税, 一手 100 份, 小资金友好
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from config import BacktestConfig
from data_source import DataSource
from strategies.ma_cross import MACrossStrategy
from strategies.gated import GatedStrategy
from market_gate import MarketGate
from portfolio_engine import PortfolioEngine
from portfolio import warmup_start, equal_weight_benchmark

ETF_POOL = {
    '510300': '沪深300ETF',
    '510500': '中证500ETF',
    '159915': '创业板ETF',
    '512880': '证券ETF',
    '512690': '酒ETF',
    '515030': '新能源车ETF',
}


def load_etf_pool(start_date='20180101'):
    ds = DataSource()
    data = {}
    for code, name in ETF_POOL.items():
        try:
            df = ds.get_etf_daily(code, start_date=start_date)
            data[code] = df
            print(f"  {code} {name}: {len(df)} 条, {df.index[0].date()} ~ {df.index[-1].date()}")
        except Exception as e:
            print(f"  {code} {name}: 跳过 ({e})")
    return data


def run(data, idx, start=None, end=None, capital=100000):
    """评估区间由 eval_start 控制, 之前的数据只用于指标预热"""
    eval_start = pd.Timestamp(start) if start else None
    eval_end = pd.Timestamp(end) if end else None

    sub = {}
    for s, df in data.items():
        d = df if eval_end is None else df[df.index <= eval_end]
        if len(d) > 0:
            sub[s] = d

    config = BacktestConfig(
        initial_capital=capital,
        max_positions=len(sub),
        stop_loss_pct=0.08,
        take_profit_pct=0.0,      # 0 = 关闭固定止盈
        trailing_stop_pct=0.0,
        cash_yield=0.02,
        enable_stamp_tax=False,   # ETF 免印花税
    )

    def factory(symbol):
        core = MACrossStrategy(fast_window=20, slow_window=150)
        g = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
        g.set_index_data(idx)
        return GatedStrategy(core, g, enable_stock_filter=False)

    return PortfolioEngine(sub, factory, config,
                           trading_days=idx.index if idx is not None else None,
                           eval_start=eval_start).run()


def main():
    history_start = warmup_start('20180101')
    print("=" * 78)
    print("加载 ETF 数据")
    print("=" * 78)
    data = load_etf_pool(history_start)
    if len(data) < 2:
        print("数据不足, 退出")
        return

    g = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
    idx = g.load_index_data(start_date=history_start)

    print("\n" + "=" * 78)
    print("一手成本")
    print("=" * 78)
    for s, df in data.items():
        print(f"  {s} {ETF_POOL[s]}: {float(df['close'].iloc[-1]) * 100:>8,.0f} 元/手")

    r = run(data, idx, start='2018-01-01')
    m = r.metrics

    eval_end = r.equity_curve.index[-1]
    bench = equal_weight_benchmark(data, start='2018-01-01', end=eval_end)

    print("\n" + "=" * 78)
    print("ETF 组合回测 (2018-2026, 趋势策略 MA20/150)")
    print("=" * 78)
    print(f"  策略总收益:   {m['total_return']:>10.2%}")
    print(f"  等权基准收益: {bench.iloc[-1] - 1:>10.2%}")
    print(f"  年化收益:     {m['annual_return']:>10.2%}")
    print(f"  最大回撤:     {m['max_drawdown']:>10.2%}")
    print(f"  基准最大回撤: {(bench / bench.cummax() - 1).min():>10.2%}")
    print(f"  夏普:         {m['sharpe_ratio']:>10.3f}")
    print(f"  卡玛:         {m['calmar_ratio']:>10.3f}")
    print(f"  胜率:         {m['win_rate']:>10.1%}")
    print(f"  盈亏比:       {m['profit_factor']:>10.2f}")
    print(f"  交易笔数:     {m['total_trades']:>10}")

    if r.trades:
        t = pd.DataFrame([x.__dict__ for x in r.trades])
        print("\n  各 ETF 贡献:")
        for s, g in t.groupby('symbol'):
            print(f"    {s} {ETF_POOL.get(s, '')}: {len(g):>3} 笔, "
                  f"胜率 {(g['pnl'] > 0).mean():>5.1%}, 盈亏 {g['pnl'].sum():>10,.0f}")

    print("\n" + "=" * 78)
    print("样本内外")
    print("=" * 78)
    for label, s, e in [('样本内 2018-2022', '2018-01-01', '2022-12-31'),
                        ('样本外 2023-2026', '2023-01-01', '2026-10-31')]:
        rr = run(data, idx, s, e)
        print(f"  {label}: 收益 {rr.metrics['total_return']:>8.2%}, "
              f"回撤 {rr.metrics['max_drawdown']:>7.2%}, "
              f"笔数 {rr.metrics['total_trades']:>3}")

    # 小资金档位
    print("\n" + "=" * 78)
    print("不同本金 (ETF 一手约 400~2000 元, 门槛远低于股票)")
    print("=" * 78)
    for cap in [1000, 3000, 5000, 10000, 50000]:
        mm = run(data, idx, start='2018-01-01', capital=cap).metrics
        print(f"  {cap:>7,} 元: 收益 {mm['total_return']:>7.2%}, "
              f"净赚 {cap * mm['total_return']:>8,.0f} 元, "
              f"笔数 {mm['total_trades']:>3}")


if __name__ == '__main__':
    main()
