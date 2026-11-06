"""
用项目回测框架分析场外基金
把单位净值构造成 OHLC 序列, 走与股票相同的策略与风控
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import akshare as ak
import pandas as pd
import numpy as np

from config import BacktestConfig
from engine import BacktestEngine
from strategies.ma_cross import MACrossStrategy


def load_nav_as_ohlc(code):
    df = ak.fund_open_fund_info_em(symbol=code, indicator='单位净值走势')
    df['净值日期'] = pd.to_datetime(df['净值日期'])
    df = df.sort_values('净值日期').set_index('净值日期')
    close = pd.to_numeric(df['单位净值'], errors='coerce')
    out = pd.DataFrame({
        'open': close, 'high': close, 'low': close, 'close': close,
        'volume': 1.0,
    })
    return out.dropna()


def run(data, fast, slow, capital=10000):
    config = BacktestConfig(
        initial_capital=capital,
        max_positions=1,
        position_size_pct=0.95,
        stop_loss_pct=0.0,          # 关闭固定止损, 由均线决定离场
        take_profit_pct=9.99,       # 关闭固定止盈
        trailing_stop_pct=0.0,
        commission_rate=0.0,        # C类无申购费
        slippage=0.0,
        stamp_tax=0.0,
        min_commission=0.0,
        enable_stamp_tax=False,
        cash_yield=0.0,
    )
    strategy = MACrossStrategy(fast_window=fast, slow_window=slow)
    return BacktestEngine(data, strategy, config).run()


def main():
    code = sys.argv[1] if len(sys.argv) > 1 else '025924'
    data = load_nav_as_ohlc(code)
    first, last = data['close'].iloc[0], data['close'].iloc[-1]
    dd = (data['close'] / data['close'].cummax() - 1).min()

    print("=" * 78)
    print(f"基金 {code} 净值回测 (用项目趋势策略)")
    print("=" * 78)
    print(f"  区间: {data.index[0].date()} ~ {data.index[-1].date()}  ({len(data)} 个交易日)")
    print(f"  净值: {first:.4f} -> {last:.4f}")
    print(f"  买入持有: {last/first-1:+.2%}   期间最大回撤: {dd:.2%}")

    print("\n" + "=" * 78)
    print("参数扫描 (样本内, 需警惕过拟合)")
    print("=" * 78)
    print(f"{'参数':<12} {'策略收益':>10} {'买入持有':>10} {'超额':>9} "
          f"{'最大回撤':>9} {'笔数':>5} {'胜率':>7}")
    print("-" * 78)

    rows = []
    for fast in [5, 10, 20]:
        for slow in [20, 30, 60]:
            if fast >= slow:
                continue
            r = run(data, fast, slow)
            m = r.metrics
            rows.append((fast, slow, m, r))
            print(f"{f'MA{fast}/{slow}':<12} {m['total_return']:>9.2%} "
                  f"{last/first-1:>9.2%} {m['total_return']-(last/first-1):>8.2%} "
                  f"{m['max_drawdown']:>8.2%} {m['total_trades']:>5} {m['win_rate']:>6.1%}")

    best = max(rows, key=lambda x: x[2]['total_return'])
    f_, s_, m_, r_ = best
    print("-" * 78)
    print(f"  最优参数 MA{f_}/{s_}: {m_['total_return']:+.2%}")

    print("\n" + "=" * 78)
    print(f"最优参数 MA{f_}/{s_} 交易明细")
    print("=" * 78)
    if r_.trades:
        for t in r_.trades:
            print(f"  {t.symbol}  {t.entry_date.date()} 买 {t.entry_price:.4f} -> "
                  f"{t.exit_date.date()} 卖 {t.exit_price:.4f}  "
                  f"盈亏 {t.pnl:>+9,.0f} ({t.pnl_pct:+.2%})  原因 {t.exit_reason}")
    else:
        print("  无交易")

    # 当前状态
    c = data['close']
    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()
    print("\n" + "=" * 78)
    print("当前策略状态 (最新交易日)")
    print("=" * 78)
    print(f"  最新净值 {c.iloc[-1]:.4f}   MA20 {ma20.iloc[-1]:.4f}   MA60 {ma60.iloc[-1]:.4f}")
    above20 = c.iloc[-1] > ma20.iloc[-1]
    above60 = c.iloc[-1] > ma60.iloc[-1]
    print(f"  价格{'高于' if above20 else '低于'}MA20, {'高于' if above60 else '低于'}MA60")
    print(f"  策略结论: {'允许持有' if (above20 and above60) else '空仓 / 不买入'}")


if __name__ == '__main__':
    main()
