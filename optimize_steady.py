"""
稳中求高收益 - 风控参数扫描
目标函数: 卡玛比率 (年化收益 / 最大回撤), 兼顾收益与稳定性
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import itertools
import numpy as np
import pandas as pd

from config import BacktestConfig
from strategies.combined import CombinedStrategy
from strategies.gated import GatedStrategy
from market_gate import MarketGate
from portfolio_engine import PortfolioEngine
from portfolio import DEFAULT_POOL, load_pool, warmup_start


def run(data, idx, max_positions, take_profit, trailing, activation,
        start=None, end=None):
    """评估区间由 eval_start 控制, 之前的数据只用于指标预热"""
    eval_start = pd.Timestamp(start) if start else None
    eval_end = pd.Timestamp(end) if end else None

    sub = {}
    for s, df in data.items():
        d = df if eval_end is None else df[df.index <= eval_end]
        if len(d) > 0:
            sub[s] = d
    if len(sub) < 2:
        return None

    config = BacktestConfig(
        initial_capital=100000,
        max_positions=min(max_positions, len(sub)),
        stop_loss_pct=0.08,
        take_profit_pct=take_profit,
        trailing_stop_pct=trailing,
        trailing_activation_pct=activation,
    )

    def factory(symbol):
        combined = CombinedStrategy(fast_window=7, slow_window=20,
                                    boll_window=25, boll_std=2.0,
                                    adx_window=14, adx_threshold=35.0)
        g = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
        g.set_index_data(idx)
        return GatedStrategy(combined, g, stock_ma_window=20,
                             stock_max_drop=-0.06, enable_stock_filter=True)

    result = PortfolioEngine(sub, factory, config,
                             trading_days=idx.index if idx is not None else None,
                             eval_start=eval_start).run()
    return result


def calmar(m):
    """卡玛比率直接用 metrics 的口径 (年化 / |最大回撤|), 不再叠加经验常数"""
    return m['calmar_ratio']


def main():
    history_start = warmup_start('20180101')
    data = load_pool(DEFAULT_POOL, start_date=history_start)
    g = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
    idx = g.load_index_data(start_date=history_start)

    combos = []
    for mp in [6, 8, 12]:
        for tp in [0.20, 0.30, 0.0]:   # 0 = 关闭固定止盈
            combos.append((mp, tp, 0.0, 0.0))
            for ts in [0.08, 0.12]:
                for act in [0.10, 0.20]:
                    combos.append((mp, tp, ts, act))

    print(f"参数组合: {len(combos)} 组, 全样本 2018-2026")
    print(f"股票池: {len(data)} 只\n")

    rows = []
    for i, (mp, tp, ts, act) in enumerate(combos, 1):
        r = run(data, idx, mp, tp, ts, act)
        if r is None:
            continue
        m = r.metrics
        rows.append({
            'max_pos': mp,
            'take_profit': tp,
            'trailing': ts,
            'activation': act,
            'ret': m['total_return'],
            'annual': m['annual_return'],
            'dd': m['max_drawdown'],
            'sharpe': m['sharpe_ratio'],
            'win': m['win_rate'],
            'pf': m['profit_factor'],
            'trades': m['total_trades'],
            'calmar': calmar(m),
        })
        if i % 10 == 0:
            print(f"  已完成 {i}/{len(combos)}")

    df = pd.DataFrame(rows).sort_values('calmar', ascending=False).reset_index(drop=True)
    print("\n" + "=" * 100)
    print("按卡玛比率排序 (前10)")
    print("=" * 100)
    print(f"{'持仓数':>6} {'止盈':>7} {'移动止损':>8} {'启用阈值':>8} {'收益':>10} {'年化':>9} {'回撤':>9} {'卡玛':>7} {'胜率':>7} {'盈亏比':>7} {'笔数':>5}")
    print("-" * 100)
    for _, r in df.head(10).iterrows():
        tp_txt = '关闭' if r['take_profit'] <= 0 else f"{r['take_profit']:.0%}"
        ts_txt = '关闭' if r['trailing'] == 0 else f"{r['trailing']:.0%}"
        act_txt = '—' if r['trailing'] == 0 else f"{r['activation']:.0%}"
        print(f"{r['max_pos']:>6.0f} {tp_txt:>7} {ts_txt:>8} {act_txt:>8} "
              f"{r['ret']:>9.2%} {r['annual']:>8.2%} {r['dd']:>8.2%} "
              f"{r['calmar']:>7.3f} {r['win']:>6.1%} {r['pf']:>7.2f} {r['trades']:>5.0f}")

    out = Path(__file__).parent / 'results' / 'opt_steady.csv'
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\n已保存: {out}")

    # 前3名做样本外验证
    print("\n" + "=" * 100)
    print("前3名 样本内外验证")
    print("=" * 100)
    print(f"{'参数':<30} {'样本内(18-22)':>16} {'样本外(23-26)':>16} {'样本外回撤':>12}")
    print("-" * 100)
    for _, r in df.head(3).iterrows():
        ms = {}
        for label, s, e in [('内', '2018-01-01', '2022-12-31'),
                            ('外', '2023-01-01', '2026-10-31')]:
            rr = run(data, idx, int(r['max_pos']), r['take_profit'],
                     r['trailing'], r['activation'], s, e)
            ms[label] = rr.metrics
        tp_txt = '关闭' if r['take_profit'] <= 0 else f"{r['take_profit']:.0%}"
        label = f"持仓{int(r['max_pos'])}/止盈{tp_txt}/移{int(r['trailing']*100)}%"
        print(f"{label:<30} {ms['内']['total_return']:>15.2%} "
              f"{ms['外']['total_return']:>15.2%} {ms['外']['max_drawdown']:>11.2%}")


if __name__ == '__main__':
    main()
