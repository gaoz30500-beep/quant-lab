"""
闸门参数优化
找到大盘闸门的最优参数，平衡收益与稳健性
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import itertools
import time
import pandas as pd

from config import BacktestConfig
from data_source import DataSource
from strategies.combined import CombinedStrategy
from strategies.gated import GatedStrategy
from market_gate import MarketGate
from engine import BacktestEngine


def optimize_gate(symbol='000001', start_date='20220101'):
    print("=" * 60)
    print(f"大盘闸门参数优化 - {symbol}")
    print("=" * 60)

    ds = DataSource(use_cache=True)
    data = ds.get_stock_daily(symbol, start_date=start_date, adjust='qfq')
    print(f"数据: {len(data)} 条\n")

    config = BacktestConfig(
        initial_capital=100000,
        stop_loss_pct=0.08,
        take_profit_pct=0.20,
        trailing_stop_pct=0.06,
    )

    # 组合策略 (固定优化后参数)
    combined = CombinedStrategy(
        fast_window=7, slow_window=20,
        boll_window=25, boll_std=2.0,
        adx_window=14, adx_threshold=35.0,
    )

    # 加载一次指数数据，所有参数组合复用
    print("加载上证指数数据...")
    temp_gate = MarketGate()
    index_data = temp_gate.load_index_data(start_date=start_date)
    if index_data is None:
        print("⚠ 指数数据加载失败，无法优化闸门")
        return None
    print(f"指数数据: {len(index_data)} 条\n")

    # 闸门参数网格
    param_grid = {
        'ma_window': [20, 30, 40, 60],
        'min_return': [-0.05, -0.03, -0.02, 0.0],
        'use_momentum': [True, False],
    }
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    total = len(combos)
    print(f"参数组合: {total} 组")

    results = []
    t0 = time.perf_counter()

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        gate = MarketGate(
            ma_window=params['ma_window'],
            min_return=params['min_return'],
            use_momentum_filter=params['use_momentum'],
        )
        gate.set_index_data(index_data)  # 复用预加载数据

        gated = GatedStrategy(combined, gate)
        engine = BacktestEngine(data, gated, config)
        result = engine.run()

        m = result.metrics
        # 综合评分: 收益 - 回撤惩罚 (偏向稳健)
        score = m['annual_return'] - 0.5 * m['max_drawdown']

        results.append({
            **params,
            'total_return': m['total_return'],
            'annual_return': m['annual_return'],
            'sharpe': m['sharpe_ratio'],
            'max_drawdown': m['max_drawdown'],
            'win_rate': m['win_rate'],
            'trades': m['total_trades'],
            'score': score,
        })

        if (i + 1) % max(1, total // 5) == 0:
            print(f"  进度 {i+1}/{total}, 已用 {time.perf_counter()-t0:.1f}s")

    df = pd.DataFrame(results)
    df = df.sort_values('score', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("Top 10 闸门参数 (综合评分 = 年化收益 - 0.5×回撤)")
    print("=" * 60)
    print(f"{'ma_win':>7} {'min_ret':>8} {'动量':>6} {'收益':>9} {'夏普':>7} {'回撤':>8} {'胜率':>7} {'交易':>5} {'评分':>8}")
    print("-" * 66)
    for _, r in df.head(10).iterrows():
        mom = '开' if r['use_momentum'] else '关'
        print(f"{r['ma_window']:>7.0f} {r['min_return']:>8.2%} {mom:>6} "
              f"{r['total_return']:>8.2%} {r['sharpe']:>7.3f} {r['max_drawdown']:>7.2%} "
              f"{r['win_rate']:>6.1%} {r['trades']:>5.0f} {r['score']:>8.3f}")

    # 保存
    output_dir = Path(__file__).parent / 'results'
    df.to_csv(output_dir / f'opt_gate_{symbol}.csv', index=False, encoding='utf-8-sig')
    print(f"\n结果已保存: {output_dir / f'opt_gate_{symbol}.csv'}")

    return df


if __name__ == '__main__':
    optimize_gate()
    print("\n闸门参数优化完成!")
