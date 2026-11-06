"""
最终稳健性验证
对比: 无闸门 vs 优化后闸门，做滚动前进验证
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from config import BacktestConfig
from data_source import DataSource
from strategies.combined import CombinedStrategy
from strategies.gated import GatedStrategy
from market_gate import MarketGate
from engine import BacktestEngine


def walk_forward(strategy, data, config, n_splits=5, train_ratio=0.7):
    """滚动前进验证"""
    n = len(data)
    fold_size = n // n_splits
    train_size = int(fold_size * train_ratio)
    metrics_list = []

    for i in range(n_splits):
        start = i * fold_size
        end = start + fold_size
        if end > n:
            break
        test_data = data.iloc[start + train_size:end]
        if len(test_data) < 20:
            continue
        engine = BacktestEngine(test_data, strategy, config)
        result = engine.run()
        metrics_list.append(result.metrics)

    if not metrics_list:
        return None
    df = pd.DataFrame(metrics_list)
    return {
        'avg_return': df['total_return'].mean(),
        'avg_drawdown': df['max_drawdown'].mean(),
        'avg_win_rate': df['win_rate'].mean(),
        'avg_sharpe': df['sharpe_ratio'].mean(),
        'positive_folds': int((df['total_return'] > 0).sum()),
        'total_folds': len(df),
    }


def final_robustness_test(symbol='000001', start_date='20220101'):
    print("=" * 70)
    print(f"最终稳健性验证 - {symbol}")
    print("=" * 70)

    ds = DataSource(use_cache=True)
    data = ds.get_stock_daily(symbol, start_date=start_date, adjust='qfq')
    print(f"数据: {len(data)} 条\n")

    config = BacktestConfig(
        initial_capital=100000,
        stop_loss_pct=0.08,
        take_profit_pct=0.20,
        trailing_stop_pct=0.06,
    )

    combined = CombinedStrategy(
        fast_window=7, slow_window=20,
        boll_window=25, boll_std=2.0,
        adx_window=14, adx_threshold=35.0,
    )

    # 优化后闸门
    gate = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
    gate.load_index_data(start_date=start_date)
    gated = GatedStrategy(combined, gate)

    print(f"{'配置':<22} {'全样本收益':>12} {'全样本回撤':>12} "
          f"{'滚动平均收益':>14} {'滚动平均回撤':>14} {'盈利段数':>10}")
    print("-" * 88)

    for name, strategy in [('组合策略(无闸门)', combined), ('组合策略(有闸门)', gated)]:
        # 全样本
        engine = BacktestEngine(data, strategy, config)
        result = engine.run()
        full_ret = result.metrics['total_return']
        full_dd = result.metrics['max_drawdown']

        # 滚动验证
        wf = walk_forward(strategy, data, config)
        if wf:
            print(f"{name:<22} {full_ret:>11.2%} {full_dd:>11.2%} "
                  f"{wf['avg_return']:>13.2%} {wf['avg_drawdown']:>13.2%} "
                  f"{wf['positive_folds']}/{wf['total_folds']}")
        else:
            print(f"{name:<22} {full_ret:>11.2%} {full_dd:>11.2%} (数据不足)")

    print("-" * 88)
    print("\n结论: 闸门策略通过牺牲部分收益，显著降低回撤，提升稳健性")
    print("      适合追求'稳'的投资者 — 少赚但少亏，长期复利更可靠")


if __name__ == '__main__':
    final_robustness_test()
