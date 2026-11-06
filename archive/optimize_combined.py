"""
组合策略参数优化
优化 ADX 阈值 + 子策略参数，找到最优搭配
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from config import BacktestConfig
from data_source import DataSource
from strategies.combined import CombinedStrategy
from optimizer import ParameterOptimizer


def optimize_combined(symbol: str = '000001'):
    print("=" * 60)
    print(f"组合策略参数优化 - {symbol}")
    print("=" * 60)

    ds = DataSource(use_cache=True)
    data = ds.get_stock_daily(symbol, start_date='20220101', adjust='qfq')
    print(f"数据: {len(data)} 条\n")

    config = BacktestConfig(
        initial_capital=100000,
        stop_loss_pct=0.08,
        take_profit_pct=0.20,
        trailing_stop_pct=0.06,
    )

    # 重点优化 ADX 阈值和趋势/震荡策略参数
    param_grid = {
        'adx_threshold': [18, 20, 25, 30, 35],     # ADX 切换阈值
        'fast_window': [3, 5, 7],                  # 趋势策略快线
        'slow_window': [20, 30, 40],               # 趋势策略慢线
        'boll_window': [15, 20, 25],               # 布林带周期
        'boll_std': [1.5, 2.0, 2.5],               # 布林带标准差
    }
    # 组合数: 5×3×3×3×3 = 405 组

    optimizer = ParameterOptimizer(
        strategy_class=CombinedStrategy,
        param_grid=param_grid,
        data=data,
        config=config,
        metric='sharpe_ratio',
    )

    result = optimizer.grid_search(verbose=True)
    print(result.summary(top_n=10))

    # 滚动验证
    print("\n--- 滚动前进验证 ---")
    wf = optimizer.walk_forward_analysis(result.best_params, n_splits=5, train_ratio=0.7)
    if 'message' not in wf:
        print(f"验证段数: {wf['n_folds']}")
        print(f"平均收益: {wf['avg_return']:.2%} (标准差 {wf['std_return']:.2%})")
        print(f"平均夏普: {wf['avg_sharpe']:.3f}")
        print(f"平均回撤: {wf['avg_drawdown']:.2%}")
        print(f"平均胜率: {wf['avg_win_rate']:.1%}")
        print(f"盈利段数: {wf['positive_folds']}/{wf['n_folds']}")
        if wf['positive_folds'] >= wf['n_folds'] * 0.6:
            print("✓ 多数时间段盈利")
        else:
            print("⚠ 盈利段不足，可能过拟合")
    else:
        print(wf['message'])

    # 保存
    output_dir = Path(__file__).parent / 'results'
    result.results_df.to_csv(output_dir / f'opt_combined_{symbol}.csv', index=False, encoding='utf-8-sig')
    print(f"\n优化结果已保存: {output_dir / f'opt_combined_{symbol}.csv'}")

    return result


if __name__ == '__main__':
    optimize_combined('000001')
    print("\n组合策略参数优化完成!")
