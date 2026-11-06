"""
参数优化运行脚本
对双均线和布林带策略在真实数据上做参数寻优 + 滚动验证
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from config import BacktestConfig
from data_source import DataSource
from strategies.ma_cross import MACrossStrategy
from strategies.bollinger import BollingerStrategy
from optimizer import ParameterOptimizer, plot_optimization_heatmap


def optimize_ma_cross(symbol: str = '000001'):
    """双均线参数优化"""
    print("\n" + "=" * 60)
    print(f"双均线策略参数优化 - {symbol}")
    print("=" * 60)

    ds = DataSource(use_cache=True)
    data = ds.get_stock_daily(symbol, start_date='20220101', adjust='qfq')
    print(f"数据: {len(data)} 条, {data.index[0].date()} ~ {data.index[-1].date()}")

    config = BacktestConfig(
        initial_capital=100000,
        stop_loss_pct=0.08,
        take_profit_pct=0.20,
        trailing_stop_pct=0.06,
    )

    # 参数网格
    param_grid = {
        'fast_window': [3, 5, 7, 10, 12, 15],
        'slow_window': [20, 25, 30, 40, 50, 60],
    }

    optimizer = ParameterOptimizer(
        strategy_class=MACrossStrategy,
        param_grid=param_grid,
        data=data,
        config=config,
        metric='sharpe_ratio',  # 以夏普比率为优化目标
    )

    result = optimizer.grid_search(verbose=True)
    print(result.summary(top_n=10))

    # 滚动前进验证 (检验是否过拟合)
    print("\n--- 滚动前进验证 (Walk-Forward) ---")
    wf = optimizer.walk_forward_analysis(result.best_params, n_splits=5, train_ratio=0.7)
    if 'message' not in wf:
        print(f"验证段数: {wf['n_folds']}")
        print(f"平均收益: {wf['avg_return']:.2%} (标准差 {wf['std_return']:.2%})")
        print(f"平均夏普: {wf['avg_sharpe']:.3f}")
        print(f"平均回撤: {wf['avg_drawdown']:.2%}")
        print(f"平均胜率: {wf['avg_win_rate']:.1%}")
        print(f"盈利段数: {wf['positive_folds']}/{wf['n_folds']}")
        if wf['positive_folds'] >= wf['n_folds'] * 0.6:
            print("✓ 多数时间段盈利，策略稳健性较好")
        else:
            print("⚠ 盈利时间段不足半数，可能存在过拟合")
    else:
        print(wf['message'])

    # 保存结果
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    result.results_df.to_csv(output_dir / f'opt_ma_cross_{symbol}.csv', index=False, encoding='utf-8-sig')
    print(f"\n优化结果已保存: {output_dir / f'opt_ma_cross_{symbol}.csv'}")

    # 热力图
    try:
        plot_optimization_heatmap(
            result, param_x='fast_window', param_y='slow_window',
            save_path=str(output_dir / f'opt_ma_cross_{symbol}_heatmap.png'),
        )
    except Exception as e:
        print(f"热力图绘制失败: {e}")

    return result


def optimize_bollinger(symbol: str = '000001'):
    """布林带参数优化"""
    print("\n" + "=" * 60)
    print(f"布林带策略参数优化 - {symbol}")
    print("=" * 60)

    ds = DataSource(use_cache=True)
    data = ds.get_stock_daily(symbol, start_date='20220101', adjust='qfq')
    print(f"数据: {len(data)} 条")

    config = BacktestConfig(
        initial_capital=100000,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        trailing_stop_pct=0.03,
    )

    param_grid = {
        'window': [10, 15, 20, 25, 30],
        'num_std': [1.0, 1.5, 2.0, 2.5, 3.0],
    }

    optimizer = ParameterOptimizer(
        strategy_class=BollingerStrategy,
        param_grid=param_grid,
        data=data,
        config=config,
        metric='sharpe_ratio',
    )

    result = optimizer.grid_search(verbose=True)
    print(result.summary(top_n=10))

    print("\n--- 滚动前进验证 ---")
    wf = optimizer.walk_forward_analysis(result.best_params, n_splits=5, train_ratio=0.7)
    if 'message' not in wf:
        print(f"平均收益: {wf['avg_return']:.2%}")
        print(f"平均夏普: {wf['avg_sharpe']:.3f}")
        print(f"盈利段数: {wf['positive_folds']}/{wf['n_folds']}")
    else:
        print(wf['message'])

    output_dir = Path(__file__).parent / 'results'
    result.results_df.to_csv(output_dir / f'opt_bollinger_{symbol}.csv', index=False, encoding='utf-8-sig')

    try:
        plot_optimization_heatmap(
            result, param_x='window', param_y='num_std',
            save_path=str(output_dir / f'opt_bollinger_{symbol}_heatmap.png'),
        )
    except Exception as e:
        print(f"热力图绘制失败: {e}")

    return result


if __name__ == '__main__':
    optimize_ma_cross('000001')
    optimize_bollinger('000001')
    print("\n参数优化全部完成!")
