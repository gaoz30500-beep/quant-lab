"""
组合策略 vs 单一策略 对比回测
在真实数据上跑三个策略，对比绩效
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from config import BacktestConfig
from data_source import DataSource
from strategies.ma_cross import MACrossStrategy
from strategies.bollinger import BollingerStrategy
from strategies.combined import CombinedStrategy
from engine import BacktestEngine
from plotting import Plotter
from market_state import MarketStateDetector, MarketState


def compare_strategies(symbol: str = '000001', start_date: str = '20220101'):
    """对比三个策略在同一只股票上的表现"""
    print("=" * 70)
    print(f"策略对比回测 - {symbol}")
    print("=" * 70)

    # 加载真实数据
    ds = DataSource(use_cache=True)
    data = ds.get_stock_daily(symbol, start_date=start_date, adjust='qfq')
    print(f"数据: {len(data)} 条, {data.index[0].date()} ~ {data.index[-1].date()}\n")

    # 市场状态分布
    detector = MarketStateDetector(adx_window=14, adx_threshold=25.0)
    states = detector.classify(data)
    dist = detector.get_state_distribution(states)
    print(f"市场状态分布: 趋势市 {dist['TRENDING']}, 震荡市 {dist['RANGING']}, 未知 {dist['UNKNOWN']}")
    print()

    config = BacktestConfig(
        initial_capital=100000,
        stop_loss_pct=0.08,
        take_profit_pct=0.20,
        trailing_stop_pct=0.06,
    )

    # 三个策略
    strategies = {
        '双均线(5,30)': MACrossStrategy(fast_window=5, slow_window=30),
        '布林带(20,2)': BollingerStrategy(window=20, num_std=2.0),
        '组合策略': CombinedStrategy(
            fast_window=5, slow_window=30,
            boll_window=20, boll_std=2.0,
            adx_window=14, adx_threshold=25.0,
        ),
    }

    results = {}
    for name, strategy in strategies.items():
        print(f"--- {name} ---")
        engine = BacktestEngine(data, strategy, config, symbol=symbol)
        result = engine.run()
        results[name] = result

        m = result.metrics
        print(f"  总收益: {m['total_return']:.2%}  年化: {m['annual_return']:.2%}")
        print(f"  夏普: {m['sharpe_ratio']:.3f}  最大回撤: {m['max_drawdown']:.2%}")
        print(f"  交易次数: {m['total_trades']}  胜率: {m['win_rate']:.1%}  盈亏比: {m['profit_factor']:.2f}")
        print()

        # 保存
        output_dir = Path(__file__).parent / 'results'
        safe_name = name.replace('(', '_').replace(')', '').replace(',', '_').replace(' ', '')
        result.equity_curve.to_csv(output_dir / f'compare_{safe_name}_equity.csv')

    # 绘制净值对比图
    _plot_comparison(results, symbol)

    # 打印对比总结
    print("=" * 70)
    print("策略对比总结")
    print("=" * 70)
    print(f"{'策略':<16} {'总收益':>10} {'夏普':>8} {'回撤':>10} {'胜率':>8} {'盈亏比':>8}")
    print("-" * 70)
    for name, result in results.items():
        m = result.metrics
        print(f"{name:<16} {m['total_return']:>9.2%} {m['sharpe_ratio']:>8.3f} "
              f"{m['max_drawdown']:>9.2%} {m['win_rate']:>7.1%} {m['profit_factor']:>8.2f}")
    print("=" * 70)

    return results


def _plot_comparison(results: dict, symbol: str):
    """绘制多策略净值对比图"""
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {'双均线(5,30)': '#3498db', '布林带(20,2)': '#e67e22', '组合策略': '#2ecc71'}

    for name, result in results.items():
        equity = result.equity_curve / result.equity_curve.iloc[0]  # 归一化
        ax.plot(equity.index, equity.values, label=name,
                color=colors.get(name, None), linewidth=1.3)

    ax.set_title(f'{symbol} 多策略净值对比 (归一化)')
    ax.set_ylabel('净值 (起始=1.0)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_dir = Path(__file__).parent / 'results'
    plt.tight_layout()
    plt.savefig(output_dir / f'compare_{symbol}.png', dpi=150, bbox_inches='tight')
    print(f"对比图已保存: {output_dir / f'compare_{symbol}.png'}\n")
    plt.close()


if __name__ == '__main__':
    compare_strategies('000001', '20220101')
