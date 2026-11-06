"""
多维度综合评估
跨多只股票测试，从收益、风险、稳健性、一致性等维度全面评估
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from config import BacktestConfig
from data_source import DataSource
from strategies.combined import CombinedStrategy
from strategies.gated import GatedStrategy
from market_gate import MarketGate
from engine import BacktestEngine


def evaluate_stock(symbol, name, start_date='20220101'):
    """评估单只股票"""
    ds = DataSource(use_cache=True)
    data = ds.get_stock_daily(symbol, start_date=start_date, adjust='qfq')

    config = BacktestConfig(
        initial_capital=100000,
        stop_loss_pct=0.08,
        take_profit_pct=0.20,
        trailing_stop_pct=0.06,
    )

    # 优化后组合策略
    combined = CombinedStrategy(
        fast_window=7, slow_window=20,
        boll_window=25, boll_std=2.0,
        adx_window=14, adx_threshold=35.0,
    )

    # 带闸门
    gate = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
    gate.load_index_data(start_date=start_date)
    gated = GatedStrategy(combined, gate)

    results = {}
    for label, strategy in [('无闸门', combined), ('有闸门', gated)]:
        engine = BacktestEngine(data, strategy, config)
        result = engine.run()
        m = result.metrics

        # 买入持有基准
        benchmark_ret = (data['close'].iloc[-1] / data['close'].iloc[0] - 1)

        results[label] = {
            'return': m['total_return'],
            'annual': m['annual_return'],
            'sharpe': m['sharpe_ratio'],
            'sortino': m['sortino_ratio'],
            'max_dd': m['max_drawdown'],
            'calmar': m['calmar_ratio'],
            'win_rate': m['win_rate'],
            'trades': m['total_trades'],
            'profit_factor': m['profit_factor'],
            'benchmark': benchmark_ret,
            'excess_return': m['total_return'] - benchmark_ret,
        }
    return results


def main():
    print("=" * 80)
    print("量化交易系统 - 多维度综合评估")
    print("=" * 80)

    stocks = [
        ('000001', '平安银行'),
        ('600519', '贵州茅台'),
        ('000858', '五粮液'),
    ]

    all_results = {}
    for symbol, name in stocks:
        try:
            print(f"\n评估 {name}({symbol})...")
            all_results[f"{name}({symbol})"] = evaluate_stock(symbol, name)
        except Exception as e:
            print(f"  失败: {e}")

    # 汇总表
    print("\n" + "=" * 80)
    print("汇总: 有闸门策略表现")
    print("=" * 80)
    print(f"{'股票':<16} {'策略收益':>10} {'基准收益':>10} {'超额':>8} "
          f"{'夏普':>7} {'回撤':>8} {'胜率':>7} {'卡玛':>7} {'盈亏比':>7}")
    print("-" * 80)

    gated_returns = []
    gated_sharpes = []
    gated_drawdowns = []
    excess_returns = []
    win_rates = []

    for stock_name, res in all_results.items():
        g = res['有闸门']
        print(f"{stock_name:<16} {g['return']:>9.2%} {g['benchmark']:>9.2%} "
              f"{g['excess_return']:>7.2%} {g['sharpe']:>7.3f} {g['max_dd']:>7.2%} "
              f"{g['win_rate']:>6.1%} {g['calmar']:>7.3f} {g['profit_factor']:>7.2f}")
        gated_returns.append(g['return'])
        gated_sharpes.append(g['sharpe'])
        gated_drawdowns.append(g['max_dd'])
        excess_returns.append(g['excess_return'])
        win_rates.append(g['win_rate'])

    print("-" * 80)
    print(f"{'平均':<16} {np.mean(gated_returns):>9.2%} "
          f"{np.mean([r['benchmark'] for r in [res['有闸门'] for res in all_results.values()]]):>9.2%} "
          f"{np.mean(excess_returns):>7.2%} {np.mean(gated_sharpes):>7.3f} "
          f"{np.mean(gated_drawdowns):>7.2%} {np.mean(win_rates):>6.1%}")
    print("=" * 80)

    # 维度评分
    print("\n" + "=" * 80)
    print("多维度评分 (满分10分)")
    print("=" * 80)

    scores = {}
    # 1. 盈利能力 (平均超额收益)
    avg_excess = np.mean(excess_returns)
    scores['盈利能力'] = min(10, max(0, 5 + avg_excess * 100))

    # 2. 风险控制 (平均回撤越小越好)
    avg_dd = np.mean(gated_drawdowns)
    scores['风险控制'] = min(10, max(0, 10 - avg_dd * 30))

    # 3. 风险调整收益 (平均夏普)
    avg_sharpe = np.mean(gated_sharpes)
    scores['风险调整收益'] = min(10, max(0, 5 + avg_sharpe * 5))

    # 4. 稳健性 (胜率)
    avg_wr = np.mean(win_rates)
    scores['稳健性'] = min(10, max(0, avg_wr * 10))

    # 5. 一致性 (超额收益为正的股票占比)
    positive_ratio = sum(1 for e in excess_returns if e > 0) / len(excess_returns)
    scores['一致性'] = positive_ratio * 10

    total = sum(scores.values()) / len(scores)

    for dim, score in scores.items():
        bar = '█' * int(score) + '░' * (10 - int(score))
        print(f"  {dim:<12} {bar} {score:.1f}/10")
    print(f"  {'综合评分':<12} {'█' * int(total)}{'░' * (10 - int(total))} {total:.1f}/10")
    print("=" * 80)

    # 结论
    print("\n评估结论:")
    if total >= 7:
        print("✓ 系统表现优秀，具备实盘模拟的基础条件")
    elif total >= 5:
        print("△ 系统基本可用，但仍有改进空间，建议先小资金模拟")
    else:
        print("✗ 系统表现不佳，不建议实盘，需进一步优化")

    print(f"\n  - 平均超额收益: {avg_excess:.2%}")
    print(f"  - 平均最大回撤: {avg_dd:.2%}")
    print(f"  - 平均胜率: {avg_wr:.1%}")
    print(f"  - 跑赢基准的股票: {sum(1 for e in excess_returns if e > 0)}/{len(excess_returns)}")


if __name__ == '__main__':
    main()
