"""
统一入口 - 所有研究与回测命令

用法:
    python main.py demo                         # 模拟数据演示
    python main.py compare  --symbol 000001     # 单票多策略对比
    python main.py portfolio                    # 多股票组合回测
    python main.py optimize                     # 组合风控参数扫描
    python main.py validate                     # 样本内外 + 滚动验证
"""
import sys
from pathlib import Path
import argparse
import pandas as pd

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).parent))

from config import BacktestConfig
from data_loader import DataLoader
from data_source import DataSource
from strategies.ma_cross import MACrossStrategy
from strategies.bollinger import BollingerStrategy
from engine import BacktestEngine
from plotting import Plotter


def _save_result(result, name: str, output_dir: Path):
    """保存回测结果"""
    output_dir.mkdir(exist_ok=True)
    result.equity_curve.to_csv(output_dir / f'{name}_equity.csv')
    if result.trades:
        trades_df = pd.DataFrame([t.__dict__ for t in result.trades])
        trades_df.to_csv(output_dir / f'{name}_trades.csv', index=False, encoding='utf-8-sig')
    try:
        Plotter.plot_result(result, save_path=str(output_dir / f'{name}_result.png'))
    except Exception as e:
        print(f"绘图失败 (可忽略): {e}")


def run_ma_cross_backtest():
    """双均线策略回测 (模拟数据)"""
    print("=" * 60)
    print("策略: 双均线 (趋势跟踪) - 模拟数据")
    print("=" * 60)

    print("加载模拟行情数据...")
    data = DataLoader.generate_sample_data(days=500, seed=42)
    print(f"数据范围: {data.index[0].date()} ~ {data.index[-1].date()}, 共{len(data)}条")

    config = BacktestConfig(
        initial_capital=100000,
        stop_loss_pct=0.08,
        take_profit_pct=0.20,
        trailing_stop_pct=0.06,
    )

    strategy = MACrossStrategy(fast_window=5, slow_window=20)
    print(f"策略参数: {strategy.describe()}")

    print("\n运行回测中...")
    engine = BacktestEngine(data, strategy, config)
    result = engine.run()

    print(result.summary())
    output_dir = Path(__file__).parent / 'results'
    _save_result(result, 'ma_cross_sim', output_dir)
    return result


def run_bollinger_backtest():
    """布林带策略回测 (模拟数据)"""
    print("\n" + "=" * 60)
    print("策略: 布林带均值回归 - 模拟数据")
    print("=" * 60)

    data = DataLoader.generate_sample_data(days=500, seed=42)
    print(f"数据范围: {data.index[0].date()} ~ {data.index[-1].date()}, 共{len(data)}条")

    config = BacktestConfig(
        initial_capital=100000,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        trailing_stop_pct=0.03,
    )

    strategy = BollingerStrategy(window=20, num_std=2.0)
    print(f"策略参数: {strategy.describe()}")

    print("\n运行回测中...")
    engine = BacktestEngine(data, strategy, config)
    result = engine.run()

    print(result.summary())
    output_dir = Path(__file__).parent / 'results'
    _save_result(result, 'bollinger_sim', output_dir)
    return result


def run_real_data_backtest(
    symbol: str = '000001',
    start_date: str = '20230101',
    end_date: str = None,
    strategy_name: str = 'ma_cross',
):
    """
    真实A股数据回测

    参数:
        symbol: 股票代码，如 '000001'(平安银行), '600519'(贵州茅台)
        start_date: 起始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD (默认今天)
        strategy_name: 'ma_cross' 或 'bollinger'
    """
    print("\n" + "=" * 60)
    print(f"真实数据回测: {symbol}")
    print("=" * 60)

    # 1. 从 AKShare 拉取真实行情
    print(f"正在从 AKShare 拉取 {symbol} 日线数据...")
    ds = DataSource(use_cache=True)
    data = ds.get_stock_daily(symbol, start_date=start_date, end_date=end_date, adjust='qfq')
    print(f"数据范围: {data.index[0].date()} ~ {data.index[-1].date()}, 共{len(data)}条")

    # 2. 配置和策略
    config = BacktestConfig(
        initial_capital=100000,
        stop_loss_pct=0.08,
        take_profit_pct=0.20,
        trailing_stop_pct=0.06,
    )

    if strategy_name == 'ma_cross':
        strategy = MACrossStrategy(fast_window=5, slow_window=20)
    elif strategy_name == 'bollinger':
        strategy = BollingerStrategy(window=20, num_std=2.0)
    else:
        raise ValueError(f"未知策略: {strategy_name}")

    print(f"策略: {strategy.describe()}")

    # 3. 运行回测
    print("\n运行回测中...")
    engine = BacktestEngine(data, strategy, config, symbol=symbol)
    result = engine.run()

    print(result.summary())

    # 4. 保存
    output_dir = Path(__file__).parent / 'results'
    name = f"real_{symbol}_{strategy_name}"
    _save_result(result, name, output_dir)
    print(f"\n结果已保存至: {output_dir}")

    # 5. 当前实时价格 (可选)
    quote = ds.get_realtime_quote(symbol)
    if quote:
        print(f"\n[实时行情] {quote['name']}({symbol}): "
              f"现价 {quote['price']:.2f}, 涨跌幅 {quote['pct_change']:.2f}%")

    return result


def cmd_demo(_args):
    run_ma_cross_backtest()
    run_bollinger_backtest()


def cmd_compare(args):
    from compare import compare_strategies
    compare_strategies(args.symbol, args.start)


def cmd_portfolio(args):
    from portfolio import run_portfolio
    run_portfolio(start_date=args.start, mode=args.mode)


def cmd_optimize(_args):
    from optimize_trend import main as optimize_main
    optimize_main()


def cmd_validate(_args):
    from validate_trend import main as validate_main
    validate_main()


def cmd_forecast(args):
    from trend_forecast import run_forecast
    run_forecast(symbol=args.symbol, start=args.start,
                 oos_start=args.oos, vol_target=args.vol_target)


def cmd_fund_rules(_args):
    from fund_rules_backtest import run_fund_rules
    run_fund_rules()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='量化交易研究框架')
    sub = p.add_subparsers(dest='command')

    d = sub.add_parser('demo', help='模拟数据演示')
    d.set_defaults(func=cmd_demo)

    c = sub.add_parser('compare', help='单票多策略对比')
    c.add_argument('--symbol', default='000001')
    c.add_argument('--start', default='20220101')
    c.set_defaults(func=cmd_compare)

    pf = sub.add_parser('portfolio', help='多股票组合回测')
    pf.add_argument('--start', default='20180101')
    pf.add_argument('--mode', default='trend',
                    choices=['trend', 'mean_reversion'],
                    help='trend=长期趋势持有(默认) / mean_reversion=短线均值回归')
    pf.set_defaults(func=cmd_portfolio)

    o = sub.add_parser('optimize', help='组合风控参数扫描')
    o.set_defaults(func=cmd_optimize)

    v = sub.add_parser('validate', help='样本内外 + 滚动验证')
    v.set_defaults(func=cmd_validate)

    f = sub.add_parser('forecast', help='指数趋势概率预测 (次日 + 20日)')
    f.add_argument('--symbol', default='000001', help='指数代码, 默认上证指数')
    f.add_argument('--start', default='20100101')
    f.add_argument('--oos', default='20180101', help='样本外起点')
    f.add_argument('--vol-target', type=float, default=0.12, help='目标年化波动')
    f.set_defaults(func=cmd_forecast)

    fr = sub.add_parser('fund-rules', help='自选基金规则回测 (预注册6条规则, 真实费率)')
    fr.set_defaults(func=cmd_fund_rules)

    return p


def main():
    """包入口 (pyproject [project.scripts]): 先配置日志, 再分发子命令"""
    from logging_setup import setup_logging
    setup_logging()

    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, 'command', None):
        parser.print_help()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
