"""
多股票组合回测
股票池 + 共享资金 + 等权基准对比
"""
import sys
import json
import hashlib
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from config import BacktestConfig
from data_source import DataSource, validate_daily
from strategies.ma_cross import MACrossStrategy
from strategies.combined import CombinedStrategy
from strategies.gated import GatedStrategy
from market_gate import MarketGate
from portfolio_engine import PortfolioEngine


# 股票池 (覆盖银行/白酒/保险/家电/医药/地产/券商, 12只大盘蓝筹)
DEFAULT_POOL = [
    '000001',  # 平安银行
    '600036',  # 招商银行
    '601166',  # 兴业银行
    '600519',  # 贵州茅台
    '000858',  # 五粮液
    '600887',  # 伊利股份
    '601318',  # 中国平安
    '600030',  # 中信证券
    '000651',  # 格力电器
    '000333',  # 美的集团
    '600276',  # 恒瑞医药
    '000002',  # 万科A
]


def build_strategy(symbol: str, mode: str = 'trend') -> GatedStrategy:
    """
    为每只股票构建策略

    mode:
        trend           - 长期趋势持有 MA20/150 (默认)
        mean_reversion  - 市场状态自适应组合 (短线均值回归)
    """
    gate = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
    if mode == 'trend':
        # MA20/150 需要约 150 根 K 线预热; 预热由引擎的 eval_start 在评估窗口外完成
        core = MACrossStrategy(fast_window=20, slow_window=150)
        # 趋势策略下个股暴跌过滤实测无差异, 关闭以减少参数
        return GatedStrategy(core, gate, enable_stock_filter=False)

    core = CombinedStrategy(
        fast_window=7, slow_window=20,
        boll_window=25, boll_std=2.0,
        adx_window=14, adx_threshold=35.0,
    )
    return GatedStrategy(core, gate, stock_ma_window=20,
                         stock_max_drop=-0.06, enable_stock_filter=True)


def load_pool(pool, start_date='20220101', calendar=None, adjust='hfq'):
    """
    加载股票池行情

    复权口径默认后复权 (hfq): 前复权 (qfq) 是把历史价格按累计分红/送股往下减,
    长期高分红个股会出现负价格 (实测茅台 2015 年前复权收盘 -117.62),
    负价格上的百分比收益没有意义, 且会被 validate_daily 当作脏数据剔除,
    等于把这段历史静默截断。后复权的价格恒为正, 收益率与真实总回报一致。
    """
    ds = DataSource(use_cache=True)
    data_dict = {}
    for symbol in pool:
        try:
            df = ds.get_stock_daily(symbol, start_date=start_date, adjust=adjust)
            if adjust == 'hfq':
                df = _normalize_to_latest_price(df, ds, symbol, start_date)
            df, issues = validate_daily(df, calendar=calendar)
            if issues:
                detail = ', '.join(f'{k}={v}' for k, v in issues.items())
                print(f"  [校验] {symbol}: {detail}")
            data_dict[symbol] = df
        except Exception as e:
            print(f"  {symbol} 数据获取失败: {e}")
    return data_dict


WARMUP_YEARS = 3   # 评估起点之前额外加载的历史 (只用于指标预热, 不参与交易)


def warmup_start(start_date: str, years: int = WARMUP_YEARS) -> str:
    """把评估起点往前推若干年, 供均线类指标在窗口外预热"""
    return (pd.to_datetime(start_date) - pd.DateOffset(years=years)).strftime('%Y%m%d')


def _normalize_to_latest_price(df, ds, symbol, start_date):
    """
    把后复权价整体缩放回"当前真实价格"的量级

    后复权价是"以上市首日为基准"的价格, 现在的茅台后复权价约 7000 元,
    直接用它会失真一手(100股)的资金约束 —— 12 只票里只有 2 只能买得起。
    整体乘一个常数 k = 最新真实收盘 / 最新后复权收盘, 收益率完全不变
    (相邻价格之比不变, 涨跌停判定也不变), 但成交金额回到真实量级。
    """
    raw = ds.get_stock_daily(symbol, start_date=start_date, adjust='')
    if raw is None or raw.empty or df.empty:
        return df

    last_adj = float(df['close'].iloc[-1])
    last_raw = float(raw['close'].iloc[-1])
    if not np.isfinite(last_adj) or not np.isfinite(last_raw) or last_adj <= 0:
        return df

    k = last_raw / last_adj
    out = df.copy()
    for col in ('open', 'high', 'low', 'close'):
        out[col] = out[col] * k
    out['raw_close'] = raw['close'].reindex(out.index)
    return out


def equal_weight_benchmark(data_dict, start=None, end=None) -> pd.Series:
    """
    等权买入持有基准

    窗口必须与策略评估窗口完全一致: 基准若比策略早几个月起步,
    两者比的就是不同的市场区间, 结论会被区间差异主导。
    每只股票按各自窗口内首个有效价格归一, 缺数据的标的逐日跳过。
    """
    closes = pd.DataFrame({s: df['close'] for s, df in data_dict.items()}).ffill()
    if start is not None:
        closes = closes[closes.index >= pd.to_datetime(start)]
    if end is not None:
        closes = closes[closes.index <= pd.to_datetime(end)]
    if closes.empty:
        return pd.Series(dtype=float)

    def _norm(col):
        first = col.dropna()
        return col / first.iloc[0] if len(first) else col

    normalized = closes.apply(_norm, axis=0)
    return normalized.mean(axis=1, skipna=True)


def _data_fingerprint(df: pd.DataFrame) -> str:
    """数据版本指纹: 行数 + 起止日期 + 最新收盘价"""
    key = f"{len(df)}|{df.index[0].date()}|{df.index[-1].date()}|{float(df['close'].iloc[-1]):.4f}"
    return hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]


def save_experiment_snapshot(result, data_dict, pool, mode, start_date, index_data=None,
                             history_start=None):
    """实验快照: 配置参数 + 数据版本 hash, 保证结果可复现/可追溯"""
    out = Path(__file__).parent / 'results'
    out.mkdir(exist_ok=True)
    metrics = {
        k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
        for k, v in result.metrics.items()
    }
    snapshot = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'command': f'python main.py portfolio --start {start_date} --mode {mode}',
        'mode': mode,
        'pool': list(pool),
        'config': result.config.to_dict(),
        'eval_window': [str(start_date), str(result.equity_curve.index[-1].date())],
        'warmup_start': history_start,
        'data_version': {s: _data_fingerprint(df) for s, df in data_dict.items()},
        'data_range': {
            s: [str(df.index[0].date()), str(df.index[-1].date()), len(df)]
            for s, df in data_dict.items()
        },
        'index_version': _data_fingerprint(index_data) if index_data is not None else None,
        'metrics': metrics,
    }
    path = out / f"experiment_portfolio_{mode}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str),
                    encoding='utf-8')
    print(f"实验快照已保存: {path}")
    return path


def run_portfolio(pool=None, start_date='20180101', max_positions=12, mode='trend'):
    pool = pool or DEFAULT_POOL
    history_start = warmup_start(start_date)
    print("=" * 78)
    print(f"多股票组合回测 - 股票池 {len(pool)} 只, 模式 {mode}")
    print(f"评估窗口 {start_date} 起; 历史数据自 {history_start} 加载 (窗口外仅做指标预热)")
    print("=" * 78)

    # 预加载指数数据（所有策略共享同一份, 同时作为交易日历）
    print("\n加载上证指数数据...")
    gate_probe = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
    index_data = gate_probe.load_index_data(start_date=history_start)
    if index_data is None:
        print("⚠ 指数数据不可用，闸门默认全开, 交易日历退化为个股并集")
        calendar = None
    else:
        print(f"指数数据: {len(index_data)} 条 (作为交易日历)")
        calendar = index_data.index

    print("\n加载行情数据...")
    data_dict = load_pool(pool, history_start, calendar=calendar)
    if len(data_dict) < 2:
        print("可用股票不足2只，无法组合回测")
        return None
    for s, df in data_dict.items():
        print(f"  {s}: {len(df)} 条, {df.index[0].date()} ~ {df.index[-1].date()}")

    config = BacktestConfig(
        initial_capital=100000,
        stop_loss_pct=0.08,
        # 趋势模式关闭固定止盈 (0 = 关闭), 让趋势自然走完; 均值回归模式保留止盈
        take_profit_pct=0.0 if mode == 'trend' else 0.20,
        # 移动止损在日线级别会打断正常波动, 默认关闭
        trailing_stop_pct=0.0,
        max_positions=max_positions,
    )

    def factory(symbol):
        strategy = build_strategy(symbol, mode=mode)
        # 注入预加载的指数数据，避免每只股票重复请求
        strategy.gate.set_index_data(index_data)
        return strategy

    engine = PortfolioEngine(data_dict, factory, config, trading_days=calendar,
                             eval_start=start_date)
    result = engine.run()
    m = result.metrics

    # 基准: 与策略同一评估窗口
    eval_end = result.equity_curve.index[-1]
    bench = equal_weight_benchmark(data_dict, start=start_date, end=eval_end)
    bench_ret = bench.iloc[-1] - 1 if len(bench) else 0.0
    bench_dd = (bench / bench.cummax() - 1).min() if len(bench) else 0.0

    print("\n" + "=" * 78)
    print("组合绩效")
    print("=" * 78)
    print(f"  总收益        : {m['total_return']:.2%}")
    print(f"  年化收益      : {m['annual_return']:.2%}")
    print(f"  夏普比率      : {m['sharpe_ratio']:.3f}")
    print(f"  索提诺比率    : {m['sortino_ratio']:.3f}")
    print(f"  最大回撤      : {m['max_drawdown']:.2%}")
    print(f"  卡玛比率      : {m['calmar_ratio']:.3f}")
    print(f"  交易笔数      : {m['total_trades']}")
    print(f"  胜率          : {m['win_rate']:.1%}")
    print(f"  盈亏比        : {m['profit_factor']:.2f}")
    print(f"  平均持仓天数  : {m['avg_holding_days']:.1f}")
    print("-" * 78)
    print(f"  等权基准收益  : {bench_ret:.2%}")
    print(f"  等权基准回撤  : {bench_dd:.2%}")
    print(f"  超额收益      : {m['total_return'] - bench_ret:.2%}")

    # 单只股票贡献
    trades_df = pd.DataFrame([t.__dict__ for t in result.trades])
    if not trades_df.empty:
        print("\n" + "=" * 78)
        print("各股票交易贡献")
        print("=" * 78)
        print(f"{'代码':<10} {'笔数':>6} {'总盈亏':>12} {'胜率':>8} {'平均收益':>10}")
        print("-" * 50)
        for symbol, g in trades_df.groupby('symbol'):
            print(f"{symbol:<10} {len(g):>6} {g['pnl'].sum():>11.0f} "
                  f"{(g['pnl'] > 0).mean():>7.1%} {g['pnl_pct'].mean():>9.2%}")
        print("-" * 50)
        equity_gain = result.equity_curve.iloc[-1] / config.initial_capital - 1
        print(f"  [口径] 逐笔 pnl 合计 {trades_df['pnl'].sum():,.0f} 元 (已含买入佣金); "
              f"组合净赚 {equity_gain * config.initial_capital:,.0f} 元, "
              f"差额为空仓资金利息, 不计入任何一笔交易")

    # 组合 vs 单票 对比
    print("\n" + "=" * 78)
    print("组合 vs 单票 对比")
    print("=" * 78)
    print(f"{'指标':<14} {'组合策略':>12} {'等权持有':>12}")
    print("-" * 42)
    print(f"{'总收益':<14} {m['total_return']:>11.2%} {bench_ret:>11.2%}")
    print(f"{'最大回撤':<14} {m['max_drawdown']:>11.2%} {bench_dd:>11.2%}")
    print(f"{'胜率':<14} {m['win_rate']:>11.1%} {'—':>12}")

    # 绘图
    _plot(result.equity_curve, bench, pool, mode)

    # 保存
    out = Path(__file__).parent / 'results'
    out.mkdir(exist_ok=True)
    result.equity_curve.to_csv(out / f'portfolio_equity_{mode}.csv', encoding='utf-8-sig')
    trades_df.to_csv(out / f'portfolio_trades_{mode}.csv', index=False, encoding='utf-8-sig')
    print(f"\n结果已保存: {out / f'portfolio_equity_{mode}.csv'}")

    save_experiment_snapshot(result, data_dict, pool, mode, start_date, index_data=index_data,
                             history_start=history_start)

    return result


def _plot(equity: pd.Series, bench: pd.Series, pool, mode: str = 'trend'):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

    e = equity / equity.iloc[0]
    b = bench / bench.iloc[0]
    ax1.plot(e.index, e.values, label='组合策略', color='#2ecc71', linewidth=1.4)
    ax1.plot(b.index, b.values, label='等权买入持有', color='#95a5a6', linewidth=1.1)
    ax1.set_title(f'多股票组合净值对比 ({len(pool)}只, {mode})')
    ax1.set_ylabel('净值 (起始=1.0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    dd = e / e.cummax() - 1
    ax2.fill_between(dd.index, dd.values, 0, color='#e74c3c', alpha=0.4)
    ax2.set_title('组合策略回撤')
    ax2.set_ylabel('回撤')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).parent / 'results' / f'portfolio_result_{mode}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"图表已保存: {out}")
    plt.close()


if __name__ == '__main__':
    run_portfolio()
