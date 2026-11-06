"""
趋势策略参数网格
筛选标准: 样本内与样本外同时为正, 再比卡玛比率

口径约定:
- 数据一律从 HISTORY_START 加载, 评估窗口由 run(start=, end=) 控制
- 指标预热在评估窗口之外完成 (引擎 eval_start), 不靠"切数据后重新累积均线"
- 段间比较用年化收益 (区间长度不同, 总收益不可直接比)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from config import BacktestConfig
from strategies.ma_cross import MACrossStrategy
from strategies.gated import GatedStrategy
from market_gate import MarketGate
from portfolio_engine import PortfolioEngine
from portfolio import DEFAULT_POOL, load_pool

HISTORY_START = '20150101'      # 比评估起点早 3 年, 保证 MA150/MA200 有足够预热
EVAL_IN = ('2018-01-01', '2022-12-31')
EVAL_OUT = ('2023-01-01', '2026-10-31')


def run(data, idx, fast, slow, start=None, end=None, cash_yield=0.02,
        return_equity=False):
    """
    在 [start, end] 评估, 指标预热在窗口之外完成

    旧写法先按 start 切数据再算信号: MA150 每段开头要重新累积 150 根 K 线,
    段首数月被动空仓, 在下跌段会系统性躲过行情, 从而高估稳定性。
    现在把完整历史交给引擎, 由 eval_start 决定从哪天开始交易。

    cash_yield: 闲置资金年化 (默认 0.02 与 BacktestConfig 一致, 不改变现有口径)
    return_equity: True 时返回 (metrics, equity_curve) — 供混合底仓等二次加工
    """
    eval_start = pd.Timestamp(start) if start else None
    eval_end = pd.Timestamp(end) if end else None

    sub = {}
    for s, df in data.items():
        d = df if eval_end is None else df[df.index <= eval_end]
        if len(d) > 0:
            sub[s] = d

    config = BacktestConfig(
        initial_capital=100000,
        max_positions=len(sub),
        stop_loss_pct=0.08,
        take_profit_pct=0.0,        # 0 = 关闭固定止盈
        trailing_stop_pct=0.0,
        cash_yield=cash_yield,
    )

    def factory(symbol):
        core = MACrossStrategy(fast_window=fast, slow_window=slow)
        g = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
        g.set_index_data(idx)
        # 趋势策略下个股暴跌过滤实测无差异, 统一关闭以减少参数
        return GatedStrategy(core, g, enable_stock_filter=False)

    result = PortfolioEngine(
        sub, factory, config,
        trading_days=idx.index if idx is not None else None,
        eval_start=eval_start,
    ).run()
    if return_equity:
        return result.metrics, result.equity_curve
    return result.metrics


def main():
    print(f"加载历史行情 ({HISTORY_START} 起, 评估窗口之外的数据仅用于指标预热)...")
    data = load_pool(DEFAULT_POOL, start_date=HISTORY_START)
    g = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
    idx = g.load_index_data(start_date=HISTORY_START)

    print("=" * 112)
    print("趋势参数网格 (预热在窗口外完成; 双段一致性优先)")
    print("=" * 112)
    print(f"{'参数':<12} {'全样本':>9} {'全回撤':>8} {'内年化':>8} {'外年化':>8} "
          f"{'外回撤':>8} {'卡玛':>7} {'双段为正':>8} {'笔数':>6}")
    print("-" * 112)

    rows = []
    for fast in [20, 30, 40, 60]:
        for slow in [100, 120, 150, 200]:
            if fast >= slow:
                continue
            m_all = run(data, idx, fast, slow)
            m_in = run(data, idx, fast, slow, EVAL_IN[0], EVAL_IN[1])
            m_out = run(data, idx, fast, slow, EVAL_OUT[0], EVAL_OUT[1])
            both_pos = m_in['total_return'] > 0 and m_out['total_return'] > 0
            rows.append({
                'fast': fast, 'slow': slow,
                'ret': m_all['total_return'], 'dd': m_all['max_drawdown'],
                'in': m_in['total_return'], 'out': m_out['total_return'],
                'in_ann': m_in['annual_return'], 'out_ann': m_out['annual_return'],
                'out_dd': m_out['max_drawdown'],
                'calmar': m_all['calmar_ratio'],
                'both_pos': both_pos, 'trades': m_all['total_trades'],
            })
            label = f'MA{fast}/{slow}'
            print(f"{label:<12} {m_all['total_return']:>8.2%} {m_all['max_drawdown']:>7.2%} "
                  f"{m_in['annual_return']:>7.2%} {m_out['annual_return']:>7.2%} "
                  f"{m_out['max_drawdown']:>7.2%} {m_all['calmar_ratio']:>7.3f} "
                  f"{('是' if both_pos else '否'):>6} {m_all['total_trades']:>6}")

    df = pd.DataFrame(rows)
    out = Path(__file__).parent / 'results' / 'opt_trend.csv'
    df.to_csv(out, index=False, encoding='utf-8-sig')

    # 稳健候选: 双段为正, 按"较差一段的年化收益"排序 (段长不同, 不能比总收益)
    cand = df[df['both_pos']].copy()
    cand['min_seg_ann'] = cand[['in_ann', 'out_ann']].min(axis=1)
    cand = cand.sort_values(['min_seg_ann', 'calmar'], ascending=False)
    print("\n" + "=" * 112)
    print("双段均为正的候选 (按较差一段的年化收益排序)")
    print("=" * 112)
    if cand.empty:
        print("无候选: 所有参数都无法在样本内/样本外同时盈利")
    else:
        for _, r in cand.head(5).iterrows():
            print(f"  MA{int(r['fast'])}/{int(r['slow'])}: 样本内年化 {r['in_ann']:.2%}, "
                  f"样本外年化 {r['out_ann']:.2%}, 样本外回撤 {r['out_dd']:.2%}, "
                  f"卡玛 {r['calmar']:.3f}")
    print(f"\n已保存: {out}")


if __name__ == '__main__':
    main()
