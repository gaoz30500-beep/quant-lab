"""
最低成本下的真实收益
不同本金档位跑同一策略, 含最低佣金 5 元 / 印花税 / 滑点 / 100股整数倍约束
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from config import BacktestConfig
from market_gate import MarketGate
from portfolio_engine import PortfolioEngine
from portfolio import DEFAULT_POOL, build_strategy, load_pool, warmup_start

EVAL_START = '2018-01-01'   # 评估起点; 之前的历史只用于指标预热


def run_at_capital(data, idx, capital, max_positions=12):
    config = BacktestConfig(
        initial_capital=capital,
        max_positions=max_positions,
        stop_loss_pct=0.08,
        take_profit_pct=0.0,      # 0 = 关闭固定止盈
        trailing_stop_pct=0.0,
        cash_yield=0.02,
        min_commission=5.0,
    )

    def factory(symbol):
        st = build_strategy(symbol, mode='trend')
        st.gate.set_index_data(idx)
        return st

    result = PortfolioEngine(data, factory, config,
                             trading_days=idx.index if idx is not None else None,
                             eval_start=EVAL_START).run()
    m = result.metrics

    trades = pd.DataFrame([t.__dict__ for t in result.trades]) if result.trades else pd.DataFrame()
    symbols = sorted(trades['symbol'].unique()) if not trades.empty else []
    avg_ticket = trades['entry_price'].mul(trades['shares']).mean() if not trades.empty else 0.0
    return m, trades, symbols, avg_ticket


def main():
    history_start = warmup_start(EVAL_START[:10].replace('-', ''))
    data = load_pool(DEFAULT_POOL, start_date=history_start)
    g = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
    idx = g.load_index_data(start_date=history_start)

    # 各股票一手成本 (当前价 * 100)
    one_lot = {}
    for s, df in data.items():
        one_lot[s] = round(float(df['close'].iloc[-1]) * 100)

    print("=" * 96)
    print("各标的一手成本 (决定最低本金)")
    print("=" * 96)
    for s, c in sorted(one_lot.items(), key=lambda x: x[1]):
        print(f"  {s}: {c:>9,} 元")
    print(f"  12只全买一手合计: {sum(one_lot.values()):>10,} 元")
    print(f"  最便宜一手:       {min(one_lot.values()):>10,} 元")

    print("\n" + "=" * 96)
    print("不同本金下的真实回测 (2018-2026, 策略与参数完全相同)")
    print("=" * 96)
    print(f"{'本金':>10} {'总收益':>9} {'净赚':>11} {'年化':>7} {'回撤':>8} "
          f"{'笔数':>5} {'可交易标的':>9} {'平均单笔':>10} {'每仓额度':>10}")
    print("-" * 96)

    for cap in [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]:
        m, trades, symbols, avg_ticket = run_at_capital(data, idx, cap)
        print(f"{cap:>10,} {m['total_return']:>8.2%} {cap * m['total_return']:>10,.0f} "
              f"{m['annual_return']:>6.2%} {m['max_drawdown']:>7.2%} "
              f"{m['total_trades']:>5} {len(symbols):>9} {avg_ticket:>9,.0f} {cap/12:>9,.0f}")

    print("-" * 96)
    print("说明: 每仓额度 = 本金/12, 低于最低一手成本时该标的无法成交")
    print("      平均单笔 = 实际成交单笔金额, 低于1000元时5元最低佣金占比超过0.5%")


if __name__ == '__main__':
    main()
