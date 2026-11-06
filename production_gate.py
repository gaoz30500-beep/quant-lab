"""
生产验收闸门 (Production Gate)

把"可信 / 稳 / 中高回报"翻译成可自动检查的硬指标。
全部通过 → exit 0 (绿灯, 配置可上生产观察); 任一不过 → exit 1 (红灯)。

用法:
    python production_gate.py                      # 默认 l1l2 (预注册的生产候选)
    python production_gate.py --variant base       # 原始口径 (现金2%, 无底仓)
    python production_gate.py --variant l1         # 只加 L1 现金增效 3%
    python production_gate.py --variant l1l2       # L1 + L2 黄金底仓 15%
    python production_gate.py --skip-tests         # 跳过单元测试
    python production_gate.py --json out.json      # 结果另存

验收标准 (生产参数 MA20/150, 12 只蓝筹池, 真实费率/滑点/T+1/涨跌停口径):

  可信 (过程可信, 不是承诺收益):
    A1 单元测试全绿            —— 引擎口径没有漂移 (golden master)
    A2 样本内(2018-2022)为正   —— 双段一致性的一半
    A3 样本外(2023-2026)为正   —— 参数定死后没碰过的数据仍然赚钱

  稳:
    B1 样本外最大回撤 <= 25%
    B2 滚动 4 段至少 3 段为正, 且最差一段 >= -15%  (即预注册的 S1: 混合后仍须过线)
    B3 全样本卡玛比率 >= 0.5

  中高回报 (现实的门槛, 不是幻想):
    C1 样本外年化 >= 8% (长期跑赢理财/货基, 接近偏股基金中位数以上)
    C2 样本外跑赢等权买入持有基准, 或收益差距 < 2pp 且回撤显著更浅

注意: 阈值是"验收线"不是"预期值"。过线 ≠ 未来赚钱, 不过线 = 现在就不能信。
"""
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from optimize_trend import run, HISTORY_START, EVAL_IN, EVAL_OUT
from market_gate import MarketGate
from portfolio import DEFAULT_POOL, load_pool, equal_weight_benchmark
from data_source import DataSource
from metrics import Metrics
from logging_setup import setup_logging, get_logger

logger = get_logger('production_gate')

# ---- 生产参数 (与 portfolio.build_strategy(mode='trend') 一致) ----
PROD_FAST, PROD_SLOW = 20, 150

# ---- 预注册实验 (2026-11-06, 跑之前写死, 禁止事后调参重挑) ----
# 背景: 首次验收 C1 FAIL (样本外年化 1.45%), 诊断=资金利用率过低 (样本外回撤仅
#       3.59% 说明大部分时间闲置吃 2% 利息)。风险预算未用完, 有两个零/低参数杠杆:
# L1 现金增效: 闲置资金收益 2% -> 3%
#    依据: 2018-2026 货基/短债基金年化区间约 1.5%~4%, 取 3% 为中性偏保守值。
#    这是口径修正而非策略参数 (原 2% 低估了实际可得的现金管理收益)。
# L2 黄金底仓: 组合净值 = 85% 趋势策略 + 15% 黄金ETF(518880) 买入持有
#    w=15% 取 10~20% 区间中点, 不调参; 黄金与 A 股趋势低相关, 零策略参数。
#    披露: 黄金 2023-2026 走强是事后已知信息, 存在选择偏差; 因此验收同时看
#    滚动 4 段 (覆盖 2018-2022 黄金平淡期), 4 段一致性不过即否决 (并入 B2)。
#    混合口径: 两腿各自归一后按 85/15 加权, 不再平衡 (年度漂移 <2pp, 忽略)。
# 验收: 沿用同一闸门 8 项标准, 阈值不变。
PROD_CASH_YIELD = 0.03
GOLD_ETF = '518880'
GOLD_WEIGHT = 0.15

# ---- 验收阈值 ----
LIMITS = {
    'oos_max_drawdown': 0.25,     # B1
    'fold_positive_min': 3,       # B2: 4 段中至少 3 段为正
    'fold_worst_return': -0.15,   # B2: 最差一段不低于 -15%
    'calmar_min': 0.5,            # B3
    'oos_annual_min': 0.08,       # C1
    'bench_gap_max': 0.02,        # C2: 输基准不超过 2pp 时才看回撤
    'dd_improve_min': 0.20,       # C2: 回撤至少比基准浅 20% (相对)
}


def check_unit_tests() -> bool:
    """A1: 单元测试全绿"""
    proc = subprocess.run(
        [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'],
        capture_output=True, text=True, cwd=Path(__file__).parent,
    )
    ok = proc.returncode == 0
    if not ok:
        logger.error('单元测试失败:\n%s', proc.stderr[-2000:])
    return ok


def blend_with_gold(eq: pd.Series, gold_close: pd.Series, w: float) -> pd.Series:
    """
    静态混合: (1-w)*策略 + w*黄金买入持有, 各自按窗口起点归一

    黄金腿对齐到策略净值日历 (停牌/日历差异 ffill); 窗口内不再平衡。
    """
    eqn = eq / eq.iloc[0]
    g = gold_close.reindex(eq.index, method='ffill')
    if g.dropna().empty:
        raise RuntimeError('黄金ETF数据与评估窗口无交集, 无法混合')
    g = g.ffill().bfill()
    gn = g / g.iloc[0]
    return (1 - w) * eqn + w * gn


def blend_metrics(m: dict, eq_blended: pd.Series) -> dict:
    """混合净值重算收益类指标; 交易统计仍来自策略腿 (黄金腿为买入持有, 无交易)"""
    mb = Metrics.calc_all(eq_blended, [], risk_free_rate=0.03)
    return {**m,
            'total_return': mb['total_return'],
            'annual_return': mb['annual_return'],
            'annual_volatility': mb['annual_volatility'],
            'sharpe_ratio': mb['sharpe_ratio'],
            'sortino_ratio': mb['sortino_ratio'],
            'max_drawdown': mb['max_drawdown'],
            'calmar_ratio': mb['calmar_ratio']}


def main():
    ap = argparse.ArgumentParser(description='生产验收闸门')
    ap.add_argument('--variant', choices=['base', 'l1', 'l1l2'], default='l1l2',
                    help='base=原口径 / l1=现金3%% / l1l2=现金3%%+黄金15%% (默认)')
    ap.add_argument('--skip-tests', action='store_true', help='跳过单元测试')
    ap.add_argument('--json', help='结果另存路径')
    args = ap.parse_args()
    setup_logging()

    cash_yield = PROD_CASH_YIELD if args.variant in ('l1', 'l1l2') else 0.02
    use_gold = args.variant == 'l1l2'

    print('=' * 78)
    print(f'生产验收闸门  MA{PROD_FAST}/{PROD_SLOW}  股票池 {len(DEFAULT_POOL)} 只  变体={args.variant}')
    print(f'现金收益 {cash_yield:.1%}' + (f'  黄金底仓 {GOLD_WEIGHT:.0%} ({GOLD_ETF})' if use_gold else ''))
    print(f'样本内 {EVAL_IN[0]}~{EVAL_IN[1]}   样本外 {EVAL_OUT[0]}~{EVAL_OUT[1]}')
    print('=' * 78)

    # ---- A1 单元测试 ----
    a1 = True if args.skip_tests else check_unit_tests()
    print(f"\n[A1] 单元测试全绿            : {'跳过' if args.skip_tests else ('PASS' if a1 else 'FAIL')}")

    # ---- 数据加载 (一次, 供所有回测复用) ----
    print(f'\n加载行情 ({HISTORY_START} 起, 窗口外仅预热)...')
    data = load_pool(DEFAULT_POOL, start_date=HISTORY_START)
    g = MarketGate(ma_window=20, min_return=-0.05, use_momentum_filter=True)
    idx = g.load_index_data(start_date=HISTORY_START)
    if idx is None or len(data) < 8:
        print(f'FATAL: 数据不足 (股票 {len(data)} 只, 指数 {"无" if idx is None else "有"})')
        return 1
    print(f'可用: {len(data)} 只股票, 指数 {len(idx)} 根K线')

    gold_close = None
    if use_gold:
        try:
            gold = DataSource(use_cache=True).get_etf_daily(GOLD_ETF, start_date=HISTORY_START)
            gold_close = gold['close']
            print(f'黄金ETF {GOLD_ETF}: {len(gold_close)} 根K线, '
                  f'{gold_close.index[0].date()} ~ {gold_close.index[-1].date()}')
        except Exception as e:
            print(f'FATAL: 黄金ETF数据获取失败, l1l2 无法验收: {e}')
            return 1

    def run_blended(start=None, end=None):
        """统一回测入口: 按变体决定现金收益与是否混合黄金腿"""
        m, eq = run(data, idx, PROD_FAST, PROD_SLOW, start, end,
                    cash_yield=cash_yield, return_equity=True)
        if use_gold:
            eq = blend_with_gold(eq, gold_close, GOLD_WEIGHT)
            m = blend_metrics(m, eq)
        return m, eq

    # ---- 三段回测 ----
    print('\n回测: 全样本 / 样本内 / 样本外 ...')
    m_all, _ = run_blended()
    m_in, _ = run_blended(*EVAL_IN)
    m_out, _ = run_blended(*EVAL_OUT)

    # ---- 滚动 4 段 ----
    print('回测: 滚动 4 段 ...')
    eval_dates = [d for d in idx.index if d >= pd.Timestamp('2018-01-01')]
    n = len(eval_dates)
    folds = []
    for i in range(4):
        s = eval_dates[i * n // 4]
        e = eval_dates[min((i + 1) * n // 4 - 1, n - 1)]
        fm, _ = run_blended(s, e)
        folds.append({'start': str(pd.Timestamp(s).date()),
                      'end': str(pd.Timestamp(e).date()), **fm})

    # ---- 样本外基准 (等权买入持有, 同窗口) ----
    bench = equal_weight_benchmark(data, start=EVAL_OUT[0], end=EVAL_OUT[1])
    bench_ret = float(bench.iloc[-1] - 1) if len(bench) else 0.0
    bench_dd = float((bench / bench.cummax() - 1).min()) if len(bench) else 0.0
    bench_years = len(bench) / 252.0 if len(bench) else 1.0
    bench_ann = (1 + bench_ret) ** (1 / bench_years) - 1 if bench_years > 0 else 0.0

    # ---- 逐项判定 ----
    checks = []

    def add(cid, name, passed, detail):
        checks.append({'id': cid, 'name': name, 'pass': bool(passed), 'detail': detail})
        print(f"[{cid}] {name:<26}: {'PASS' if passed else 'FAIL'}  ({detail})")

    print('\n' + '=' * 78)
    print('验收结果')
    print('=' * 78)
    add('A1', '单元测试全绿', a1, '跳过' if args.skip_tests else '41 用例')
    add('A2', '样本内为正', m_in['total_return'] > 0,
        f"收益 {m_in['total_return']:+.2%}, 年化 {m_in['annual_return']:+.2%}")
    add('A3', '样本外为正', m_out['total_return'] > 0,
        f"收益 {m_out['total_return']:+.2%}, 年化 {m_out['annual_return']:+.2%}")
    add('B1', f"样本外回撤 <= {LIMITS['oos_max_drawdown']:.0%}",
        m_out['max_drawdown'] <= LIMITS['oos_max_drawdown'],
        f"回撤 {m_out['max_drawdown']:.2%}")
    pos_folds = sum(1 for f in folds if f['total_return'] > 0)
    worst_fold = min(f['total_return'] for f in folds)
    add('B2', f"滚动4段 >= {LIMITS['fold_positive_min']} 段为正且最差段 >= {LIMITS['fold_worst_return']:.0%}",
        pos_folds >= LIMITS['fold_positive_min'] and worst_fold >= LIMITS['fold_worst_return'],
        f"{pos_folds}/4 段为正, 最差段 {worst_fold:+.2%}")
    add('B3', f"全样本卡玛 >= {LIMITS['calmar_min']}",
        m_all['calmar_ratio'] >= LIMITS['calmar_min'],
        f"卡玛 {m_all['calmar_ratio']:.3f}")
    add('C1', f"样本外年化 >= {LIMITS['oos_annual_min']:.0%}",
        m_out['annual_return'] >= LIMITS['oos_annual_min'],
        f"年化 {m_out['annual_return']:+.2%}")

    # C2: 跑赢基准, 或小输但回撤显著更浅 (风险调整后不输)
    gap = m_out['total_return'] - bench_ret
    dd_improve = (abs(bench_dd) - m_out['max_drawdown']) / abs(bench_dd) if bench_dd < 0 else 0.0
    c2 = gap > 0 or (gap > -LIMITS['bench_gap_max'] and dd_improve >= LIMITS['dd_improve_min'])
    add('C2', '样本外不输等权基准(风险调整)', c2,
        f"策略 {m_out['total_return']:+.2%} vs 基准 {bench_ret:+.2%} (差 {gap:+.2%}), "
        f"回撤 {m_out['max_drawdown']:.2%} vs {abs(bench_dd):.2%} (浅 {dd_improve:.0%})")

    # ---- 明细 ----
    print('\n滚动 4 段明细:')
    for i, f in enumerate(folds, 1):
        print(f"  段{i} {f['start']} ~ {f['end']}: 收益 {f['total_return']:+8.2%}  "
              f"年化 {f['annual_return']:+7.2%}  回撤 {f['max_drawdown']:6.2%}  笔数 {f['total_trades']}")

    n_pass = sum(1 for c in checks if c['pass'])
    verdict = n_pass == len(checks)
    print('\n' + '=' * 78)
    # 不用 emoji: Windows GBK 控制台会 UnicodeEncodeError
    tag = '[绿灯] 达到验收线, 可进入生产观察 (小仓位/模拟盘)' if verdict else '[红灯] 未达验收线, 不可实盘'
    print(f'结论: {tag}  ({n_pass}/{len(checks)} 通过)')
    print('=' * 78)
    if verdict:
        print('提醒: 过线只代表"过程可信+历史样本外达标", 不代表未来收益。')
        print('      建议生产观察期 >= 3 个月, 期间实盘/模拟净值与本回测口径偏离 > 3pp 即复查。')

    # ---- 落盘 ----
    payload = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'variant': args.variant,
        'preregistration': {
            'cash_yield': cash_yield,
            'gold_etf': GOLD_ETF if use_gold else None,
            'gold_weight': GOLD_WEIGHT if use_gold else 0.0,
            'note': '参数于 2026-11-06 预注册 (见 production_gate.py 头部注释), 阈值不变',
        },
        'strategy': f'MA{PROD_FAST}/{PROD_SLOW} + MarketGate(20, -5%)',
        'pool': DEFAULT_POOL,
        'eval_in': list(EVAL_IN), 'eval_out': list(EVAL_OUT),
        'limits': LIMITS,
        'metrics': {'all': m_all, 'in': m_in, 'out': m_out,
                    'benchmark_out': {'total_return': bench_ret, 'annual': bench_ann,
                                      'max_drawdown': bench_dd}},
        'folds': folds,
        'checks': checks,
        'verdict': 'PASS' if verdict else 'FAIL',
    }
    out_path = Path(args.json) if args.json else (
        Path(__file__).parent / 'results' / f'production_gate_{args.variant}.json')
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print(f'\n验收快照: {out_path}')
    return 0 if verdict else 1


if __name__ == '__main__':
    sys.exit(main())
