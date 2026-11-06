"""
全球分散实验: L1L2 (A股趋势+黄金) + 美股 QDII 腿

预注册 (2026-11-06, 跑之前写死, 禁止事后调参重挑):
- 美股腿: 513500 博时标普500ETF (场内可买, 不用 yfinance 的 SPY —— 回测标的
  必须是用户真实能交易的品种; 513500 净值含 USD/CNY 汇率效应, 属分散特征)
- 权重: 75% L1L2 + 25% 513500 (25% 取 20~30% 区间中点, 不调参)
- 混合口径: 两腿各自归一后静态加权, 不再平衡 (与 L2 黄金腿同一口径)
- 评估窗口/滚动分段: 与 production_gate 完全一致 (2018 起, OOS 2023-01~2026-10)

采纳线 (先写死后跑, 全过才算该方向成立):
- G1 样本外年化 >= 6%   (必须高于 L1L2 单独的 5.82%, 否则分散没有意义)
- G2 样本外回撤 <= 15%  (L1L2 为 10.22%, 允许因美股腿上升但不超过 15%)
- G3 滚动4段 >= 3 段为正, 且最差段 >= -10%
- G4 全样本卡玛 >= 0.7  (L1L2 为 0.753, 不得明显退化)
- G5 样本外总收益 > 等权A股基准 (保持 C2 精神)

必须披露的偏差:
- QDII 溢价/折价未建模: 场内买入价可能偏离净值 ±1~3%, 实际收益会有噪声
- 513500 走新浪源为未复权价, 分红导致收益轻微低估 (513500 分红很少, 影响小)
- 513500 历史自 2013-07, 覆盖全部评估窗口

用法: python global_blend.py
"""
import sys
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from production_gate import (load_experiment_data, load_gold, run_l1l2,
                             blend_metrics, EVAL_IN, EVAL_OUT, HISTORY_START)
from data_source import DataSource
from portfolio import equal_weight_benchmark
from logging_setup import setup_logging, get_logger

logger = get_logger('global_blend')

# ---- 预注册参数 ----
US_ETF = '513500'
US_WEIGHT = 0.25

LIMITS = {
    'g1_oos_annual_min': 0.06,
    'g2_oos_max_drawdown': 0.15,
    'g3_fold_positive_min': 3,
    'g3_fold_worst_return': -0.10,
    'g4_calmar_min': 0.70,
}


def blend_static(eq_a: pd.Series, eq_b: pd.Series, w: float) -> pd.Series:
    """静态混合: (1-w)*A + w*B, 各自按 A 的窗口起点归一, B 对齐到 A 的日历"""
    a = eq_a / eq_a.iloc[0]
    b = eq_b.reindex(eq_a.index, method='ffill').bfill()
    b = b / b.iloc[0]
    return (1 - w) * a + w * b


def main():
    setup_logging()
    print('=' * 84)
    print(f'全球分散实验  75% L1L2(A股趋势+黄金15%) + 25% {US_ETF}(标普500 QDII)')
    print(f'样本内 {EVAL_IN[0]}~{EVAL_IN[1]}   样本外 {EVAL_OUT[0]}~{EVAL_OUT[1]}')
    print('=' * 84)

    print(f'\n加载数据 ({HISTORY_START} 起)...')
    data, idx = load_experiment_data()
    gold = load_gold()
    us = DataSource(use_cache=True).get_etf_daily(US_ETF, start_date=HISTORY_START)['close']
    print(f'股票 {len(data)} 只, 指数 {len(idx)} 根, 黄金 {len(gold)} 根, '
          f'{US_ETF} {len(us)} 根 ({us.index[0].date()} ~ {us.index[-1].date()})')

    def run_global(start=None, end=None):
        m, eq = run_l1l2(data, idx, gold, start, end, variant='l1l2')
        eq_g = blend_static(eq, us, US_WEIGHT)
        m_g = blend_metrics(m, eq_g)
        return m_g, eq_g, m, eq

    print('\n回测: 全样本 / 样本内 / 样本外 / 滚动4段 ...')
    m_all, eq_all, m_all_leg, eq_all_leg = run_global()
    m_in, _, _, _ = run_global(*EVAL_IN)
    m_out, eq_out, m_out_leg, eq_out_leg = run_global(*EVAL_OUT)

    # 滚动 4 段
    eval_dates = [d for d in idx.index if d >= pd.Timestamp('2018-01-01')]
    n = len(eval_dates)
    folds = []
    for i in range(4):
        s = eval_dates[i * n // 4]
        e = eval_dates[min((i + 1) * n // 4 - 1, n - 1)]
        fm, feq, _, _ = run_global(s, e)
        folds.append({'start': str(pd.Timestamp(s).date()),
                      'end': str(pd.Timestamp(e).date()), **fm})

    # 基准
    bench = equal_weight_benchmark(data, start=EVAL_OUT[0], end=EVAL_OUT[1])
    bench_ret = float(bench.iloc[-1] - 1) if len(bench) else 0.0

    # 分散度诊断 (不是采纳线): OOS 窗口内两腿日收益相关性
    leg_r = eq_out_leg.pct_change().dropna()
    us_r = us.reindex(eq_out_leg.index, method='ffill').pct_change().dropna()
    common = leg_r.index.intersection(us_r.index)
    corr = float(leg_r.loc[common].corr(us_r.loc[common])) if len(common) > 50 else float('nan')

    # ---- 判定 ----
    checks = []

    def add(cid, name, passed, detail):
        checks.append({'id': cid, 'name': name, 'pass': bool(passed), 'detail': detail})
        print(f"[{cid}] {name:<28}: {'PASS' if passed else 'FAIL'}  ({detail})")

    print('\n' + '=' * 84)
    print('采纳线判定 (预注册 G1-G5)')
    print('=' * 84)
    add('G1', f"样本外年化 >= {LIMITS['g1_oos_annual_min']:.0%}",
        m_out['annual_return'] >= LIMITS['g1_oos_annual_min'],
        f"年化 {m_out['annual_return']:+.2%} (L1L2单独 {m_out_leg['annual_return']:+.2%})")
    add('G2', f"样本外回撤 <= {LIMITS['g2_oos_max_drawdown']:.0%}",
        m_out['max_drawdown'] <= LIMITS['g2_oos_max_drawdown'],
        f"回撤 {m_out['max_drawdown']:.2%} (L1L2单独 {m_out_leg['max_drawdown']:.2%})")
    pos_folds = sum(1 for f in folds if f['total_return'] > 0)
    worst_fold = min(f['total_return'] for f in folds)
    add('G3', f"滚动4段 >= {LIMITS['g3_fold_positive_min']} 段为正且最差段 >= {LIMITS['g3_fold_worst_return']:.0%}",
        pos_folds >= LIMITS['g3_fold_positive_min'] and worst_fold >= LIMITS['g3_fold_worst_return'],
        f"{pos_folds}/4 段为正, 最差段 {worst_fold:+.2%}")
    add('G4', f"全样本卡玛 >= {LIMITS['g4_calmar_min']}",
        m_all['calmar_ratio'] >= LIMITS['g4_calmar_min'],
        f"卡玛 {m_all['calmar_ratio']:.3f} (L1L2单独 {m_all_leg['calmar_ratio']:.3f})")
    add('G5', '样本外总收益 > 等权A股基准',
        m_out['total_return'] > bench_ret,
        f"策略 {m_out['total_return']:+.2%} vs 基准 {bench_ret:+.2%}")

    print('\n滚动 4 段明细:')
    for i, f in enumerate(folds, 1):
        print(f"  段{i} {f['start']} ~ {f['end']}: 收益 {f['total_return']:+8.2%}  "
              f"年化 {f['annual_return']:+7.2%}  回撤 {f['max_drawdown']:6.2%}")

    print(f'\n分散度诊断: OOS 窗口两腿日收益相关系数 = {corr:.2f}'
          + ('  (低相关=分散有效)' if corr < 0.5 else '  (相关性偏高, 分散效果有限)'))

    n_pass = sum(1 for c in checks if c['pass'])
    verdict = n_pass == len(checks)
    print('\n' + '=' * 84)
    tag = '[采纳] 全球分散成立, 可并入生产候选' if verdict else '[否决] 全球分散不成立, 如实归档'
    print(f'判定: {tag}  ({n_pass}/{len(checks)})')
    print('=' * 84)

    # 落盘
    out_dir = Path(__file__).parent / 'results'
    out_dir.mkdir(exist_ok=True)
    payload = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'preregistered': {'us_etf': US_ETF, 'us_weight': US_WEIGHT,
                          'limits': LIMITS,
                          'note': '2026-11-06 预注册, 阈值跑前写死'},
        'metrics': {'all': m_all, 'in': m_in, 'out': m_out,
                    'l1l2_leg': {'all': m_all_leg, 'out': m_out_leg},
                    'benchmark_out_total': bench_ret,
                    'oos_leg_correlation': corr},
        'folds': folds,
        'checks': checks,
        'verdict': 'PASS' if verdict else 'FAIL',
    }
    (out_dir / 'global_blend.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    pd.DataFrame({'global': eq_all, 'l1l2_leg': eq_all_leg}).to_csv(
        out_dir / 'global_blend_equity.csv', encoding='utf-8-sig')
    print(f'\n已保存: results/global_blend.json / _equity.csv')
    return 0 if verdict else 1


if __name__ == '__main__':
    sys.exit(main())
