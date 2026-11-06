"""
分散组合 + 类别级 MA60/MA20 闸门 (今日实验的正证据组合)

设计依据 (全部来自本日已归档实验的正证据, 非新发明):
- 分散持有 10 只基金等权: 卡玛 0.96, 是今日所有配置的基准最强者
- "破MA60走/回MA20进" 规则: fund_rules 实验中唯一回撤改善显著的规则
  (中位回撤变浅 9.9pp, 5/10 基金有效)
本实验 = 把后者从"单基金择时"升级为"类别级闸门"加在前者之上:
每个类别的复合净值跌破 MA60 → 该类别整体转现金; 重新站上 MA20 → 回场。
不预测、不轮动、不集中, 只做一件事: 在类别趋势破坏时撤出。

预注册 (2026-11-06, 跑之前写死):
- 组合: 10 只基金动态等权 (与 fund_rotation 基准完全相同)
- 闸门: 类别复合净值 (成员归一后均值) vs 其 MA60/MA20,
  复用 fund_rules_backtest.rule_ma60_exit_ma20_reentry 的精确逻辑
  (离场优先级高于回场, 有单测守护)
- 成交: 逐基金滞后 (QDII T+2, 其余 T+1), 与 fund_rules 同口径
- 费用: 服务费 0.4%/年, 赎回费 0 (类别级持有远超 7 天), 现金 1.5%/年
- 评估: 2023-01-01 起

采纳线 (先写死后跑):
- R1 回撤比买入持有降 >= 5pp (17.37% -> <= 12.4%)
- R2 保留买入持有总收益的 >= 85% (76.98% -> >= 65.4%)
- R3 卡玛 > 买入持有 (0.962)
- R4 年化 >= 10%

用法: python fund_gate_overlay.py
"""
import sys
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from fund_rules_backtest import (get_nav, QDII, SERVICE_FEE, CASH_YIELD,
                                 rule_ma60_exit_ma20_reentry)
from fund_rotation import CATEGORIES, EVAL_START, yearly_table
from metrics import Metrics
from logging_setup import setup_logging, get_logger

logger = get_logger('fund_gate_overlay')

LIMITS = {
    'r1_dd_reduce_min': 0.05,      # 回撤至少降 5pp
    'r2_retain_min': 0.85,         # 至少保留 85% 收益
    'r4_annual_min': 0.10,
}

RESULTS_DIR = Path(__file__).parent / 'results'


def category_composite(nav_panel: pd.DataFrame, codes) -> pd.Series:
    """类别复合净值: 成员各自归一到 1 后取均值 (动态处理成立先后)"""
    members = [c for c in codes if c in nav_panel.columns]
    norm = nav_panel[members].div(nav_panel[members].dropna().iloc[0])
    return norm.mean(axis=1, skipna=True)


def build_weights(nav_panel: pd.DataFrame, dates) -> pd.DataFrame:
    """类别级闸门 -> 逐基金生效权重 (含 T+1/T+2 滞后)"""
    all_codes = list(nav_panel.columns)
    cat_pos = {}
    for cat, codes in CATEGORIES.items():
        comp = category_composite(nav_panel, codes)
        pos = rule_ma60_exit_ma20_reentry(comp)          # 0/1 序列, 复用已测逻辑
        cat_pos[cat] = pos.reindex(nav_panel.index).ffill().fillna(0.0)

    # 每个交易日的目标权重: 活跃类别间等权, 类别内成员等权
    target = pd.DataFrame(0.0, index=nav_panel.index, columns=all_codes)
    for d in nav_panel.index:
        active = [cat for cat, p in cat_pos.items() if float(p.loc[d]) > 0.5]
        if not active:
            continue
        per_cat = 1.0 / len(active)
        for cat in active:
            members = [c for c in CATEGORIES[cat] if c in all_codes]
            if members:
                for c in members:
                    target.at[d, c] = per_cat / len(members)

    # 逐基金滞后生效 (事件表 + ffill)
    events = {c: {} for c in all_codes}
    idx_list = list(nav_panel.index)
    pos_map = {d: i for i, d in enumerate(idx_list)}
    for d in dates:
        i = pos_map[d]
        for c in all_codes:
            lag = 2 if c in QDII else 1
            j = i + lag
            if j < len(idx_list):
                events[c][idx_list[j]] = float(target.at[d, c])
    eff = pd.DataFrame({
        c: pd.Series(ev, dtype=float).reindex(dates).ffill().fillna(0.0)
        for c, ev in events.items()
    })
    return eff


def main():
    setup_logging()
    print('=' * 84)
    print('分散组合 + 类别级 MA60/MA20 闸门 (正证据组合实验)')
    print(f'评估: {EVAL_START} 起   基准: 10只基金动态等权买入持有')
    print('=' * 84)

    navs = {}
    for cat, codes in CATEGORIES.items():
        for code in codes:
            try:
                navs[code] = get_nav(code)
            except Exception as e:
                logger.error('%s 获取失败: %s', code, e)
    nav_panel = pd.DataFrame(navs).sort_index()
    print(f'净值面板: {nav_panel.shape[0]} 天 x {nav_panel.shape[1]} 只')

    dates = nav_panel.index[nav_panel.index >= pd.Timestamp(EVAL_START)]
    rets = nav_panel.pct_change()

    # ---- 策略 ----
    eff_w = build_weights(nav_panel, dates)
    eq, equity = 1.0, pd.Series(index=dates, dtype=float)
    for d in dates:
        w = eff_w.loc[d]
        r = rets.loc[d].fillna(0.0)
        daily = float((w * (r - SERVICE_FEE / 252)).sum())
        invested = float(w.sum())
        daily += (1 - invested) * CASH_YIELD / 252
        eq *= (1 + daily)
        equity.loc[d] = eq

    # ---- 基准 (与 fund_rotation 完全相同) ----
    avail = nav_panel.notna()
    bench_ret = (rets.fillna(0.0) - SERVICE_FEE / 252).mul(avail).sum(axis=1) / avail.sum(axis=1)
    bench_ret = bench_ret[bench_ret.index >= pd.Timestamp(EVAL_START)].fillna(0.0)
    bench = (1 + bench_ret).cumprod()
    bench = bench / bench.iloc[0]

    m = Metrics.calc_all(equity, [], risk_free_rate=CASH_YIELD)
    bm = Metrics.calc_all(bench, [], risk_free_rate=CASH_YIELD)
    yt = yearly_table(equity, bench)

    print('\n' + '=' * 84)
    print('结果')
    print('=' * 84)
    print(f"  {'指标':<12}{'组合+闸门':>12}{'等权持有':>12}")
    print('-' * 40)
    for key, name, is_pct in [('total_return', '总收益', 1), ('annual_return', '年化', 1),
                              ('max_drawdown', '回撤', 1), ('calmar_ratio', '卡玛', 0),
                              ('sharpe_ratio', '夏普', 0)]:
        a, b = m[key], bm[key]
        print(f'  {name:<12}{(f"{a:.2%}" if is_pct else f"{a:.2f}"):>12}'
              f'{(f"{b:.2%}" if is_pct else f"{b:.2f}"):>12}')
    print('\n分年度:')
    for _, r in yt.iterrows():
        print(f"  {int(r['year'])}: 策略 {r['strategy']:+7.2%}  基准 {r['benchmark']:+7.2%}  超额 {r['excess']:+7.2%}")

    dd_reduce = bm['max_drawdown'] - m['max_drawdown']
    retain = m['total_return'] / bm['total_return'] if bm['total_return'] > 0 else 0.0

    checks = []

    def add(cid, name, passed, detail):
        checks.append({'id': cid, 'name': name, 'pass': bool(passed), 'detail': detail})
        print(f"[{cid}] {name:<28}: {'PASS' if passed else 'FAIL'}  ({detail})")

    print('\n采纳线 (预注册 R1-R4):')
    add('R1', f"回撤降幅 >= {LIMITS['r1_dd_reduce_min']:.0%}",
        dd_reduce >= LIMITS['r1_dd_reduce_min'],
        f"降 {dd_reduce:.2%} ({bm['max_drawdown']:.2%} -> {m['max_drawdown']:.2%})")
    add('R2', f"保留收益 >= {LIMITS['r2_retain_min']:.0%}",
        retain >= LIMITS['r2_retain_min'],
        f"保留 {retain:.0%} ({m['total_return']:+.2%} / {bm['total_return']:+.2%})")
    add('R3', '卡玛 > 基准', m['calmar_ratio'] > bm['calmar_ratio'],
        f"{m['calmar_ratio']:.3f} vs {bm['calmar_ratio']:.3f}")
    add('R4', f"年化 >= {LIMITS['r4_annual_min']:.0%}",
        m['annual_return'] >= LIMITS['r4_annual_min'], f"{m['annual_return']:+.2%}")

    n_pass = sum(1 for c in checks if c['pass'])
    verdict = n_pass == len(checks)
    print(f'\n判定: {"[采纳] 组合+闸门成立" if verdict else "[否决] 不成立, 如实归档"} ({n_pass}/{len(checks)})')

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({'gated': equity, 'benchmark': bench}).to_csv(
        RESULTS_DIR / 'fund_gate_overlay_equity.csv', encoding='utf-8-sig')
    (RESULTS_DIR / 'fund_gate_overlay.json').write_text(json.dumps({
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'preregistered': {'limits': LIMITS, 'eval_start': EVAL_START,
                          'gate': 'category-level MA60 exit / MA20 reentry'},
        'strategy': {k: float(v) for k, v in m.items()},
        'benchmark': {k: float(v) for k, v in bm.items()},
        'dd_reduce': float(dd_reduce), 'retain_ratio': float(retain),
        'yearly': yt.to_dict('records'),
        'checks': checks,
        'verdict': 'PASS' if verdict else 'FAIL',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n已保存: results/fund_gate_overlay.json / _equity.csv')
    return 0 if verdict else 1


if __name__ == '__main__':
    sys.exit(main())
