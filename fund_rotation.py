"""
基金类别轮动实验 (截面方法论用在基金上)

核心思想: 不在单只基金上择时 (数据已证明做不到), 而是在资产类别间轮动 ——
每月持有 60 日动量最强的 2 个类别。这与股票截面选股同一哲学: 赚"相对强弱"的钱。

预注册 (2026-11-06, 跑之前写死, 禁止事后调参):
- 类别池 (用现有 10 只基金):
    A股权益: 014409, 025942, 017992
    黄金:    009478, 004253
    有色:    018168
    海外QDII: 018036, 021190, 012922
    纯债:    015831
- 动量: 60 日收益 (类别 = 成员基金 60 日收益的均值, 动态处理成立先后)
- 持仓: 动量前 2 类别, 类别间等权, 类别内成员等权
- 调仓: 每 21 个交易日; 信号 T 日净值 -> 逐基金滞后成交 (QDII T+2, 其余 T+1,
  与 fund_rules_backtest 同口径)
- 费用: C类销售服务费 0.4%/年按日计提; 赎回费 0 (最短持有 21 天 > 7 天免赎回费);
  空仓资金 1.5%/年 (轮动满仓, 实际几乎无空仓)
- 评估起点: 2023-01-01 (此前类别覆盖不全)
- 基准: 10 只基金动态等权买入持有 (同样扣服务费)

采纳线 (先写死后跑, 全过才成立):
- F1 评估期总收益 > 基准
- F2 最大回撤 < 基准 (轮动的核心价值是降回撤)
- F3 分年度超额为正 >= 2/3 年 (2023 / 2024 / 2025+)
- F4 卡玛比率 > 基准
- F5 年化收益 >= 6%

披露: 基金成立日不同导致早期类别覆盖不全 (动态处理, 不回填);
      类别内等权未做波动率调整。

用法: python fund_rotation.py
"""
import sys
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from fund_rules_backtest import get_nav, QDII, SERVICE_FEE, CASH_YIELD
from metrics import Metrics
from logging_setup import setup_logging, get_logger

logger = get_logger('fund_rotation')

# ---- 预注册参数 ----
CATEGORIES = {
    'A股权益': ['014409', '025942', '017992'],
    '黄金': ['009478', '004253'],
    '有色': ['018168'],
    '海外QDII': ['018036', '021190', '012922'],
    '纯债': ['015831'],
}
MOM_WINDOW = 60
TOP_CATS = 2
REBALANCE_DAYS = 21
EVAL_START = '2023-01-01'

LIMITS = {
    'f5_annual_min': 0.06,
    'f3_positive_years_min': 2,
}

RESULTS_DIR = Path(__file__).parent / 'results'


def category_momentum(nav_panel: pd.DataFrame) -> pd.DataFrame:
    """各类别 60 日动量 = 成员基金 60 日收益的均值 (动态处理成立先后)"""
    mom60 = nav_panel.pct_change(MOM_WINDOW)
    cols = {}
    for cat, codes in CATEGORIES.items():
        present = [c for c in codes if c in mom60.columns]
        cols[cat] = mom60[present].mean(axis=1, skipna=True) if present \
            else pd.Series(np.nan, index=mom60.index)
    return pd.DataFrame(cols)


def build_effective_weights(nav_panel: pd.DataFrame, dates) -> pd.DataFrame:
    """
    预计算每个交易日的生效权重

    信号日 T (每 21 个交易日) 按类别动量选前 2 类; 每只基金按各自滞后
    (QDII T+2, 其余 T+1) 在生效日切换到新权重 (可为 0 = 卖出), 持有到下次切换。
    """
    mom = category_momentum(nav_panel)
    all_codes = list(nav_panel.columns)
    events = {c: {} for c in all_codes}

    for k, d in enumerate(dates):
        if k % REBALANCE_DAYS != 0:
            continue
        m = mom.loc[d].dropna()
        if len(m) < TOP_CATS:
            continue
        top_cats = m.nlargest(TOP_CATS).index.tolist()
        w = pd.Series(0.0, index=all_codes)
        per_cat = 1.0 / len(top_cats)
        for cat in top_cats:
            members = [c for c in CATEGORIES[cat] if c in all_codes]
            if members:
                for c in members:
                    w[c] = per_cat / len(members)
        for c in all_codes:
            lag = 2 if c in QDII else 1
            if k + lag < len(dates):
                events[c][dates[k + lag]] = float(w[c])

    eff = pd.DataFrame({
        c: pd.Series(ev, dtype=float).reindex(dates).ffill().fillna(0.0)
        for c, ev in events.items()
    })
    return eff


def yearly_table(equity: pd.Series, bench: pd.Series) -> pd.DataFrame:
    rows = []
    for year in sorted(set(equity.index.year)):
        e = equity[equity.index.year == year]
        b = bench[bench.index.year == year]
        e_prev = equity[equity.index < e.index[0]]
        b_prev = bench[bench.index < b.index[0]]
        e0 = e_prev.iloc[-1] if len(e_prev) else e.iloc[0]
        b0 = b_prev.iloc[-1] if len(b_prev) else b.iloc[0]
        er, br = e.iloc[-1] / e0 - 1, b.iloc[-1] / b0 - 1
        rows.append({'year': year, 'strategy': er, 'benchmark': br, 'excess': er - br})
    return pd.DataFrame(rows)


def main():
    setup_logging()
    print('=' * 84)
    print(f'基金类别轮动  动量{MOM_WINDOW}日  持有前{TOP_CATS}类别  每{REBALANCE_DAYS}交易日调仓')
    print(f'费用: 服务费{SERVICE_FEE:.1%}/年, 赎回费0 (持有>7天), 空仓{CASH_YIELD:.1%}/年')
    print(f'评估: {EVAL_START} 起   基准: 10只基金动态等权买入持有')
    print('=' * 84)

    # ---- 数据 ----
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

    # ---- 回测 ----
    eff_w = build_effective_weights(nav_panel, dates)
    eq, equity = 1.0, pd.Series(index=dates, dtype=float)
    for d in dates:
        w = eff_w.loc[d]
        r = rets.loc[d].fillna(0.0)
        daily = float((w * (r - SERVICE_FEE / 252)).sum())
        invested = float(w.sum())
        daily += (1 - invested) * CASH_YIELD / 252
        eq *= (1 + daily)
        equity.loc[d] = eq

    # ---- 基准: 动态等权买入持有 (同样扣服务费) ----
    avail = nav_panel.notna()
    bench_ret = (rets.fillna(0.0) - SERVICE_FEE / 252).mul(avail).sum(axis=1) / avail.sum(axis=1)
    bench_ret = bench_ret[bench_ret.index >= pd.Timestamp(EVAL_START)].fillna(0.0)
    bench = (1 + bench_ret).cumprod()
    bench = bench / bench.iloc[0]

    # ---- 指标 ----
    m = Metrics.calc_all(equity, [], risk_free_rate=CASH_YIELD)
    bm = Metrics.calc_all(bench, [], risk_free_rate=CASH_YIELD)
    yt = yearly_table(equity, bench)
    pos_years = int((yt['excess'] > 0).sum())

    print('\n' + '=' * 84)
    print('结果')
    print('=' * 84)
    print(f"  {'指标':<12}{'轮动':>12}{'等权持有':>12}")
    print('-' * 40)
    for key, name, is_pct in [('total_return', '总收益', 1), ('annual_return', '年化', 1),
                              ('max_drawdown', '回撤', 1), ('calmar_ratio', '卡玛', 0),
                              ('sharpe_ratio', '夏普', 0)]:
        a, b = m[key], bm[key]
        print(f'  {name:<12}{(f"{a:.2%}" if is_pct else f"{a:.2f}"):>12}'
              f'{(f"{b:.2%}" if is_pct else f"{b:.2f}"):>12}')
    print('\n分年度:')
    for _, r in yt.iterrows():
        print(f"  {int(r['year'])}: 轮动 {r['strategy']:+7.2%}  基准 {r['benchmark']:+7.2%}  超额 {r['excess']:+7.2%}")

    # ---- 采纳线 ----
    checks = []

    def add(cid, name, passed, detail):
        checks.append({'id': cid, 'name': name, 'pass': bool(passed), 'detail': detail})
        print(f"[{cid}] {name:<28}: {'PASS' if passed else 'FAIL'}  ({detail})")

    print('\n采纳线 (预注册 F1-F5):')
    add('F1', '总收益 > 基准', m['total_return'] > bm['total_return'],
        f"{m['total_return']:+.2%} vs {bm['total_return']:+.2%}")
    add('F2', '回撤 < 基准', m['max_drawdown'] < bm['max_drawdown'],
        f"{m['max_drawdown']:.2%} vs {bm['max_drawdown']:.2%}")
    add('F3', f"超额为正年份 >= {LIMITS['f3_positive_years_min']}/3",
        pos_years >= LIMITS['f3_positive_years_min'], f"{pos_years}/{len(yt)} 年")
    add('F4', '卡玛 > 基准', m['calmar_ratio'] > bm['calmar_ratio'],
        f"{m['calmar_ratio']:.3f} vs {bm['calmar_ratio']:.3f}")
    add('F5', f"年化 >= {LIMITS['f5_annual_min']:.0%}",
        m['annual_return'] >= LIMITS['f5_annual_min'], f"{m['annual_return']:+.2%}")

    n_pass = sum(1 for c in checks if c['pass'])
    verdict = n_pass == len(checks)
    print(f'\n判定: {"[采纳] 类别轮动成立" if verdict else "[否决] 不成立, 如实归档"} ({n_pass}/{len(checks)})')

    # ---- 落盘 ----
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({'rotation': equity, 'benchmark': bench}).to_csv(
        RESULTS_DIR / 'fund_rotation_equity.csv', encoding='utf-8-sig')
    (RESULTS_DIR / 'fund_rotation.json').write_text(json.dumps({
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'preregistered': {'categories': CATEGORIES, 'mom_window': MOM_WINDOW,
                          'top_cats': TOP_CATS, 'rebalance_days': REBALANCE_DAYS,
                          'eval_start': EVAL_START, 'limits': LIMITS},
        'strategy': {k: float(v) for k, v in m.items()},
        'benchmark': {k: float(v) for k, v in bm.items()},
        'yearly': yt.to_dict('records'),
        'checks': checks,
        'verdict': 'PASS' if verdict else 'FAIL',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n已保存: results/fund_rotation.json / _equity.csv')
    return 0 if verdict else 1


if __name__ == '__main__':
    sys.exit(main())
