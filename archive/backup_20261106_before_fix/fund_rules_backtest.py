"""
基金规则回测 (量化口径)

纪律:
- 6 条规则全部预注册, 不调参数, 不做事后挑选, 结果全量报告
- 信号按 T 日净值计算 -> 次日净值成交 (T+1); QDII 净值滞后一天 -> T+2
- 真实费率 (C 类口径): 申购费 0; 赎回费 持有<7天 1.5%, 否则 0;
  销售服务费 0.4%/年 (按日计提); 空仓资金按货币基金 1.5%/年计息
- 一致性检验: 分年度对比, 规则必须在多数年份改善才算稳健, 不允许单一年份驱动结论
"""
import sys
import time
from datetime import datetime, timedelta, date
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).parent / 'data' / 'cache'
RESULTS_DIR = Path(__file__).parent / 'results'

FUNDS = [
    ('014409', '创金合信兴选产业趋势', '主动混合'),
    ('025942', '广发新动力混合C', '主动混合'),
    ('018168', '国泰有色矿业ETF联接C', '有色'),
    ('009478', '中银上海金ETF联接C', '黄金'),
    ('004253', '国泰黄金ETF联接C', '黄金'),
    ('018036', '长城全球新能源车股票', 'QDII'),
    ('015831', '平安惠复纯债C', '纯债'),
    ('021190', '南方亚太精选ETF联接', 'QDII'),
    ('012922', '易方达全球成长精选混合', 'QDII'),
    ('017992', '华泰柏瑞致远混合C', '主动混合'),
]
QDII = {'018036', '021190', '012922'}

SERVICE_FEE = 0.004      # C 类销售服务费 / 年
RED_FEE_LT7 = 0.015      # 持有 <7 天赎回费
CASH_YIELD = 0.015       # 空仓资金货基收益 / 年


# ---------- 数据 ----------
def _expected_nav_date() -> date:
    """当前应已公布的最新净值日期 (场外基金 T 日净值通常 20:00 后公布)"""
    now = datetime.now()
    day = now.date() if now.hour >= 20 else now.date() - timedelta(days=1)
    while day.weekday() >= 5:      # 周末回退到最近工作日
        day -= timedelta(days=1)
    return day


def get_nav(code: str) -> pd.Series:
    """带 CSV 缓存的基金净值; 缓存已含应公布的最新净值才复用, 否则重拉"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f'fund_{code}.csv'
    expected = _expected_nav_date()
    if path.exists():
        df = pd.read_csv(path, parse_dates=['date'], index_col='date')
        if len(df) > 0 and df.index[-1].date() >= expected:
            return df['nav'].dropna()
        print(f'[数据] {code} 缓存最新 {df.index[-1].date()} < 应为 {expected}, 重新拉取')
    df = ak.fund_open_fund_info_em(symbol=code, indicator='单位净值走势')
    df['净值日期'] = pd.to_datetime(df['净值日期'])
    df = df.sort_values('净值日期').set_index('净值日期')
    nav = pd.to_numeric(df['单位净值'], errors='coerce').dropna()
    nav.rename('nav').rename_axis('date').to_csv(path, encoding='utf-8-sig')
    time.sleep(0.8)
    return nav


# ---------- 预注册规则: 返回目标仓位序列 (0 或 1), 按 T 日净值计算 ----------
def rule_buy_hold(nav):
    return pd.Series(1.0, index=nav.index)

def rule_ma20(nav):
    return (nav > nav.rolling(20).mean()).astype(float)

def rule_ma60(nav):
    return (nav > nav.rolling(60).mean()).astype(float)

def rule_gold_cross(nav):
    ma20, ma60 = nav.rolling(20).mean(), nav.rolling(60).mean()
    return (ma20 > ma60).astype(float)

def rule_ma60_exit_ma20_reentry(nav):
    """跌破 MA60 离场, 重新站上 MA20 才回 (防抖)"""
    ma20, ma60 = nav.rolling(20).mean(), nav.rolling(60).mean()
    pos = pd.Series(np.nan, index=nav.index)
    pos[nav < ma60] = 0.0
    pos[nav > ma20] = 1.0
    return pos.ffill().fillna(0.0)

def rule_mom20(nav):
    return (nav / nav.shift(20) > 1).astype(float)

RULES = {
    '买入持有': rule_buy_hold,
    'MA20上方': rule_ma20,
    'MA60上方': rule_ma60,
    'MA20>MA60': rule_gold_cross,
    '破MA60走/回MA20进': rule_ma60_exit_ma20_reentry,
    '20日动量>0': rule_mom20,
}


# ---------- 回测 ----------
def backtest(nav: pd.Series, signal: pd.Series, qdii: bool) -> dict:
    lag = 2 if qdii else 1
    pos = signal.shift(lag).fillna(0.0)          # T 信号, T+lag 净值生效
    ret = nav.pct_change().fillna(0.0)

    eq = 1.0
    equity = pd.Series(index=nav.index, dtype=float)
    hold_days = 0
    fees_paid = 0.0
    trades = 0
    prev = 0.0
    for i in range(len(nav)):
        p = float(pos.iloc[i])
        if p < prev:                              # 卖出
            fee = (prev - p) * (RED_FEE_LT7 if hold_days < 7 else 0.0)
            eq *= (1 - fee)
            fees_paid += fee
            trades += 1
            hold_days = 0
        if p > prev:                              # 买入
            trades += 1
            hold_days = 0
        hold_days += 1 if p > 0 else 0
        daily = p * (ret.iloc[i] - SERVICE_FEE / 252) + (1 - p) * (CASH_YIELD / 252)
        eq *= (1 + daily)
        equity.iloc[i] = eq
        prev = p
    return {'equity': equity, 'trades': trades, 'fees': fees_paid, 'pos': pos}


def perf(equity: pd.Series) -> dict:
    r = equity.pct_change().dropna()
    n = len(equity)
    years = n / 252.0
    total = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252 - 0.02) / vol if vol > 0 else np.nan
    mdd = (equity / equity.cummax() - 1).min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    return {'total': total, 'cagr': cagr, 'vol': vol, 'sharpe': sharpe,
            'mdd': mdd, 'calmar': calmar, 'years': years}


def yearly_consistency(equity: pd.Series, bh_equity: pd.Series) -> dict:
    """分年度: 规则跑赢买入持有的年份占比"""
    y_ret = equity.groupby(equity.index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)
    y_bh = bh_equity.groupby(bh_equity.index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)
    win = (y_ret > y_bh)
    return {'years_win': f'{int(win.sum())}/{len(win)}', 'win_ratio': win.mean()}


# ---------- 主流程 ----------
def run_fund_rules():
    print('=' * 100)
    print('基金规则回测  (预注册6条规则, T+1/T+2净值成交, C类费率: 申购0/赎回<7天1.5%/服务费0.4%年, 空仓1.5%年)')
    print('=' * 100)

    navs = {}
    for code, name, ftype in FUNDS:
        try:
            navs[code] = get_nav(code)
            print(f'[数据] {code} {name}: {len(navs[code])}条, '
                  f'{navs[code].index[0].date()} ~ {navs[code].index[-1].date()}')
        except Exception as e:
            print(f'[数据] {code} {name}: 获取失败 {type(e).__name__}: {e}')

    rows = []
    for code, name, ftype in FUNDS:
        if code not in navs:
            continue
        nav = navs[code]
        bh_eq = backtest(nav, rule_buy_hold(nav), code in QDII)['equity']
        bh_perf = perf(bh_eq)
        for rname, rfn in RULES.items():
            bt = backtest(nav, rfn(nav), code in QDII)
            p = perf(bt['equity'])
            cons = yearly_consistency(bt['equity'], bh_eq)
            rows.append({
                'code': code, 'name': name, 'type': ftype, 'rule': rname,
                'total': p['total'], 'cagr': p['cagr'], 'vol': p['vol'],
                'sharpe': p['sharpe'], 'mdd': p['mdd'], 'calmar': p['calmar'],
                'trades': bt['trades'], 'fees': bt['fees'],
                'bh_total': bh_perf['total'], 'bh_calmar': bh_perf['calmar'],
                'bh_mdd': bh_perf['mdd'],
                'calmar_delta': p['calmar'] - bh_perf['calmar'],
                'mdd_delta': p['mdd'] - bh_perf['mdd'],   # >0 表示回撤变浅
                'years_win': cons['years_win'], 'win_ratio': cons['win_ratio'],
                'start': nav.index[0].date(), 'end': nav.index[-1].date(),
            })

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / 'fund_rules_detail.csv', index=False, encoding='utf-8-sig')

    # ---- 规则级汇总: 中位 Calmar 增量 / 回撤改善 / 一致性 ----
    alt = df[df['rule'] != '买入持有']
    summary = alt.groupby('rule').agg(
        funds=('code', 'nunique'),
        calmar_delta_med=('calmar_delta', 'median'),
        mdd_delta_med=('mdd_delta', 'median'),
        cagr_med=('cagr', 'median'),
        trades_med=('trades', 'median'),
        win_ratio_med=('win_ratio', 'median'),
        improved=('calmar_delta', lambda s: (s > 0).sum()),
    ).reset_index()
    summary['improved'] = summary['improved'].astype(str) + '/' + summary['funds'].astype(str)
    print('\n【规则汇总】(calmar_delta>0 = 风险调整后收益优于买入持有)')
    print(summary[['rule', 'improved', 'calmar_delta_med', 'mdd_delta_med',
                   'cagr_med', 'trades_med', 'win_ratio_med']].to_string(
        index=False, float_format=lambda x: f'{x:+.3f}'))

    # ---- 每只基金的最优规则 (按 Calmar) ----
    print('\n【每只基金的最优规则】')
    for (code, name), g in df.groupby(['code', 'name']):
        best = g.loc[g['calmar'].idxmax()]
        bh = g[g['rule'] == '买入持有'].iloc[0]
        flag = '优于持有' if best['calmar'] > bh['calmar'] else '仍不如持有'
        print(f"  {code} {name:<16} 最优: {best['rule']:<14} "
              f"Calmar {best['calmar']:.2f} vs 持有 {bh['calmar']:.2f}  "
              f"回撤 {best['mdd']:.1%} vs {bh['mdd']:.1%}  [{flag}]")

    print(f'\n已保存: results/fund_rules_detail.csv')
    return df, summary


def main():
    run_fund_rules()


if __name__ == '__main__':
    main()
