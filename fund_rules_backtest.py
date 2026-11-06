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

sys.path.insert(0, str(Path(__file__).parent))
from metrics import Metrics
from logging_setup import get_logger

logger = get_logger(__name__)

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
    """
    带 CSV 缓存的基金净值; 缓存已含应公布的最新净值才复用, 否则重拉

    失败策略 (商业化口径):
    - 缓存损坏/为空 → 忽略缓存, 走网络重拉, 不中断
    - 网络拉取失败但本地有旧缓存 → 降级用旧缓存并 WARNING (数据滞后好过整批失败)
    - 网络失败且无缓存 → 抛出带基金代码的明确异常, 由上层按只跳过
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f'fund_{code}.csv'
    expected = _expected_nav_date()

    cached = None
    if path.exists():
        try:
            df = pd.read_csv(path, parse_dates=['date'], index_col='date')
            nav = df['nav'].dropna()
            if len(nav) > 0:
                if nav.index[-1].date() >= expected:
                    return nav
                logger.info('%s 缓存最新 %s < 应为 %s, 重新拉取', code, nav.index[-1].date(), expected)
                cached = nav
        except Exception as e:
            logger.warning('%s 缓存损坏 (%s), 忽略并重新拉取', code, e)

    try:
        df = ak.fund_open_fund_info_em(symbol=code, indicator='单位净值走势')
        df['净值日期'] = pd.to_datetime(df['净值日期'])
        df = df.sort_values('净值日期').set_index('净值日期')
        nav = pd.to_numeric(df['单位净值'], errors='coerce').dropna()
        if len(nav) == 0:
            raise ValueError('接口返回空净值序列')
        nav.rename('nav').rename_axis('date').to_csv(path, encoding='utf-8-sig')
        time.sleep(0.8)
        return nav
    except Exception as e:
        if cached is not None and len(cached) > 0:
            logger.warning('%s 拉取失败 (%s), 降级使用旧缓存至 %s',
                           code, str(e)[:80], cached.index[-1].date())
            return cached
        raise RuntimeError(f'基金 {code} 净值获取失败且无可用缓存: {e}') from e


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
    """
    跌破 MA60 离场, 重新站上 MA20 才回 (防抖)

    注意赋值顺序: 离场条件必须后写。
    旧实现先写离场(pos[nav<ma60]=0) 再写回场(pos[nav>ma20]=1),
    于是"跌破 MA60 但仍高于 MA20"这一档被回场条件覆盖, 等于从来没有离场。
    """
    ma20, ma60 = nav.rolling(20).mean(), nav.rolling(60).mean()
    pos = pd.Series(np.nan, index=nav.index)
    pos[nav > ma20] = 1.0     # 回场条件 (优先级低)
    pos[nav < ma60] = 0.0     # 离场条件 (优先级高, 必须覆盖前者)
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
    equity.iloc[0] = 1.0                        # 起点净值固定为 1, 总收益才不含首日缺失
    hold_days = 0
    fees_paid = 0.0
    trades = 0
    prev = float(pos.iloc[0]) if len(nav) else 0.0
    for i in range(1, len(nav)):
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


def perf(equity: pd.Series, risk_free: float = CASH_YIELD) -> dict:
    """
    统一走 metrics.Metrics 的口径

    旧实现自带一套公式: 夏普用 (mean*252 - 0.02)/vol (算术年化 + 固定 2% 无风险),
    年化用 equity**(1/years) (几何)。两者混用会让同一策略在不同模块里算出不同的
    "年化/夏普", 无法横向比较。这里直接复用 Metrics, 全项目一个口径。
    """
    m = Metrics.calc_all(equity, [], risk_free_rate=risk_free)
    return {
        'total': m['total_return'],
        'cagr': m['annual_return'],
        'vol': m['annual_volatility'],
        'sharpe': m['sharpe_ratio'],
        'mdd': -m['max_drawdown'],          # 保留负号, 便于和旧输出对照
        'calmar': m['calmar_ratio'],
        'years': len(equity) / 252.0,
    }


def yearly_consistency(equity: pd.Series, bh_equity: pd.Series) -> dict:
    """
    分年度: 规则跑赢买入持有的年份占比

    年度收益以上一年末净值为基准。旧写法 groupby(year).apply(首末比) 用
    "当年第一个交易日" 当基准, 跨年那一天的收益既不算进上一年也不算进下一年。
    """
    eq_y = equity.groupby(equity.index.year).last()
    bh_y = bh_equity.groupby(bh_equity.index.year).last()

    def _year_returns(last_by_year, full):
        r = last_by_year / last_by_year.shift(1) - 1
        if len(r):
            r.iloc[0] = last_by_year.iloc[0] / full.iloc[0] - 1
        return r

    y_ret = _year_returns(eq_y, equity)
    y_bh = _year_returns(bh_y, bh_equity)
    win = (y_ret > y_bh).reindex(y_ret.index).fillna(False).astype(bool)
    return {'years_win': f'{int(win.sum())}/{len(win)}',
            'win_ratio': float(win.mean()) if len(win) else 0.0}


# ---------- 主流程 ----------
def run_fund_rules():
    print('=' * 100)
    print('基金规则回测  (预注册6条规则, T+1/T+2净值成交, C类费率: 申购0/赎回<7天1.5%/服务费0.4%年, 空仓1.5%年)')
    print('=' * 100)

    MIN_NAV_LEN = 70   # MA60 规则需要至少 60 根预热 + 若干评估点
    navs = {}
    for code, name, ftype in FUNDS:
        try:
            nav = get_nav(code)
        except Exception as e:
            logger.error('%s %s: 获取失败 %s: %s', code, name, type(e).__name__, e)
            continue
        if len(nav) < MIN_NAV_LEN:
            logger.warning('%s %s: 净值仅%d条 (< %d), 指标预热不足, 跳过',
                           code, name, len(nav), MIN_NAV_LEN)
            continue
        navs[code] = nav
        print(f'[数据] {code} {name}: {len(nav)}条, '
              f'{nav.index[0].date()} ~ {nav.index[-1].date()}')

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
    # 基金代码是 6 位字符串, 前导零必须保留 (否则 014409 会变成 14409)
    df['code'] = df['code'].astype(str).str.zfill(6)
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

    # ---- 各规则"夺魁"次数: 说明"最优规则"的稳定性, 而不是给出一张选择建议表 ----
    best_per_fund = df.loc[df.groupby('code')['calmar'].idxmax()]
    win_count = best_per_fund['rule'].value_counts()
    print('\n【各规则在基金间夺魁次数 (事后统计, 不能当作选规则依据)】')
    for rule, cnt in win_count.items():
        print(f'  {rule:<20} {cnt} 只')
    print('  注: 这是样本内事后挑选。选规则应看上方"规则汇总"的 improved 与 win_ratio 中位数,'
          '\n      单只基金的"最优规则"只代表它在历史上拟合得最好, 不构成未来收益预期。')

    print(f'\n已保存: results/fund_rules_detail.csv (code 列为 6 位字符串, 直接读请指定 dtype=str)')
    return df, summary


def main():
    run_fund_rules()


if __name__ == '__main__':
    main()
