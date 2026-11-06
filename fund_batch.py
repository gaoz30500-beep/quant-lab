"""
批量基金趋势诊断
输入基金代码清单, 输出最新净值、均线位置、RSI、动量与下一交易日倾向
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import akshare as ak
import pandas as pd
import numpy as np

from logging_setup import get_logger

logger = get_logger(__name__)


FUNDS = [
    ('014409', '创金合信兴选产业趋势', 'A股权益'),
    ('025942', '广发新动力混合C', 'A股权益'),
    ('018168', '国泰有色矿业ETF联接C', '商品-有色'),
    ('009478', '中银上海金ETF联接C', '商品-黄金'),
    ('004253', '国泰黄金ETF联接C', '商品-黄金'),
    ('018036', '长城全球新能源车股票', '海外-QDII'),
    ('015831', '平安惠复纯债C', '债券'),
    ('021190', '南方亚太精选ETF联接', '海外-QDII'),
    ('012922', '易方达全球成长精选', '海外-QDII'),
    ('017992', '华泰柏瑞致远混合C', 'A股权益'),
]


def get_nav(code):
    for i in range(2):
        try:
            df = ak.fund_open_fund_info_em(symbol=code, indicator='单位净值走势')
            df['净值日期'] = pd.to_datetime(df['净值日期'])
            df = df.sort_values('净值日期').set_index('净值日期')
            c = pd.to_numeric(df['单位净值'], errors='coerce').dropna()
            if len(c) == 0:
                raise ValueError('接口返回空净值序列')
            return c
        except Exception as e:
            if i == 1:
                logger.error('%s 净值获取失败 (已重试): %s', code, str(e)[:80])
            time.sleep(1.5)
    return None


def diagnose(c):
    last = c.iloc[-1]
    ma5 = c.rolling(5).mean().iloc[-1]
    ma20 = c.rolling(20).mean().iloc[-1]
    ma60 = c.rolling(60).mean().iloc[-1] if len(c) >= 60 else float('nan')

    d = c.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = -d.clip(upper=0).rolling(14).mean()
    rsi = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).iloc[-1]

    mom5 = last / c.iloc[-6] - 1 if len(c) > 6 else float('nan')
    mom20 = last / c.iloc[-21] - 1 if len(c) > 21 else float('nan')

    score = 50
    score += 15 if last > ma20 else -15
    score += 15 if (not np.isnan(ma60) and ma20 > ma60) else -15
    score += 10 if last > ma5 else -5
    score += 10 if mom5 > 0 else -10
    score += 10 if rsi < 35 else (-10 if rsi > 70 else 0)
    score = max(0, min(100, score))

    below = c < c.rolling(20).mean()
    nxt = (c.shift(-1) / c - 1) > 0
    n_below = int(below.sum())
    p_up = float(nxt[below].mean()) if n_below else float('nan')

    return dict(last=last, ma5=ma5, ma20=ma20, ma60=ma60, rsi=rsi,
                mom5=mom5, mom20=mom20, score=score, p_up=p_up, n_below=n_below)


def verdict(d):
    s = d['score']
    if s >= 70:
        return '偏多'
    if s >= 55:
        return '中性偏多'
    if s >= 45:
        return '中性'
    if s >= 30:
        return '中性偏空'
    return '偏空'


def main():
    print("=" * 96)
    print("10 只基金趋势诊断 (用于下一交易日倾向判断)")
    print("=" * 96)
    print(f"{'代码':<8}{'名称':<22}{'类别':<10}{'净值日期':<12}{'净值':>8}"
          f"{'RSI':>6}{'5日':>8}{'20日':>8}{'评分':>5}{'倾向':>8}")
    print("-" * 96)

    results = []
    for code, name, cat in FUNDS:
        c = get_nav(code)
        if c is None or len(c) < 20:
            print(f"{code:<8}{name:<22}{cat:<10} 数据不足")
            continue
        d = diagnose(c)
        results.append((code, name, cat, c.index[-1], d))
        print(f"{code:<8}{name:<20}{cat:<10}{c.index[-1].date()!s:<12}"
              f"{d['last']:>8.4f}{d['rsi']:>6.1f}"
              f"{d['mom5']*100:>7.2f}%{d['mom20']*100:>7.2f}%"
              f"{d['score']:>5.0f}{verdict(d):>8}")

    print("-" * 96)
    print("\n分类小结")
    for cat in ['A股权益', '商品-黄金', '商品-有色', '海外-QDII', '债券']:
        group = [r for r in results if r[2] == cat]
        if not group:
            continue
        avg = np.mean([r[4]['score'] for r in group])
        print(f"  {cat:<10} {len(group)} 只, 平均评分 {avg:.0f} -> {verdict({'score': avg})}")

    print("\n" + "=" * 96)
    print("各基金次日上涨的历史概率 (收盘低于MA20时, 仅供参考)")
    print("=" * 96)
    for code, name, cat, dt, d in results:
        if not np.isnan(d['p_up']):
            print(f"  {code} {name:<20} 低于MA20共 {d['n_below']:>3} 天, "
                  f"次日上涨占比 {d['p_up']:.1%}")


if __name__ == '__main__':
    main()
