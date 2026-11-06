"""
今日操作参考 (每日信号)

输出两部分:
1. 类别闸门状态 —— 采用 fund_gate_overlay 的预注册规则:
   类别复合净值 跌破MA60 -> 该类别应离场(转现金); 站回MA20 -> 回场。
   单日涨跌不触发规则, 只有均线关系变化才触发 (趋势纪律, 不追单日波动)。
2. 各基金诊断 —— 最新净值、距 MA20/MA60 距离、RSI、动量、倾向。

口径: 信号基于最新已公布净值 (场外基金通常 T 日 20:00 后公布),
      操作在下一交易日净值成交 (T+1, QDII T+2)。

用法: python today_signal.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from fund_rules_backtest import get_nav, QDII, rule_ma60_exit_ma20_reentry
from fund_rotation import CATEGORIES
from fund_gate_overlay import category_composite
from logging_setup import setup_logging, get_logger

logger = get_logger('today_signal')

FUND_NAMES = {
    '014409': '创金合信兴选产业趋势', '025942': '广发新动力混合C',
    '018168': '国泰有色矿业ETF联接C', '009478': '中银上海金ETF联接C',
    '004253': '国泰黄金ETF联接C', '018036': '长城全球新能源车股票',
    '015831': '平安惠复纯债C', '021190': '南方亚太精选ETF联接',
    '012922': '易方达全球成长精选混合', '017992': '华泰柏瑞致远混合C',
}


def main():
    setup_logging()
    print('=' * 92)
    print('今日操作参考  (信号基于最新已公布净值; 成交在下一交易日净值, QDII 再延一天)')
    print('=' * 92)

    navs = {}
    for codes in CATEGORIES.values():
        for c in codes:
            try:
                navs[c] = get_nav(c)
            except Exception as e:
                logger.error('%s 获取失败: %s', c, e)

    # ---- 1. 类别闸门 (采用规则) ----
    print('\n【类别闸门】(采用规则: 破MA60离场 / 回MA20回场; 单日涨跌不触发)')
    print('-' * 92)
    print(f"{'类别':<10}{'闸门状态':<10}{'复合净值':>10}{'MA20':>10}{'MA60':>10}{'距MA60':>9}")
    for cat, codes in CATEGORIES.items():
        panel = pd.DataFrame({c: navs[c] for c in codes if c in navs})
        comp = category_composite(panel, codes)
        if len(comp.dropna()) < 61:
            print(f"{cat:<10}{'数据不足':<10}")
            continue
        last = float(comp.iloc[-1])
        ma20 = float(comp.rolling(20).mean().iloc[-1])
        ma60 = float(comp.rolling(60).mean().iloc[-1])
        pos = float(rule_ma60_exit_ma20_reentry(comp).iloc[-1])
        state = '持有' if pos > 0.5 else '离场(现金)'
        dist60 = last / ma60 - 1
        print(f"{cat:<10}{state:<10}{last:>10.4f}{ma20:>10.4f}{ma60:>10.4f}{dist60:>+8.2%}")

    # ---- 2. 各基金诊断 ----
    print('\n【各基金诊断】')
    print('-' * 92)
    print(f"{'代码':<8}{'名称':<20}{'净值日期':<12}{'净值':>9}{'距MA20':>8}{'距MA60':>8}{'RSI':>6}{'20日动量':>9}{'倾向':>8}")
    for cat, codes in CATEGORIES.items():
        for c in codes:
            if c not in navs:
                continue
            nav = navs[c]
            last = float(nav.iloc[-1])
            ma20 = float(nav.rolling(20).mean().iloc[-1])
            ma60 = float(nav.rolling(60).mean().iloc[-1]) if len(nav) >= 60 else float('nan')
            d = nav.diff()
            gain = d.clip(lower=0).rolling(14).mean()
            loss = -d.clip(upper=0).rolling(14).mean()
            rsi = float((100 - 100 / (1 + gain / loss.replace(0, float('nan')))).iloc[-1])
            mom20 = last / float(nav.iloc[-21]) - 1 if len(nav) > 21 else float('nan')

            score = 50
            score += 15 if last > ma20 else -15
            score += 15 if (ma60 == ma60 and ma20 > ma60) else -15
            score += 10 if mom20 > 0 else -10
            verdict = '偏多' if score >= 65 else ('中性' if score >= 45 else '偏空')

            print(f"{c:<8}{FUND_NAMES.get(c, ''):<20}{str(nav.index[-1].date()):<12}"
                  f"{last:>9.4f}{last/ma20-1:>+7.2%}"
                  f"{(last/ma60-1 if ma60 == ma60 else float('nan')):>+7.2%}"
                  f"{rsi:>6.1f}{mom20:>+8.2%}{verdict:>8}")

    print('\n注: 以上为规则信号与诊断, 非收益承诺。闸门规则的价值在危机保护,')
    print('    震荡市会付出收益代价 (见 fund_gate_overlay 实验: 2023/2024 超额为负)。')


if __name__ == '__main__':
    main()
