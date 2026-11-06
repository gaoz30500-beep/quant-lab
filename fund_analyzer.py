"""
场外基金技术分析
输入基金代码, 输出净值趋势指标与下一交易日倾向
注意: 基金净值 T+1 公布, 周末无数据
"""
import sys
import akshare as ak
import pandas as pd
import numpy as np


def get_nav(code, days=250):
    df = ak.fund_open_fund_info_em(symbol=code, indicator='单位净值走势')
    df['净值日期'] = pd.to_datetime(df['净值日期'])
    df = df.sort_values('净值日期').set_index('净值日期')
    df['close'] = pd.to_numeric(df['单位净值'], errors='coerce')
    return df.tail(days)


def rsi(close, n=14):
    d = close.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = -d.clip(upper=0).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def analyze(code):
    name_df = ak.fund_name_em()
    row = name_df[name_df['基金代码'].astype(str).str.zfill(6) == code.zfill(6)]
    name = row['基金简称'].iloc[0] if not row.empty else '未知'
    ftype = row['基金类型'].iloc[0] if not row.empty else '未知'

    df = get_nav(code)
    c = df['close'].dropna()
    ret = np.log(c).diff()

    ma5 = c.rolling(5).mean().iloc[-1]
    ma10 = c.rolling(10).mean().iloc[-1]
    ma20 = c.rolling(20).mean().iloc[-1]
    ma60 = c.rolling(60).mean().iloc[-1]
    last = c.iloc[-1]
    prev = c.iloc[-2]

    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9).mean()
    hist = dif - dea

    r = rsi(c).iloc[-1]
    vol20 = ret.tail(20).std() * np.sqrt(252)
    mom5 = last / c.iloc[-6] - 1
    mom20 = last / c.iloc[-21] - 1
    dd60 = last / c.tail(60).max() - 1

    # 趋势评分
    score = 50
    score += 15 if last > ma20 else -15
    score += 15 if ma20 > ma60 else -15
    score += 10 if last > ma5 else -5
    score += 10 if hist.iloc[-1] > 0 else -10
    score += 10 if mom5 > 0 else -10
    score += 10 if r < 35 else (-10 if r > 70 else 0)
    score = max(0, min(100, score))

    # 历史条件统计: 收盘价低于MA20时, 次日上涨比例
    below = c < c.rolling(20).mean()
    next_up = (c.shift(-1) / c - 1) > 0
    cond_n = int(below.sum())
    cond_up = float(next_up[below].mean()) if cond_n > 0 else float('nan')
    # 连续下跌后次日表现
    downchain = (c.diff() < 0) & (c.diff().shift(1) < 0) & (c.diff().shift(2) < 0)
    n2 = int(downchain.sum())
    up2 = float(next_up[downchain].mean()) if n2 > 0 else float('nan')

    print("=" * 68)
    print(f"{name} ({code})  {ftype}")
    print("=" * 68)
    print(f"  最新净值日期: {c.index[-1].date()}   净值: {last:.4f}   "
          f"日涨跌: {(last/prev-1)*100:+.2f}%")
    print(f"  数据区间: {c.index[0].date()} ~ {c.index[-1].date()}  ({len(c)}个交易日)")
    print("-" * 68)
    print(f"  MA5      {ma5:.4f}   当前{'高于' if last > ma5 else '低于'}MA5  "
          f"({(last/ma5-1)*100:+.2f}%)")
    print(f"  MA10     {ma10:.4f}   当前{'高于' if last > ma10 else '低于'}MA10  "
          f"({(last/ma10-1)*100:+.2f}%)")
    print(f"  MA20     {ma20:.4f}   当前{'高于' if last > ma20 else '低于'}MA20  "
          f"({(last/ma20-1)*100:+.2f}%)")
    print(f"  MA60     {ma60:.4f}   当前{'高于' if last > ma60 else '低于'}MA60  "
          f"({(last/ma60-1)*100:+.2f}%)")
    print(f"  RSI(14)  {r:.1f}  {'超卖' if r < 30 else ('偏弱' if r < 45 else ('中性' if r < 60 else ('偏强' if r < 70 else '超买')))}")
    print(f"  MACD     柱状 {hist.iloc[-1]:+.5f}  {'多头' if hist.iloc[-1] > 0 else '空头'}")
    print(f"  5日动量  {mom5:+.2%}    20日动量 {mom20:+.2%}")
    print(f"  20日年化波动 {vol20:.1%}   距60日高点 {dd60:+.2%}")
    print("-" * 68)
    print(f"  趋势评分: {score}/100  ->  "
          f"{'偏多' if score >= 65 else ('中性偏空' if score < 45 else '中性')}")

    print("-" * 68)
    print("  历史同类状态统计 (仅参考, 样本有限)")
    print(f"    收盘低于MA20的天数 {cond_n} 天, 其中次日上涨占比 "
          f"{cond_up:.1%}" if cond_n else "    数据不足")
    if n2:
        print(f"    连续3日下跌 {n2} 次, 次日上涨占比 {up2:.1%}")
    return df


if __name__ == '__main__':
    code = sys.argv[1] if len(sys.argv) > 1 else '025924'
    analyze(code)
