"""
大类资产类别扫描 (找"候选区", 不是荐基)

用与闸门相同的纪律扫描主要资产类别:
- 趋势完好 (收盘价 > MA60) 且 不过热 (RSI < 70)  -> 候选区 (可以考虑, 非推荐)
- 趋势完好 但 过热 (RSI >= 70)                    -> 持有区 (有就拿着, 别追)
- 趋势破坏 (收盘价 < MA60)                        -> 回避区 (等站回 MA60)

扫描对象为指数/ETF (类别层面), 不挑主动基金经理 —— 避免"按历史业绩挑基金"
的幸存者偏差。候选区类别若想买, 选跟踪它的低费率指数基金即可。

披露: 这是信息扫描工具, 不是已采纳的交易规则; 任何新仓位都应小步、可承受、
并纳入每日信号跟踪。扫描结果不构成收益承诺。

用法: python category_screen.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from data_source import DataSource
from logging_setup import setup_logging, get_logger

logger = get_logger('category_screen')

# ---- 扫描池: 大类指数 (A股) + 黄金ETF ----
A_INDEXES = {
    '沪深300': '000300', '中证500': '000905', '创业板指': '399006',
    '科创50': '000688', '中证红利': '000922', '北证50': '899050',
    '证券公司': '399975', '中证银行': '399986', '中证医药': '399933',
    '中证军工': '399967', '中证传媒': '399971', '主要消费': '000932',
}
GOLD_ETF = '518880'
# 港股指数 (akshare 独立接口, 失败则跳过)
HK_INDEXES = {'恒生指数': 'HSI', '恒生科技': 'HSTECH'}


def _rsi(close: pd.Series, window: int = 14) -> float:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = -d.clip(upper=0).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return float(rsi.iloc[-1])


def _screen_one(name: str, close: pd.Series) -> dict:
    if len(close.dropna()) < 70:
        return {'name': name, 'error': '数据不足'}
    c = close.dropna()
    last = float(c.iloc[-1])
    ma20 = float(c.rolling(20).mean().iloc[-1])
    ma60 = float(c.rolling(60).mean().iloc[-1])
    rsi = _rsi(c)
    mom20 = last / float(c.iloc[-21]) - 1
    hi60 = float(c.rolling(60).max().iloc[-1])
    above60 = last > ma60
    if above60 and rsi < 70:
        zone = '候选区'
    elif above60:
        zone = '持有区(别追)'
    else:
        zone = '回避区'
    return {'name': name, 'date': str(c.index[-1].date()), 'close': last,
            'vs_ma20': last / ma20 - 1, 'vs_ma60': last / ma60 - 1,
            'rsi': rsi, 'mom20': mom20, 'vs_hi60': last / hi60 - 1, 'zone': zone}


def main():
    setup_logging()
    print('=' * 96)
    print('大类资产扫描  (趋势>MA60 且 RSI<70 = 候选区; 趋势在但超买 = 别追; 破趋势 = 回避)')
    print('=' * 96)

    ds = DataSource(use_cache=True)
    rows = []

    for name, code in A_INDEXES.items():
        try:
            df = ds.get_index_daily(code, start_date='20250101')
            rows.append(_screen_one(name, df['close']))
        except Exception as e:
            logger.warning('%s(%s) 获取失败: %s', name, code, str(e)[:50])

    try:
        gold = ds.get_etf_daily(GOLD_ETF, start_date='20250101')
        rows.append(_screen_one('黄金ETF', gold['close']))
    except Exception as e:
        logger.warning('黄金ETF 获取失败: %s', str(e)[:50])

    try:
        import akshare as ak
        for name, sym in HK_INDEXES.items():
            try:
                hk = ak.stock_hk_index_daily_em(symbol=sym)
                hk['date'] = pd.to_datetime(hk['date'])
                hk = hk.sort_values('date').set_index('date')
                rows.append(_screen_one(name, hk['close']))
            except Exception as e:
                logger.warning('%s 获取失败: %s', name, str(e)[:50])
    except Exception:
        pass

    df = pd.DataFrame([r for r in rows if 'error' not in r])
    order = {'候选区': 0, '持有区(别追)': 1, '回避区': 2}
    df['_o'] = df['zone'].map(order)
    df = df.sort_values(['_o', 'rsi'])

    print(f"\n{'类别':<10}{'净值日期':<12}{'距MA20':>8}{'距MA60':>8}{'RSI':>7}{'20日动量':>9}{'距60日高':>9}  分区")
    print('-' * 96)
    for _, r in df.iterrows():
        print(f"{r['name']:<10}{r['date']:<12}{r['vs_ma20']:>+7.2%}{r['vs_ma60']:>+7.2%}"
              f"{r['rsi']:>7.1f}{r['mom20']:>+8.2%}{r['vs_hi60']:>+8.2%}  {r['zone']}")

    cands = df[df['zone'] == '候选区']['name'].tolist()
    print('\n' + '=' * 96)
    if cands:
        print(f"候选区类别: {', '.join(cands)}")
        print('说明: 候选区 = 趋势完好且不过热, 是"规则上可以考虑"的区域, 不是买入推荐。')
        print('      若参与, 选跟踪该类别的低费率指数基金, 小仓位、可承受亏损, 并纳入每日跟踪。')
    else:
        print('当前没有类别处于候选区 —— 规则的答案是: 没有值得追的, 现金等待。')
    hot = df[df['zone'] == '持有区(别追)']['name'].tolist()
    if hot:
        print(f"持有区(别追): {', '.join(hot)} —— 趋势在但已超买, 有仓拿着, 没仓别追。")
    print('=' * 96)


if __name__ == '__main__':
    main()
