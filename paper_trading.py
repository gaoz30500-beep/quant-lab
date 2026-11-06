"""
模拟盘闭环 (Paper Trading Loop)

把"信号 -> 实际操作 -> 净值对比"串成闭环, 让生产观察期可度量:

    python paper_trading.py signal                 # 每日净值公布后: 记录规则信号
    python paper_trading.py fill --follow-signal   # 记录"按信号执行了"
    python paper_trading.py fill --weights "A股权益=0.6,纯债=0.4"   # 记录实际仓位
    python paper_trading.py report                 # 对比: 实际 vs 信号 vs 回测预期

口径:
- 类别权重: A股权益/黄金/有色/海外QDII/纯债, 未分配部分 = 现金 (1.5%/年)
- 信号组合: 每日按规则闸门的目标权重 (与 fund_gate_overlay 同一逻辑)
- 实际组合: 两次 fill 之间权重不变, 按类别复合收益逐日盯市
- 偏离 > 3pp 提示 (执行纪律问题或规则失效的早期信号)
"""
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from fund_rules_backtest import get_nav, SERVICE_FEE, CASH_YIELD
from fund_rotation import CATEGORIES
from fund_gate_overlay import build_weights, category_composite
from logging_setup import setup_logging, get_logger

logger = get_logger('paper_trading')

PAPER_DIR = Path(__file__).parent / 'results' / 'paper'
SIGNAL_LOG = PAPER_DIR / 'signal_log.jsonl'
FILL_LOG = PAPER_DIR / 'fill_log.jsonl'
DIVERGENCE_ALERT_PP = 0.03     # 偏离告警阈值 3pp


def _load_nav_panel() -> pd.DataFrame:
    navs = {}
    for codes in CATEGORIES.values():
        for c in codes:
            try:
                navs[c] = get_nav(c)
            except Exception as e:
                logger.error('%s 获取失败: %s', c, e)
    return pd.DataFrame(navs).sort_index()


def _cat_returns(nav_panel: pd.DataFrame) -> pd.DataFrame:
    """各类别复合日收益 (成员等权, 动态可用)"""
    out = {}
    for cat, codes in CATEGORIES.items():
        comp = category_composite(nav_panel, codes)
        out[cat] = comp.pct_change()
    return pd.DataFrame(out)


def _today_weights(nav_panel: pd.DataFrame) -> dict:
    """今日规则信号的目标类别权重"""
    dates = nav_panel.index[-60:]
    eff = build_weights(nav_panel, dates)
    w = eff.iloc[-1]
    cat_w = {}
    for cat, codes in CATEGORIES.items():
        cat_w[cat] = float(w[[c for c in codes if c in w.index]].sum())
    return cat_w


def cmd_signal():
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    nav_panel = _load_nav_panel()
    cat_w = _today_weights(nav_panel)
    entry = {
        'date': str(nav_panel.index[-1].date()),
        'recorded_at': datetime.now().isoformat(timespec='seconds'),
        'cat_weights': cat_w,
        'cash': float(1.0 - sum(cat_w.values())),
    }
    with open(SIGNAL_LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    held = [c for c, w in cat_w.items() if w > 0]
    print(f"信号已记录 ({entry['date']}): 持有 {', '.join(held) if held else '空仓'}, "
          f"现金 {entry['cash']:.0%}")
    for c, w in cat_w.items():
        print(f"  {c:<8} {w:>6.1%}")


def cmd_fill(args):
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    if args.follow_signal:
        if not SIGNAL_LOG.exists():
            print('FATAL: 无信号记录, 先运行 signal')
            return 1
        last = json.loads(SIGNAL_LOG.read_text(encoding='utf-8').strip().splitlines()[-1])
        weights, source = last['cat_weights'], 'follow_signal'
    else:
        if not args.weights:
            print('FATAL: 需要 --weights "类别=权重,..." 或 --follow-signal')
            return 1
        weights = {}
        for part in args.weights.split(','):
            k, v = part.split('=')
            k = k.strip()
            if k not in CATEGORIES:
                print(f'FATAL: 未知类别 {k}, 可选: {list(CATEGORIES)}')
                return 1
            weights[k] = float(v)
        total = sum(weights.values())
        if total > 1.0 + 1e-9:
            print(f'FATAL: 权重合计 {total:.2%} > 100%')
            return 1
        for c in CATEGORIES:
            weights.setdefault(c, 0.0)
        source = 'manual'

    nav_panel = _load_nav_panel()
    entry = {
        'date': str(nav_panel.index[-1].date()),
        'recorded_at': datetime.now().isoformat(timespec='seconds'),
        'cat_weights': weights,
        'cash': float(1.0 - sum(weights.values())),
        'source': source,
    }
    with open(FILL_LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    held = [f"{c} {w:.0%}" for c, w in weights.items() if w > 0]
    print(f"持仓已记录 ({entry['date']}, {source}): {', '.join(held) if held else '空仓'}, "
          f"现金 {entry['cash']:.0%}")
    return 0


def _read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').strip().splitlines()]


def cmd_report():
    signals = _read_jsonl(SIGNAL_LOG)
    fills = _read_jsonl(FILL_LOG)
    if not signals and not fills:
        print('无记录: 先运行 signal / fill')
        return 1

    nav_panel = _load_nav_panel()
    cat_r = _cat_returns(nav_panel) - SERVICE_FEE / 252
    cash_r = CASH_YIELD / 252

    def nav_from_entries(entries, use_signal_dates):
        """按记录逐日推进净值 (记录之间权重不变)"""
        if not entries:
            return None
        start = pd.Timestamp(entries[0]['date'])
        dates = cat_r.index[cat_r.index >= start]
        eq, w, cash = 1.0, None, 1.0
        eqs = {}
        by_date = {e['date']: e for e in entries}
        for d in dates:
            ds = str(d.date())
            if ds in by_date:
                e = by_date[ds]
                w = pd.Series({c: e['cat_weights'].get(c, 0.0) for c in CATEGORIES})
                cash = e.get('cash', 1.0 - float(w.sum()))
            if w is None:
                continue
            r = cat_r.loc[d].fillna(0.0)
            daily = float((w * r).sum()) + cash * cash_r
            eq *= (1 + daily)
            eqs[d] = eq
        return pd.Series(eqs)

    sig_nav = nav_from_entries(signals, True)
    act_nav = nav_from_entries(fills, False)

    print('=' * 78)
    print('模拟盘报告')
    print('=' * 78)
    for name, nav in [('信号组合(按规则)', sig_nav), ('实际组合(你的操作)', act_nav)]:
        if nav is None or len(nav) < 2:
            print(f'  {name}: 记录不足')
            continue
        n = nav / nav.iloc[0]
        total = float(n.iloc[-1] - 1)
        dd = float((n / n.cummax() - 1).min())
        years = len(n) / 252.0
        annual = (1 + total) ** (1 / years) - 1 if years > 0 else 0.0
        print(f'  {name}: {n.index[0].date()} ~ {n.index[-1].date()}  '
              f'收益 {total:+.2%}  年化 {annual:+.2%}  回撤 {abs(dd):.2%}')

    if sig_nav is not None and act_nav is not None:
        common = sig_nav.index.intersection(act_nav.index)
        if len(common) > 1:
            s = sig_nav.loc[common] / sig_nav.loc[common[0]]
            a = act_nav.loc[common] / act_nav.loc[common[0]]
            div = float(a.iloc[-1] - s.iloc[-1])
            flag = '⚠ 偏离超过3pp, 检查执行纪律或规则有效性' if abs(div) > DIVERGENCE_ALERT_PP \
                else '偏离在正常范围'
            print(f'\n  实际 vs 信号 累计偏离: {div:+.2%}  ({flag})')

    # 回测预期对照
    exp_path = Path(__file__).parent / 'results' / 'fund_gate_overlay.json'
    if exp_path.exists():
        exp = json.loads(exp_path.read_text(encoding='utf-8'))
        print(f"\n  回测预期 (决策时): 年化 {exp['strategy']['annual_return']:+.2%}, "
              f"回撤 {exp['strategy']['max_drawdown']:.2%}")
        print('  注: 回测含顺风期运气, 实盘/模拟低于预期属正常; 持续大幅低于才需复查')
    return 0


def main():
    ap = argparse.ArgumentParser(description='模拟盘闭环')
    sub = ap.add_subparsers(dest='cmd')
    sub.add_parser('signal', help='记录今日规则信号')
    f = sub.add_parser('fill', help='记录实际持仓')
    f.add_argument('--weights', help='"A股权益=0.6,纯债=0.4" (未列类别=0)')
    f.add_argument('--follow-signal', action='store_true', help='记录为按最新信号执行')
    sub.add_parser('report', help='对比报告')
    args = ap.parse_args()
    setup_logging()

    if args.cmd == 'signal':
        return cmd_signal()
    if args.cmd == 'fill':
        return cmd_fill(args)
    if args.cmd == 'report':
        return cmd_report()
    ap.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
