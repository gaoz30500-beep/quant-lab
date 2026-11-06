"""
冻结窗口验证协议 (Frozen Window Protocol)

原则: 今天做决策用到的所有数据 (截至数据截止日) 都是"样本内"。
截止日次日起的数据是冻结窗口 —— 它的表现只能在未来计算,
是判断"策略是否依然有效"的唯一干净证据。

用法:
    python frozen_protocol.py snapshot     # 决策日记录快照 (只跑一次)
    python frozen_protocol.py check        # 新数据积累后, 检验冻结窗口实际表现

预注册验收带 (2026-11-06 写死, 不得修改):
    P1 冻结窗口实际收益 > -10% (绝对底线)
    P2 冻结窗口实际最大回撤 <= 决策时样本外回撤 x 2
    P3 数据管道与信号正常 (能完整跑通)
任一不过 → 红灯: 暂停生产加仓, 人工复查。
"""
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from production_gate import (load_experiment_data, load_gold, run_l1l2,
                             EVAL_IN, EVAL_OUT)
from fund_gate_overlay import compute as fund_compute
from logging_setup import setup_logging, get_logger

logger = get_logger('frozen_protocol')

RESULTS_DIR = Path(__file__).parent / 'results'
SNAPSHOT_PATH = RESULTS_DIR / 'frozen_snapshot.json'

# ---- 预注册验收带 ----
BANDS = {
    'p1_min_return': -0.10,       # 冻结窗口收益底线
    'p2_dd_multiplier': 2.0,      # 回撤容忍 = 决策时OOS回撤 x 2
}


def cmd_snapshot():
    """记录决策时快照: 配置 + 指标 + 数据截止日 (冻结窗口从次日起)"""
    print('记录决策时快照 ...')

    # ---- A股侧 (L1L2) ----
    data, idx = load_experiment_data()
    gold = load_gold()
    m_out, eq_out = run_l1l2(data, idx, gold, *EVAL_OUT, variant='l1l2')
    a_share_end = max(data[s].index[-1] for s in data)
    a_share_end = max(a_share_end, idx.index[-1], gold.index[-1])

    # ---- 基金侧 (分散+闸门) ----
    fres = fund_compute()
    fund_end = fres['nav_panel'].index[-1]

    cutoff = min(a_share_end, fund_end)   # 保守: 取两侧数据截止的较早者
    snapshot = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'data_cutoff': str(cutoff.date()),
        'frozen_window_start': str((cutoff + pd.Timedelta(days=1)).date()),
        'bands': BANDS,
        'a_share': {
            'config': 'L1L2: MA20/150+闸门, 黄金15%, 现金3%',
            'oos_window': [EVAL_OUT[0], EVAL_OUT[1]],
            'oos_total_return': float(m_out['total_return']),
            'oos_annual_return': float(m_out['annual_return']),
            'oos_max_drawdown': float(m_out['max_drawdown']),
            'oos_calmar': float(m_out['calmar_ratio']),
        },
        'fund': {
            'config': '10基分散等权 + 类别MA60/MA20闸门',
            'eval_start': '2023-01-01',
            'total_return': float(fres['m']['total_return']),
            'annual_return': float(fres['m']['annual_return']),
            'max_drawdown': float(fres['m']['max_drawdown']),
            'calmar': float(fres['m']['calmar_ratio']),
        },
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2),
                             encoding='utf-8')
    print(f"数据截止: {snapshot['data_cutoff']}  冻结窗口自 {snapshot['frozen_window_start']} 起")
    print(f"A股侧 OOS: 年化 {snapshot['a_share']['oos_annual_return']:+.2%}, "
          f"回撤 {snapshot['a_share']['oos_max_drawdown']:.2%}")
    print(f"基金侧:    年化 {snapshot['fund']['annual_return']:+.2%}, "
          f"回撤 {snapshot['fund']['max_drawdown']:.2%}")
    print(f'快照已保存: {SNAPSHOT_PATH}')
    print('下次运行 check 时, 将用截止日之后的新数据检验策略是否依然有效。')


def _window_perf(equity: pd.Series, start_after: pd.Timestamp) -> dict:
    """计算冻结窗口 (start_after 之后) 的实际表现"""
    w = equity[equity.index > start_after]
    if len(w) < 2:
        return {'bars': len(w)}
    w = w / w.iloc[0]
    dd = float((w / w.cummax() - 1).min())
    total = float(w.iloc[-1] - 1)
    years = len(w) / 252.0
    annual = (1 + total) ** (1 / years) - 1 if years > 0 else 0.0
    return {'bars': len(w), 'total_return': total, 'annual_return': annual,
            'max_drawdown': abs(dd),
            'start': str(w.index[0].date()), 'end': str(w.index[-1].date())}


def cmd_check():
    """用截止日之后的新数据检验冻结窗口表现"""
    if not SNAPSHOT_PATH.exists():
        print('FATAL: 无快照, 先运行 snapshot')
        return 1
    snap = json.loads(SNAPSHOT_PATH.read_text(encoding='utf-8'))
    cutoff = pd.Timestamp(snap['data_cutoff'])
    print(f"快照日期 {snap['created_at'][:10]}, 数据截止 {snap['data_cutoff']}, "
          f"验收带 {snap['bands']}")

    checks = []

    def add(cid, name, passed, detail):
        checks.append({'id': cid, 'name': name, 'pass': bool(passed), 'detail': detail})
        print(f"[{cid}] {name:<24}: {'PASS' if passed else 'FAIL'}  ({detail})")

    # ---- A股侧 ----
    try:
        data, idx = load_experiment_data()
        gold = load_gold()
        _, eq_full = run_l1l2(data, idx, gold, variant='l1l2')
        perf_a = _window_perf(eq_full, cutoff)
        p3a = True
    except Exception as e:
        logger.error('A股侧管道异常: %s', e)
        perf_a, p3a = {'bars': 0}, False

    # ---- 基金侧 ----
    try:
        fres = fund_compute()
        perf_f = _window_perf(fres['equity'], cutoff)
        p3f = True
    except Exception as e:
        logger.error('基金侧管道异常: %s', e)
        perf_f, p3f = {'bars': 0}, False

    print('\n' + '=' * 78)
    print('冻结窗口检验')
    print('=' * 78)
    if perf_a.get('bars', 0) < 2:
        print(f"A股侧: 冻结窗口数据不足 ({perf_a.get('bars', 0)} 根), 暂无法检验 "
              f"(建议至少积累 1 个月)")
    else:
        dd_limit_a = snap['a_share']['oos_max_drawdown'] * snap['bands']['p2_dd_multiplier']
        add('P1-A', f"收益 > {snap['bands']['p1_min_return']:.0%}",
            perf_a['total_return'] > snap['bands']['p1_min_return'],
            f"{perf_a['total_return']:+.2%} ({perf_a['start']}~{perf_a['end']})")
        add('P2-A', f"回撤 <= {dd_limit_a:.0%}",
            perf_a['max_drawdown'] <= dd_limit_a,
            f"{perf_a['max_drawdown']:.2%}")
    if perf_f.get('bars', 0) < 2:
        print(f"基金侧: 冻结窗口数据不足 ({perf_f.get('bars', 0)} 根), 暂无法检验")
    else:
        dd_limit_f = snap['fund']['max_drawdown'] * snap['bands']['p2_dd_multiplier']
        add('P1-F', f"收益 > {snap['bands']['p1_min_return']:.0%}",
            perf_f['total_return'] > snap['bands']['p1_min_return'],
            f"{perf_f['total_return']:+.2%} ({perf_f['start']}~{perf_f['end']})")
        add('P2-F', f"回撤 <= {dd_limit_f:.0%}",
            perf_f['max_drawdown'] <= dd_limit_f,
            f"{perf_f['max_drawdown']:.2%}")
    add('P3', '数据管道与信号正常', p3a and p3f,
        'A股侧+基金侧均可完整运行' if (p3a and p3f) else '存在管道异常')

    n_pass = sum(1 for c in checks if c['pass'])
    verdict = n_pass == len(checks)
    print(f"\n判定: {'[正常] 策略在冻结窗口依然有效' if verdict else '[红灯] 暂停生产加仓, 人工复查'}"
          f"  ({n_pass}/{len(checks)})")

    out = RESULTS_DIR / f"frozen_check_{datetime.now():%Y%m%d}.json"
    out.write_text(json.dumps({
        'checked_at': datetime.now().isoformat(timespec='seconds'),
        'snapshot': snap, 'a_share': perf_a, 'fund': perf_f,
        'checks': checks, 'verdict': 'PASS' if verdict else 'FAIL',
    }, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print(f'检验快照: {out}')
    return 0 if verdict else 1


def main():
    ap = argparse.ArgumentParser(description='冻结窗口验证协议')
    ap.add_argument('action', choices=['snapshot', 'check'])
    args = ap.parse_args()
    setup_logging()
    if args.action == 'snapshot':
        return cmd_snapshot()
    return cmd_check()


if __name__ == '__main__':
    sys.exit(main())
