"""
截面多因子选股 (指数增强风格) —— 方法论向头部量化私募靠近的第一块骨架

与现有"12只蓝筹+择时"的本质区别:
- 赚"截面相对强弱"的钱: 每期在宽股票池里按因子得分排序, 持有前 N 名
- 考核口径是相对基准的超额收益/跟踪误差 (指数增强产品口径), 不是绝对收益
- 股票池用指数成分 (宽、规则化), 顺带减轻手选 12 只的幸存者偏差

预注册纪律 (2026-11-06 写死, 跑完不许回头调参重挑):
- 因子: 12-2 动量 / 5 日反转 / 60 日低波, 三个经典价量因子, 截面百分位等权合成
  (等权=零拟合; 不做 IC 加权/ML 权重, 那是过拟合重灾区)
- 调仓: 每 21 个交易日, T-1 收盘得分 -> T 日生效 (滞后一天, 无未来函数)
- 持仓: 前 30 名等权; 成本按换手率 x 0.26% (佣金双边+印花税+滑点)
- 评估: 2019-01-01 起 (之前 2 年留给因子预热)

必须披露的偏差 (免费的代价, 无法消除只能诚实):
- 成分股取"当前"沪深300名单 -> 幸存者偏差, 结果偏乐观
- 向量化近似: 日收益加权=每日再平衡近似, 未建模停牌/涨跌停无法成交
  (生产口径的 engine/portfolio_engine 有完整建模, 本模块定位是研究级筛选)
- 后复权价, 分红再投资按净值口径处理

用法:
    python cross_section.py --limit 40     # 小池冒烟 (先验证管道)
    python cross_section.py                # 全量沪深300
"""
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from data_source import DataSource
from metrics import Metrics
from logging_setup import setup_logging, get_logger

logger = get_logger('cross_section')

# ---- 预注册参数 ----
UNIVERSE_INDEX = '000300'      # 股票池 = 沪深300 成分
DATA_START = '20170101'        # 因子预热 2 年
EVAL_START = '2019-01-01'      # 评估起点
TOP_N = 30                     # 持仓数 (迭代1)
REBALANCE_DAYS = 21            # 月度调仓 (交易日)
COST_RATE = 0.0026             # 单边合计成本 x 换手率
MIN_BARS = 250                 # 入选池子的最少K线数
FACTOR_NAMES = ['mom_120_20', 'rev_5', 'low_vol_60']

# ---- 迭代2 预注册 (2026-11-06, 迭代1结果出来后、跑迭代2之前写死) ----
# 迭代1 (TOP_N=30) 诊断: IR 0.27 弱, TE 11.4% 过高 (30只等权=风格暴露而非纯alpha),
#   2024/2025 超额为负 (因子近年衰减)。
# 迭代2 修正: TOP_N 30 -> 100 (分散个股噪声, TE 应向真正指增收敛), 其余全不动。
# 采纳线 (先写死后跑, 全过才进入下一步研究, 任一不过即停止该方向):
#   P1 超额为正年份 >= 5/8
#   P2 信息比率 IR >= 0.3
#   P3 跟踪误差 TE <= 8%
#   P4 2024+2025+2026 三年超额合计 > -5% (近年不能持续失血)
# 迭代2 结果 (2026-11-06): 3/4 → 按规则否决。P1 7/8 PASS, P2 IR=0.70 PASS,
#   P4 -4.03% PASS, P3 TE=8.66% FAIL (差 0.66pp)。
#   诊断: TE 超标是"等权 vs 市值加权"的结构性风格偏差, 非因子失效;
#   2025 单年超额 -11.58% 是主要风险点 (因子拥挤/衰减), 如实记录。
ITER2_TOP_N = 100

# ---- 迭代3 预注册 (2026-11-06, 迭代2否决后、跑迭代3之前写死) ----
# 修正对象: 迭代2 唯一失败项 P3 (结构性 TE)。因子/池/调仓/成本全不动。
# 唯一变更: 权重从等权 -> 反波动率加权 (w ∝ 1/vol60, vol 用 T-1 前数据),
#   单票上限 = 2x 等权 (防集中)。零拟合参数 (60日窗口与 low_vol_60 因子同源)。
# 采纳线: 沿用 P1-P4 原值不改 (改阈值=移动球门, 禁止)。
# 多重检验披露: 这是同一家族的第 3 次尝试, 假阳性风险随迭代上升;
#   若过线, 采纳前还需一次"冻结协议"复核 (换随机种子无关, 但要看 2025 年
#   单年超额是否仍 < -8% —— 若是, 视为因子衰减证据, 降级为研究观察仓)。
# 迭代3 结果 (2026-11-06): 3/4 → 否决, 且全面弱于迭代2 (年化 9.85% vs 11.51%,
#   IR 0.41 vs 0.70, 超额为正 5/8 vs 7/8, TE 10.10% vs 8.66% 反而更高)。
#   诊断: 低波倾斜本身就是相对市值加权基准的风格赌注, 2019/2020 牛市跑输;
#   无市值数据做真正的权重优化, TE<=8% 在此工具链下结构性不可达。
#   结论: 按纪律停止该方向, 三次迭代全部如实归档 (results/cross_section_*.json)。
#   首跑还抓到并修复一个实现 bug: 先 clip 后归一会让 invvol 静默退化为等权
#   (回归测试 tests/test_cross_section.py 守护)。
ITER3_WEIGHTING = 'invvol'
ITER3_SINGLE_CAP_X = 2.0        # 单票权重上限 = 2x 等权
ITER3_VOL_WINDOW = 60

RESULTS_DIR = Path(__file__).parent / 'results'


def hs300_codes() -> list:
    """取沪深300当前成分股 (披露: 非点时序, 有幸存者偏差)"""
    import akshare as ak
    for attempt in range(1, 4):
        try:
            df = ak.index_stock_cons(symbol=UNIVERSE_INDEX)
            col = '品种代码' if '品种代码' in df.columns else df.columns[0]
            codes = sorted({str(c).zfill(6) for c in df[col]})
            logger.info('沪深300成分股: %d 只', len(codes))
            return codes
        except Exception as e:
            logger.warning('成分股获取第%d次失败: %s', attempt, e)
            time.sleep(2 * attempt)
    raise RuntimeError('无法获取沪深300成分股 (akshare 不可用?)')


def load_universe(codes, ds: DataSource) -> pd.DataFrame:
    """加载收盘价面板 (列=代码, 行=日期; 缺失=停牌/未上市)"""
    panel = {}
    for i, code in enumerate(codes):
        try:
            df = ds.get_stock_daily(code, start_date=DATA_START, adjust='hfq')
            if df is not None and len(df) >= MIN_BARS:
                panel[code] = df['close']
        except Exception as e:
            logger.warning('%s 跳过: %s', code, str(e)[:60])
        if (i + 1) % 50 == 0:
            logger.info('进度 %d/%d, 已入选 %d', i + 1, len(codes), len(panel))
    if len(panel) < 10:
        raise RuntimeError(f'可用股票仅 {len(panel)} 只, 无法做截面')
    return pd.DataFrame(panel)


def factor_scores(close: pd.DataFrame) -> pd.DataFrame:
    """
    截面因子得分 (0~1 百分位, 越高越好), 只用 T 日及以前数据

    - mom_120_20: 120 日前至 20 日前的收益 (经典 12-2 动量, 跳过近月反转区)
    - rev_5:      近 5 日收益取负 (A股短期反转显著)
    - low_vol_60: 60 日日波动取负 (低波异象, A股长期稳健)
    """
    mom = close.shift(20) / close.shift(120) - 1
    rev = -(close / close.shift(5) - 1)
    lowvol = -close.pct_change().rolling(60).std()

    score = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    n_ok = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for f in (mom, rev, lowvol):
        r = f.rank(axis=1, pct=True)          # 截面百分位, 天然抗极值
        ok = r.notna()
        score += r.fillna(0.0)
        n_ok += ok.astype(float)
    # 只对因子齐全的求平均; 缺任一因子的置 NaN (不参与排序)
    score = score.where(n_ok == len(FACTOR_NAMES)) / len(FACTOR_NAMES)
    return score


def _invvol_weights(top, vol_row: pd.Series, cap: float) -> pd.Series:
    """
    反波动率权重: w ∝ 1/vol, 单票封顶 cap, 迭代 clip+归一直到全部达标

    vol 缺失的票用中位权重兜底 (预热期后极少发生); 全缺则退化为等权。
    """
    v = vol_row.reindex(top)
    w = 1.0 / v.where(v > 0)
    w = w.dropna()
    if len(w) == 0:
        out = pd.Series(1.0 / len(top), index=top)
        return out
    missing = [c for c in top if c not in w.index]
    if missing:
        med = float(w.median())
        for c in missing:
            w[c] = med
    # 必须先归一再封顶: 1/vol 原始量纲远大于是 cap, 先 clip 会把所有票削平=等权
    # (2026-11-06 实际踩过: 迭代3首跑结果与迭代2逐位相同, 即此 bug)
    w = w / w.sum()
    for _ in range(20):                      # 封顶 -> 归一, 水床效应收敛为止
        if float(w.max()) <= cap + 1e-9:
            break
        w = w.clip(upper=cap)
        w = w / w.sum()
    return w


def backtest(close: pd.DataFrame, score: pd.DataFrame,
             bench: pd.Series, top_n: int = TOP_N,
             weighting: str = 'equal') -> dict:
    """
    月度调仓截面回测 (向量化近似)

    T-1 收盘得分 -> T 日持仓生效; 持有期内按"每日等权再平衡"近似
    (月度内权重漂移 <3%, 近似误差远小于因子噪声, 已披露)

    weighting: 'equal' 等权 (迭代1/2) | 'invvol' 反波动率+单票封顶 (迭代3, 预注册)
    """
    ret = close.pct_change().fillna(0.0)
    score_lag = score.shift(1)                # 关键: 得分滞后一天, 无未来函数
    vol_lag = None
    if weighting == 'invvol':
        vol_lag = close.pct_change().rolling(ITER3_VOL_WINDOW).std().shift(1)

    dates = close.index[close.index >= pd.Timestamp(EVAL_START)]
    if len(dates) < REBALANCE_DAYS * 3:
        raise RuntimeError('评估窗口太短')

    eq, prev_w = 1.0, None
    equity = pd.Series(index=dates, dtype=float)
    weights_now = pd.Series(0.0, index=close.columns)
    n_rebal, turnover_sum = 0, 0.0
    hold_counts = []

    for k, d in enumerate(dates):
        if k % REBALANCE_DAYS == 0:
            s = score_lag.loc[d].dropna()
            if len(s) >= top_n:
                top = s.nlargest(top_n).index
                weights_now = pd.Series(0.0, index=close.columns)
                if weighting == 'invvol':
                    cap = ITER3_SINGLE_CAP_X / len(top)
                    w = _invvol_weights(top, vol_lag.loc[d], cap)
                    weights_now[w.index] = w.values
                else:
                    weights_now[top] = 1.0 / len(top)
            # 得分不足 (早期预热) 时保持原权重
            turnover = float((weights_now - prev_w).abs().sum()) if prev_w is not None else float(weights_now.sum())
            turnover_sum += turnover
            n_rebal += 1
            hold_counts.append(int((weights_now > 0).sum()))
            eq *= (1 - turnover * COST_RATE)
            prev_w = weights_now.copy()

        day_ret = float((weights_now * ret.loc[d]).sum())
        eq *= (1 + day_ret)
        equity.loc[d] = eq

    # 基准对齐同窗口
    b = bench.reindex(dates).ffill().bfill()
    bench_norm = b / b.iloc[0]

    # 超额口径 (指数增强考核)
    excess_daily = equity.pct_change().fillna(0.0) - bench_norm.pct_change().fillna(0.0)
    te = float(excess_daily.std() * np.sqrt(252))
    ir = float(excess_daily.mean() / excess_daily.std() * np.sqrt(252)) if excess_daily.std() > 0 else 0.0
    excess_cum = float((1 + excess_daily).prod() - 1)

    m = Metrics.calc_all(equity, [], risk_free_rate=0.03)
    bm = Metrics.calc_all(bench_norm, [], risk_free_rate=0.03)

    return {
        'equity': equity, 'bench': bench_norm,
        'strategy': m, 'benchmark': bm,
        'excess': {'cum': excess_cum, 'tracking_error': te, 'information_ratio': ir},
        'rebalances': n_rebal,
        'avg_turnover': turnover_sum / max(n_rebal, 1),
        'avg_holdings': float(np.mean(hold_counts)) if hold_counts else 0.0,
    }


def yearly_table(equity: pd.Series, bench: pd.Series) -> pd.DataFrame:
    """分年度: 策略 vs 基准 vs 超额 (跨年基准=上年末)"""
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
    ap = argparse.ArgumentParser(description='截面多因子选股回测 (指数增强风格)')
    ap.add_argument('--limit', type=int, default=None, help='只用前 N 只 (冒烟)')
    ap.add_argument('--top-n', type=int, default=TOP_N,
                    help=f'持仓数 (默认 {TOP_N}; 迭代2/3 预注册为 {ITER2_TOP_N})')
    ap.add_argument('--weighting', choices=['equal', 'invvol'], default='equal',
                    help='权重方案: equal 等权 / invvol 反波动率+封顶 (迭代3 预注册)')
    ap.add_argument('--no-fetch-bench', action='store_true')
    args = ap.parse_args()
    setup_logging()

    top_n = args.top_n
    print('=' * 88)
    print(f'截面多因子选股  池=沪深300当前成分(幸存者偏差已披露)  因子={"+".join(FACTOR_NAMES)} 等权')
    print(f'调仓=每{REBALANCE_DAYS}交易日  持仓=前{top_n} ({args.weighting})  '
          f'成本=换手x{COST_RATE:.2%}  评估={EVAL_START}起')
    print('=' * 88)

    ds = DataSource(use_cache=True)
    codes = hs300_codes()
    if args.limit:
        codes = codes[:args.limit]
        print(f'[冒烟模式] 只取前 {len(codes)} 只')

    print(f'\n加载 {len(codes)} 只行情 ({DATA_START} 起, hfq, 走缓存)...')
    close = load_universe(codes, ds)
    print(f'价格面板: {close.shape[0]} 天 x {close.shape[1]} 只')

    print('\n加载沪深300指数基准...')
    idx = ds.get_index_daily(UNIVERSE_INDEX, start_date=DATA_START)
    bench = idx['close']

    score = factor_scores(close)
    out = backtest(close, score, bench, top_n=top_n, weighting=args.weighting)

    m, bm, ex = out['strategy'], out['benchmark'], out['excess']
    print('\n' + '=' * 88)
    print('结果 (指数增强口径)')
    print('=' * 88)
    print(f"  {'指标':<12}{'策略':>12}{'沪深300':>12}")
    print('-' * 40)
    for key, name, is_pct in [('total_return', '总收益', 1), ('annual_return', '年化', 1),
                              ('annual_volatility', '波动', 1), ('max_drawdown', '回撤', 1),
                              ('sharpe_ratio', '夏普', 0), ('calmar_ratio', '卡玛', 0)]:
        a, b = m[key], bm[key]
        fa = f'{a:.2%}' if is_pct else f'{a:.2f}'
        fb = f'{b:.2%}' if is_pct else f'{b:.2f}'
        print(f'  {name:<12}{fa:>12}{fb:>12}')
    print('-' * 40)
    print(f"  累计超额 {ex['cum']:+.2%}   年化跟踪误差 {ex['tracking_error']:.2%}   信息比率 {ex['information_ratio']:.2f}")
    print(f"  调仓 {out['rebalances']} 次, 平均换手 {out['avg_turnover']:.2f}, 平均持仓 {out['avg_holdings']:.0f} 只")

    yt = yearly_table(out['equity'], out['bench'])
    print('\n分年度:')
    for _, r in yt.iterrows():
        print(f"  {int(r['year'])}: 策略 {r['strategy']:+7.2%}  基准 {r['benchmark']:+7.2%}  超额 {r['excess']:+7.2%}")
    pos_years = int((yt['excess'] > 0).sum())
    print(f"  超额为正年份: {pos_years}/{len(yt)}")

    # ---- 迭代3 采纳线判定 (预注册 P1-P4 原值, 见文件头) ----
    adoption = None
    if top_n == ITER2_TOP_N and args.weighting == ITER3_WEIGHTING:
        recent = yt[yt['year'] >= 2024]['excess'].sum()
        y2025 = yt[yt['year'] == 2025]['excess']
        adoption = {
            'P1_超额为正年份>=5/8': bool(pos_years >= 5),
            'P2_IR>=0.3': bool(ex['information_ratio'] >= 0.3),
            'P3_TE<=8%': bool(ex['tracking_error'] <= 0.08),
            'P4_2024起三年超额合计>-5%': bool(recent > -0.05),
        }
        print('\n迭代3 采纳线 (预注册, 阈值与迭代2相同):')
        for k, v in adoption.items():
            print(f"  {k:<28}: {'PASS' if v else 'FAIL'}")
        n_ok = sum(adoption.values())
        verdict = '[采纳] 进入冻结协议复核' if n_ok == len(adoption) else '[否决] 停止该方向, 结果如实归档'
        print(f'  判定: {verdict} ({n_ok}/{len(adoption)})')
        print(f'  2024起超额合计: {recent:+.2%}   2025单年超额: '
              f'{float(y2025.iloc[0]):+.2%}' if len(y2025) else '')

    # 落盘
    RESULTS_DIR.mkdir(exist_ok=True)
    tag = f'limit{args.limit}' if args.limit else f'hs300_top{top_n}_{args.weighting}'
    pd.DataFrame({'equity': out['equity'], 'benchmark': out['bench']}).to_csv(
        RESULTS_DIR / f'cross_section_{tag}_equity.csv', encoding='utf-8-sig')
    payload = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'universe': f'{UNIVERSE_INDEX} 当前成分 x{close.shape[1]} (幸存者偏差已披露)',
        'preregistered': {'factors': FACTOR_NAMES, 'top_n': top_n,
                          'weighting': args.weighting,
                          'rebalance_days': REBALANCE_DAYS, 'cost_rate': COST_RATE,
                          'eval_start': EVAL_START},
        'strategy_metrics': {k: float(v) for k, v in m.items()},
        'benchmark_metrics': {k: float(v) for k, v in bm.items()},
        'excess': {k: float(v) for k, v in ex.items()},
        'yearly': yt.to_dict('records'),
        'excess_positive_years': f'{pos_years}/{len(yt)}',
        'iter2_adoption': adoption,
    }
    (RESULTS_DIR / f'cross_section_{tag}.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n已保存: results/cross_section_{tag}.json / _equity.csv")


if __name__ == '__main__':
    main()
