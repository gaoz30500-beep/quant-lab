"""
指数级趋势概率预测 (次日 + 20 日)

设计原则:
- 不做"点预测", 只输出概率、条件期望收益和建议仓位
- 所有特征/分档阈值/历史胜率都只用当日及之前的数据 (严格因果, 无未来函数)
- 透明的条件概率: 复合趋势得分分 5 档, 用同档历史胜率作为概率, 不用黑箱模型
- 样本量不足时回退到全样本胜率, 避免小样本噪声
- 执行口径: 收盘出信号, 次日开盘成交, 计交易成本
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data_source import DataSource

FEATURE_COLS = ['f_trend', 'f_slope', 'f_mom', 'f_di', 'f_pos', 'f_vol']
MIN_SAMPLES = 60          # 单档最少样本, 不足则回退全样本
ZSCORE_MIN_PERIODS = 250  # 标准化至少需要的历史样本
BUCKET_WINDOW = 500       # 分档阈值用近 2 年滚动分位


def _wilder(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(alpha=1.0 / window, adjust=False).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造因果特征 (只用当日及之前的数据)"""
    c, h, l = df['close'], df['high'], df['low']
    ret1 = c.pct_change()
    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()

    up, dn = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, 14)
    pdi = 100 * _wilder(plus_dm, 14) / atr.replace(0, np.nan)
    mdi = 100 * _wilder(minus_dm, 14) / atr.replace(0, np.nan)

    vol20 = ret1.rolling(20).std() * np.sqrt(252)
    high60 = h.rolling(60).max()

    feats = pd.DataFrame(index=df.index)
    feats['f_trend'] = c / ma60 - 1                    # 站上/跌破长期均线
    feats['f_slope'] = ma20 / ma20.shift(10) - 1      # 中期均线斜率
    feats['f_mom'] = c / c.shift(20) - 1              # 20 日动量
    feats['f_di'] = (pdi - mdi) / 100                 # 方向指标
    feats['f_pos'] = c / high60 - 1                   # 距 60 日高点 (<=0)
    feats['f_vol'] = -vol20                           # 波动越低越好
    feats['vol20'] = vol20
    feats['ma20'] = ma20
    feats['ma60'] = ma60
    return feats


def composite_score(feats: pd.DataFrame) -> pd.Series:
    """6 个特征等权 z-score 平均, 不做权重优化以免过拟合"""
    zs = []
    for col in FEATURE_COLS:
        f = feats[col]
        mean = f.expanding(min_periods=ZSCORE_MIN_PERIODS).mean().shift(1)
        std = f.expanding(min_periods=ZSCORE_MIN_PERIODS).std().shift(1)
        zs.append(((f - mean) / std.replace(0, np.nan)).clip(-3, 3))
    return pd.concat(zs, axis=1).mean(axis=1, skipna=False)


def bucketize(score: pd.Series) -> pd.Series:
    """按近 2 年滚动分位切 5 档 (阈值 shift(1), 只用历史)"""
    qs = {q: score.rolling(BUCKET_WINDOW, min_periods=ZSCORE_MIN_PERIODS)
          .quantile(q).shift(1) for q in (0.2, 0.4, 0.6, 0.8)}
    conds = [
        score <= qs[0.2],
        (score > qs[0.2]) & (score <= qs[0.4]),
        (score > qs[0.4]) & (score <= qs[0.6]),
        (score > qs[0.6]) & (score <= qs[0.8]),
        score > qs[0.8],
    ]
    return pd.Series(np.select(conds, [0, 1, 2, 3, 4], default=np.nan), index=score.index)


def _bucket_stats(bucket: pd.Series, y: pd.Series, horizon: int, min_n: int = MIN_SAMPLES):
    """
    同档历史胜率 + 样本数, 按 horizon 回溯

    关键点: 必须取"当前档位"在 t-horizon 时点的累计统计。
    不能先按每行自身档位算累计值再整体 shift, 否则拿到的是
    t-horizon 那天的档位统计, 而不是今天档位的统计。

    对 1 日标签: 收盘 t 时, t-1 的标签(用到 close t)已知
    对 20 日标签: 收盘 t 时, 最近可用标签是 t-20
    """
    idx = bucket.index
    y = y.astype(float)

    means, cnts, cols = [], [], []
    for b in pd.unique(bucket.dropna()):
        sub = y.where(bucket == b).dropna()
        cols.append(float(b))
        means.append(sub.expanding().mean().reindex(idx, method='ffill'))
        cnts.append(sub.expanding().count().reindex(idx, method='ffill'))

    mean_hist = pd.DataFrame({b: m for b, m in zip(cols, means)}, index=idx)
    cnt_hist = pd.DataFrame({b: c for b, c in zip(cols, cnts)}, index=idx)

    p = pd.Series(np.nan, index=idx)
    n = pd.Series(np.nan, index=idx)
    col_map = {b: j for j, b in enumerate(mean_hist.columns)}
    mv, cv, bv = mean_hist.values, cnt_hist.values, bucket.values

    for i in range(horizon, len(idx)):
        b = bv[i]
        if np.isnan(b):
            continue
        j = col_map.get(float(b))
        if j is None:
            continue
        p.iloc[i] = mv[i - horizon, j]
        n.iloc[i] = cv[i - horizon, j]

    # 样本不足回退到全样本
    all_mean = y.expanding().mean().shift(horizon)
    all_cnt = y.expanding().count().shift(horizon)
    ok = n >= min_n
    p = p.where(ok, all_mean)
    n = n.where(ok, all_cnt)
    return p, n


def forward_return(df: pd.DataFrame, horizon: int):
    fwd = df['close'].shift(-horizon) / df['close'] - 1
    y = pd.Series(np.where(fwd > 0, 1.0, 0.0), index=df.index)
    y[fwd.isna()] = np.nan
    return fwd, y


def perf_stats(equity: pd.Series, rf: float = 0.03) -> dict:
    r = equity.pct_change().dropna()
    n = len(equity)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    years = n / 252.0
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else 0.0
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252 - rf) / vol if vol > 0 else np.nan
    mdd = (equity / equity.cummax() - 1).min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    return {'total': total, 'cagr': cagr, 'vol': vol, 'sharpe': sharpe,
            'mdd': mdd, 'calmar': calmar, 'bars': n,
            'win_days': (r > 0).mean()}


def run_backtest(df: pd.DataFrame, p20: pd.Series, feats: pd.DataFrame,
                 oos_start: str, vol_target: float = 0.12,
                 cost: float = 0.0013, dd_soft: float = 0.10,
                 dd_hard: float = 0.18):
    """
    概率分档仓位 + 目标波动率 + 回撤熔断
    收盘 t 出信号 -> 次日开盘成交 -> 持有当个交易日 (open[t+1] -> open[t+2])
    """
    r_exec = (df['open'].shift(-2) / df['open'].shift(-1) - 1).fillna(0.0)
    equity = pd.Series(1.0, index=df.index)
    pos_hist = pd.Series(0.0, index=df.index)
    eq, prev = 1.0, 0.0
    eq_window = []   # 近 252 日净值, 用于回撤熔断 (不用历史最高, 避免永久锁死)

    for i in range(len(df)):
        p = p20.iloc[i]
        if pd.isna(p) or p <= 0.45:
            base = 0.0
        elif p <= 0.55:
            base = 0.5
        elif p <= 0.65:
            base = 0.8
        else:
            base = 1.0

        vol = feats['vol20'].iloc[i]
        scale = 1.0 if (pd.isna(vol) or vol <= 0) else min(1.0, vol_target / vol)

        eq_window.append(eq)
        if len(eq_window) > 252:
            eq_window.pop(0)
        dd = eq / max(eq_window) - 1
        cap = 1.0
        if dd < -dd_hard:
            cap = 0.0
        elif dd < -dd_soft:
            cap = 0.5

        pos = base * scale * cap
        turnover = abs(pos - prev)
        eq *= 1 + pos * r_exec.iloc[i] - turnover * cost
        equity.iloc[i] = eq
        pos_hist.iloc[i] = pos
        prev = pos

    bench = (1 + r_exec).cumprod()
    return equity, bench, pos_hist, r_exec


def _fmt_pct(x):
    return 'n/a' if pd.isna(x) else f'{x:.2%}'


def _direction_label(p: float) -> str:
    if pd.isna(p):
        return '数据不足'
    if p >= 0.55:
        return '偏多(强)'
    if p >= 0.52:
        return '偏多(弱)'
    if p >= 0.48:
        return '中性'
    if p >= 0.45:
        return '偏空(弱)'
    return '偏空(强)'


def run_forecast(symbol: str = '000001', start: str = '20100101',
                 oos_start: str = '20180101', vol_target: float = 0.12):
    print('=' * 64)
    print(f'指数趋势概率  标的={symbol}  训练起点={start}  样本外起点={oos_start}')
    print('=' * 64)

    ds = DataSource(use_cache=True)
    df = ds.get_index_daily(symbol, start_date=start)
    print(f'数据: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)} 根K线')

    feats = build_features(df)
    score = composite_score(feats)
    bucket = bucketize(score)
    fwd1, y1 = forward_return(df, 1)
    fwd20, y20 = forward_return(df, 20)
    p1, n1 = _bucket_stats(bucket, y1, 1)
    p20, n20 = _bucket_stats(bucket, y20, 20)
    exp20, _ = _bucket_stats(bucket, fwd20, 20)

    i = len(df) - 1
    print('\n【下一交易日】')
    print(f'  P(上涨) = {_fmt_pct(p1.iloc[i])}   方向 = {_direction_label(p1.iloc[i])}   '
          f'同档历史上涨率样本 n={int(n1.iloc[i]) if not pd.isna(n1.iloc[i]) else 0}')
    print(f'  当前复合趋势得分 = {score.iloc[i]:+.2f}  分档 = {int(bucket.iloc[i]) if not pd.isna(bucket.iloc[i]) else -1}/4')

    print('\n【未来 20 个交易日】')
    print(f'  P(上涨) = {_fmt_pct(p20.iloc[i])}   条件期望收益 = {_fmt_pct(exp20.iloc[i])}   '
          f'同档样本 n={int(n20.iloc[i]) if not pd.isna(n20.iloc[i]) else 0}')

    # ---- 次日方向准确率 (样本外) ----
    oos = df.index >= pd.Timestamp(oos_start)
    mask1 = oos & p1.notna() & y1.notna()
    hit1 = (p1 > 0.5) == (y1 == 1)
    acc = hit1[mask1].mean()
    base_up = (y1[mask1] == 1).mean()
    brier = ((p1[mask1] - y1[mask1]) ** 2).mean()
    print(f'\n【次日方向准确率 (样本外 {oos_start} 起, n={int(mask1.sum())})】')
    print(f'  模型命中率 = {acc:.1%}   基准(一直看涨) = {base_up:.1%}   Brier = {brier:.4f}')

    tmp = pd.DataFrame({'p': p1[mask1], 'y': y1[mask1]})
    tmp['hit'] = (tmp['p'] > 0.5) == (tmp['y'] == 1)
    year_tbl = tmp.groupby(tmp.index.year).agg(n=('hit', 'size'), hit=('hit', 'mean'))
    print('  按年份:', ', '.join(f'{y}:{r.hit:.0%}({int(r.n)})' for y, r in year_tbl.iterrows()))

    bins = [0.0, 0.45, 0.48, 0.5, 0.52, 0.55, 0.6, 1.01]
    labels = ['<45%', '45-48%', '48-50%', '50-52%', '52-55%', '55-60%', '>60%']
    tmp['bin'] = pd.cut(tmp['p'], bins=bins, labels=labels, right=False)
    print('  按预测概率分档:')
    for lab, grp in tmp.groupby('bin', observed=True):
        print(f'    预测{lab}: n={len(grp):4d}, 实际上涨={grp["y"].mean():.1%}, 方向命中={grp["hit"].mean():.1%}')

    cur_bin = pd.cut(pd.Series([p1.iloc[i]]), bins=bins, labels=labels, right=False).iloc[0]
    cur_grp = tmp[tmp['bin'] == cur_bin]
    if len(cur_grp):
        print(f'  当前 {p1.iloc[i]:.2%} 落在「{cur_bin}」档: 该档历史实际上涨 {cur_grp["y"].mean():.1%} '
              f'(n={len(cur_grp)})')

    # ---- 20 日概率的准确性 ----
    mask20 = df.index >= pd.Timestamp(oos_start) + pd.Timedelta(days=1)
    mask20 = mask20 & p20.notna() & y20.notna()
    hit20 = (p20 > 0.5) == (y20 == 1)
    print(f'\n【20日方向命中率 (样本外, n={int(mask20.sum())})】')
    print(f'  模型命中率 = {hit20[mask20].mean():.1%}   基准 = {(y20[mask20] == 1).mean():.1%}')

    # ---- 仓位策略回测 ----
    equity, bench, pos, _ = run_backtest(df, p20, feats, oos_start, vol_target)
    eq_oos = equity[df.index >= oos_start]
    bm_oos = bench[df.index >= oos_start]
    eq_oos = eq_oos / eq_oos.iloc[0]
    bm_oos = bm_oos / bm_oos.iloc[0]
    s_pos = perf_stats(eq_oos)
    s_bm = perf_stats(bm_oos)
    turnover = pos.diff().abs().loc[df.index >= oos_start].sum()
    years = len(eq_oos) / 252.0
    print(f'\n【概率分档仓位策略 (样本外, 波动目标 {vol_target:.0%})】')
    print(f'  {"指标":<10}{"策略":>12}{"买入持有":>12}')
    for key, name, pct in [('total', '总收益', 1), ('cagr', '年化', 1), ('vol', '年化波动', 1),
                           ('sharpe', '夏普', 0), ('mdd', '最大回撤', 1), ('calmar', '卡玛', 0)]:
        a, b = s_pos[key], s_bm[key]
        fa = _fmt_pct(a) if pct else f'{a:.2f}'
        fb = _fmt_pct(b) if pct else f'{b:.2f}'
        print(f'  {name:<10}{fa:>12}{fb:>12}')
    print(f'  年均换手 = {turnover / years:.2f} 倍')

    # ---- 落盘 ----
    out_dir = Path(__file__).parent / 'results'
    out_dir.mkdir(exist_ok=True)
    hist = pd.DataFrame({
        'close': df['close'], 'score': score, 'bucket': bucket,
        'p_next_up': p1, 'p_20d_up': p20, 'exp_20d_ret': exp20,
        'next_ret': fwd1, 'y_next_up': y1, 'y_20d_up': y20,
    })
    hist['hit_next'] = np.nan
    valid = y1.notna() & p1.notna()
    hist.loc[valid, 'hit_next'] = ((p1[valid] > 0.5) == (y1[valid] == 1)).astype(float)
    hist.index.name = 'date'
    hist.to_csv(out_dir / f'forecast_history_{symbol}.csv', encoding='utf-8-sig')
    eq_df = pd.DataFrame({'equity': equity, 'benchmark': bench, 'position': pos})
    eq_df.index.name = 'date'
    eq_df.to_csv(out_dir / f'forecast_equity_{symbol}.csv', encoding='utf-8-sig')
    print(f'\n已保存: results/forecast_history_{symbol}.csv, results/forecast_equity_{symbol}.csv')
    return {'p1': p1.iloc[i], 'p20': p20.iloc[i], 'exp20': exp20.iloc[i],
            'strategy': s_pos, 'benchmark': s_bm, 'acc': acc, 'base_up': base_up}


def run_fund_forecast(code: str):
    """
    场外基金净值概率预测 (指数版方法论的移植)

    基金只有净值(无高低开), 用 open=high=low=close=nav 构造面板,
    特征/分档/胜率统计逻辑与指数版完全一致 (严格因果, shift(1))。

    诚实披露:
    - 单只基金历史短 (多数 2-4 年), 分档样本少, 大概率回退到全样本胜率,
      概率的置信度远低于指数版 (指数有 15 年数据)
    - 输出是"倾向"不是"预测", 不能覆盖闸门规则的趋势判断
    """
    from fund_rules_backtest import get_nav

    print('=' * 64)
    print(f'基金净值概率预测  代码={code}')
    print('=' * 64)

    nav = get_nav(code)
    df = pd.DataFrame({'open': nav, 'high': nav, 'low': nav, 'close': nav})
    print(f'数据: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)} 条净值')
    if len(df) < 300:
        print('警告: 历史不足 300 条, 特征预热不够, 结果仅供参考')

    feats = build_features(df)
    score = composite_score(feats)
    bucket = bucketize(score)
    fwd1, y1 = forward_return(df, 1)
    fwd20, y20 = forward_return(df, 20)
    p1, n1 = _bucket_stats(bucket, y1, 1)
    p20, n20 = _bucket_stats(bucket, y20, 20)
    exp20, _ = _bucket_stats(bucket, fwd20, 20)

    i = len(df) - 1
    print('\n【下一交易日】')
    print(f'  P(上涨) = {_fmt_pct(p1.iloc[i])}   方向 = {_direction_label(p1.iloc[i])}   '
          f'同档历史样本 n={int(n1.iloc[i]) if not pd.isna(n1.iloc[i]) else 0}')
    print(f'  当前复合趋势得分 = {score.iloc[i]:+.2f}  分档 = '
          f'{int(bucket.iloc[i]) if not pd.isna(bucket.iloc[i]) else -1}/4')

    print('\n【未来 20 个交易日】')
    print(f'  P(上涨) = {_fmt_pct(p20.iloc[i])}   条件期望收益 = {_fmt_pct(exp20.iloc[i])}   '
          f'同档样本 n={int(n20.iloc[i]) if not pd.isna(n20.iloc[i]) else 0}')

    # 历史命中率 (全可用区间, 样本小要打折看)
    mask1 = p1.notna() & y1.notna()
    if mask1.sum() > 30:
        hit1 = ((p1 > 0.5) == (y1 == 1))
        acc = hit1[mask1].mean()
        base_up = (y1[mask1] == 1).mean()
        print(f'\n【历史命中率 (全部可用数据, n={int(mask1.sum())}, 样本小置信度低)】')
        print(f'  模型命中率 = {acc:.1%}   基准(一直看涨) = {base_up:.1%}')
    else:
        print('\n【历史命中率】有效样本不足, 无法评估')

    # 趋势状态对照 (闸门视角)
    ma20 = nav.rolling(20).mean().iloc[-1]
    ma60 = nav.rolling(60).mean().iloc[-1] if len(nav) >= 60 else float('nan')
    last = float(nav.iloc[-1])
    trend = '趋势完好(MA60上方)' if last > ma60 else '趋势破坏(MA60下方, 闸门规则=离场)'
    print(f'\n【趋势状态】净值 {last:.4f}, 距MA20 {last/ma20-1:+.2%}, '
          f'距MA60 {last/ma60-1:+.2%} -> {trend}')
    print('\n提醒: 短期概率与趋势判断冲突时, 以闸门规则为准 (概率管仓位轻重, 趋势管去留)。')
    return {'p1': p1.iloc[i], 'p20': p20.iloc[i], 'exp20': exp20.iloc[i]}


def main():
    import argparse
    ap = argparse.ArgumentParser(description='趋势概率预测 (指数/基金)')
    ap.add_argument('--fund', help='场外基金代码 (如 018036), 用净值做概率预测')
    ap.add_argument('--symbol', default='000001', help='指数代码, 默认上证指数')
    ap.add_argument('--start', default='20100101')
    ap.add_argument('--oos', default='20180101', help='样本外起点')
    ap.add_argument('--vol-target', type=float, default=0.12, help='目标年化波动')
    args = ap.parse_args()
    if args.fund:
        run_fund_forecast(args.fund)
    else:
        run_forecast(symbol=args.symbol, start=args.start,
                     oos_start=args.oos, vol_target=args.vol_target)


if __name__ == '__main__':
    main()
