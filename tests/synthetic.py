"""
测试用确定性行情数据 (不联网, 完全可复现)

用途: 给回测引擎做 golden master — 引擎行为一旦改变, 这里的数字就会变。
"""
import numpy as np
import pandas as pd


def make_price_frame(symbol: str, n: int = 420, seed: int = 0) -> pd.DataFrame:
    """生成一段有趋势也有震荡的日线, 保证均线能多次交叉"""
    dates = pd.bdate_range('2020-01-01', periods=n)
    rng = np.random.default_rng(seed)
    base = 20.0 + (0 if seed == 0 else 5)
    t = np.arange(n)
    # 波动周期取 24 个交易日左右, 保证 MA20/MA60 在 420 根里多次交叉
    trend = np.sin(t / 12.0) * 2.6 + t * 0.035
    noise = rng.normal(0, 0.12, n).cumsum() * 0.20
    close = base + trend + noise
    close = np.maximum(close, 3.0)

    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))

    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close,
        'volume': np.full(n, 1_000_000.0),
    }, index=pd.DatetimeIndex(dates, name='date'))


def make_data_dict():
    return {
        'SYN_A': make_price_frame('SYN_A', seed=0),
        'SYN_B': make_price_frame('SYN_B', seed=1),
    }
