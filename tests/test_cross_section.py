"""
_invvol_weights 回归测试

守护的 bug (2026-11-06 实际发生): 先 clip 后归一 —— 1/vol 原始量纲远大于 cap,
clip 把所有票削成同值, 归一后恰好等于等权, 反波动率加权静默退化,
迭代3 结果与迭代2 逐位相同才暴露。
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from cross_section import _invvol_weights


class TestInvvolWeights(unittest.TestCase):

    def _vol_row(self, vols: dict) -> pd.Series:
        return pd.Series(vols)

    def test_not_degenerate_to_equal(self):
        # 波动率差 4 倍 -> 权重必须明显不等 (bug 版本会输出全等权)
        codes = [f'S{i}' for i in range(10)]
        vols = {c: 0.01 if i < 5 else 0.04 for i, c in enumerate(codes)}
        w = _invvol_weights(codes, self._vol_row(vols), cap=2.0 / len(codes))
        low_vol_mean = w[codes[:5]].mean()
        high_vol_mean = w[codes[5:]].mean()
        self.assertGreater(low_vol_mean, high_vol_mean * 1.5,
                           '低波票权重应显著更高; 全等权=clip/归一顺序 bug 复发')

    def test_sums_to_one_and_respects_cap(self):
        codes = [f'S{i}' for i in range(20)]
        rng = np.random.default_rng(7)
        vols = {c: float(rng.uniform(0.005, 0.06)) for c in codes}
        cap = 2.0 / len(codes)
        w = _invvol_weights(codes, self._vol_row(vols), cap=cap)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=9)
        self.assertLessEqual(float(w.max()), cap + 1e-9)

    def test_missing_vol_fallback_median(self):
        codes = ['A', 'B', 'C', 'D']
        vols = pd.Series({'A': 0.02, 'B': 0.04, 'C': np.nan})   # D 完全缺失
        w = _invvol_weights(codes, vols, cap=0.5)
        self.assertEqual(set(w.index), set(codes))
        self.assertAlmostEqual(float(w.sum()), 1.0, places=9)
        self.assertGreater(float(w['D']), 0.0)

    def test_all_nan_falls_back_to_equal(self):
        codes = ['A', 'B', 'C']
        vols = pd.Series({c: np.nan for c in codes})
        w = _invvol_weights(codes, vols, cap=1.0)
        self.assertAlmostEqual(float(w.max()), 1 / 3, places=9)

    def test_zero_vol_not_inf(self):
        codes = ['A', 'B']
        vols = pd.Series({'A': 0.0, 'B': 0.02})
        w = _invvol_weights(codes, vols, cap=1.0)
        self.assertTrue(np.isfinite(w.values).all())
        self.assertAlmostEqual(float(w.sum()), 1.0, places=9)


if __name__ == '__main__':
    unittest.main(verbosity=2)
