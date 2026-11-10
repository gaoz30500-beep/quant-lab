"""
category_composite 回归测试

守护的 bug (2026-11-10 实际发生): 各基金净值公布时点不一致, 面板末行若某成员
尚未公布, 旧实现用"当天已公布的成员"求均值, 复合净值因成员构成变化而跳变,
导致 A股权益 类别被误判为"破MA60离场" (实际三只基金当日并无大跌)。
修复: 先 ffill (未公布=按最后已知净值) 再平均。
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from fund_gate_overlay import category_composite, CATEGORIES


class TestCategoryComposite(unittest.TestCase):

    def _panel(self):
        idx = pd.bdate_range('2024-01-01', periods=5)
        # 成员A: 5天全有; 成员B: 最后一天未公布 (NaN)
        return pd.DataFrame({
            'A': [1.0, 1.01, 1.02, 1.03, 1.04],
            'B': [2.0, 2.02, 2.04, 2.06, np.nan],
        }, index=idx)

    def test_no_jump_when_member_not_yet_published(self):
        comp = category_composite(self._panel(), ['A', 'B'])
        # 第4天: A=1.03/1.0=1.03, B=2.06/2.0=1.03 -> 均值 1.03
        # 第5天: B未公布, ffill 用 2.06 -> A=1.04, B=1.03 -> 均值 1.035
        # 旧实现(无ffill): 第5天只有A -> 1.04, 且第4/5天成员构成不一致会跳变
        self.assertAlmostEqual(comp.iloc[3], 1.03, places=6)
        self.assertAlmostEqual(comp.iloc[4], 1.035, places=6)
        # 关键: 相邻两天变化平滑 (单日变化 < 1%), 不允许因成员缺失误跳
        self.assertLess(abs(comp.iloc[4] / comp.iloc[3] - 1), 0.01)

    def test_pre_inception_stays_nan(self):
        # 成员成立前的日期不应被回填 (ffill 只向后填)
        idx = pd.bdate_range('2024-01-01', periods=4)
        panel = pd.DataFrame({
            'A': [1.0, 1.01, 1.02, 1.03],
            'B': [np.nan, np.nan, 2.0, 2.02],   # B 第3天才成立
        }, index=idx)
        comp = category_composite(panel, ['A', 'B'])
        self.assertTrue(np.isnan(comp.iloc[0]))
        self.assertTrue(np.isnan(comp.iloc[1]))
        self.assertFalse(np.isnan(comp.iloc[2]))

    def test_real_categories_shape(self):
        # 真实类别定义完整性 (防止 CATEGORIES 被误改)
        self.assertEqual(len(CATEGORIES), 5)
        all_codes = [c for codes in CATEGORIES.values() for c in codes]
        self.assertEqual(len(all_codes), 10)
        self.assertEqual(len(set(all_codes)), 10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
