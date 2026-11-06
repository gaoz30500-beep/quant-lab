"""
核心回路测试 (标注规则的 golden master)

跑法: python -m unittest discover -s tests -v
说明: 合成数据完全确定性, 因此这些数字可以当回归基线 —— 引擎行为一旦漂移就会红。
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import numpy as np
import pandas as pd

from config import BacktestConfig
from engine import is_limit_locked
from data_source import validate_daily
from strategies.ma_cross import MACrossStrategy
from portfolio_engine import PortfolioEngine
from fund_rules_backtest import rule_ma60_exit_ma20_reentry
from synthetic import make_data_dict


def build_result(eval_start=None):
    data = make_data_dict()
    config = BacktestConfig(
        initial_capital=100000, max_positions=2, stop_loss_pct=0.08,
        take_profit_pct=0.0, trailing_stop_pct=0.0,
    )

    def factory(symbol):
        return MACrossStrategy(fast_window=20, slow_window=60)

    return PortfolioEngine(data, factory, config, trading_days=None,
                           eval_start=eval_start).run(), data


class TestLimitLock(unittest.TestCase):
    """涨跌停一字板: 买不进 / 卖不出"""

    def test_main_board_limit_up(self):
        # 主板 10%: 前收 10 元, 开盘 11 元 = 一字涨停
        self.assertTrue(is_limit_locked('600000', 'BUY', 11.0, 10.0))
        self.assertFalse(is_limit_locked('600000', 'BUY', 10.9, 10.0))

    def test_main_board_limit_down(self):
        self.assertTrue(is_limit_locked('600000', 'SELL', 9.0, 10.0))
        self.assertFalse(is_limit_locked('600000', 'SELL', 9.1, 10.0))

    def test_chinext_20pct(self):
        # 创业板 20%: 前收 10 元, 涨停价 12 元
        self.assertTrue(is_limit_locked('300750', 'BUY', 12.0, 10.0))
        self.assertFalse(is_limit_locked('300750', 'BUY', 11.9, 10.0))

    def test_etf_no_price_limit(self):
        # ETF/基金无涨跌停限制
        self.assertFalse(is_limit_locked('510300', 'BUY', 12.0, 10.0))


class TestValidateDaily(unittest.TestCase):
    """脏数据修正: 只修确定有害的项"""

    def _frame(self, closes):
        idx = pd.bdate_range('2020-01-01', periods=len(closes), name='date')
        return pd.DataFrame({
            'open': closes, 'high': closes, 'low': closes,
            'close': closes, 'volume': 1.0,
        }, index=idx)

    def test_drop_duplicate_rows(self):
        df = self._frame([10.0, 11.0, 12.0])
        df_dup = pd.concat([df, df.iloc[[1]]])
        cleaned, issues = validate_daily(df_dup)
        self.assertEqual(len(cleaned), 3)
        self.assertIn('duplicate_dates', issues)

    def test_drop_non_positive_price(self):
        df = self._frame([10.0, 0.0, 12.0])
        cleaned, issues = validate_daily(df)
        self.assertEqual(len(cleaned), 2)
        self.assertIn('non_positive', issues)


class TestMa60ExitRule(unittest.TestCase):
    """破MA60走/回MA20进: 离场优先级必须高于回场"""

    def _bounce_after_downtrend(self):
        # 60 天缓慢下跌 + 末 6 天反弹: 价格会落在 MA20 之上、MA60 之下
        down = 100.0 - 0.35 * np.arange(60)
        up = down[-1] + np.array([0.6, 1.2, 1.8, 2.4, 3.0, 3.6])
        nav = pd.Series(np.concatenate([down, up]),
                        index=pd.bdate_range('2020-01-01', periods=66))
        return nav

    def test_exit_wins_over_reentry(self):
        nav = self._bounce_after_downtrend()
        ma20 = nav.rolling(20).mean()
        ma60 = nav.rolling(60).mean()
        price = nav.iloc[-1]
        # 前提: 该时点"价格低于 MA60 但高于 MA20" —— 正是两种写法会分叉的区间
        self.assertLess(price, ma60.iloc[-1])
        self.assertGreater(price, ma20.iloc[-1])

        pos = rule_ma60_exit_ma20_reentry(nav)
        self.assertEqual(float(pos.iloc[-1]), 0.0,
                         '跌破 MA60 必须离场, 不能被"站上 MA20"覆盖')


class TestPortfolioGoldenMaster(unittest.TestCase):
    """合成数据上的完整组合回测结果 (回归基线)"""

    @classmethod
    def setUpClass(cls):
        cls.result, cls.data = build_result()

    def test_headline_metrics(self):
        m = self.result.metrics
        self.assertEqual(m['total_trades'], 10)
        self.assertAlmostEqual(m['total_return'], -0.187087, places=5)
        self.assertAlmostEqual(m['max_drawdown'], 0.257196, places=5)
        self.assertAlmostEqual(m['annual_return'], -0.116867, places=5)
        self.assertAlmostEqual(float(self.result.equity_curve.iloc[-1]), 81297.72, places=2)

    def test_trade_pnl_accounting(self):
        """逐笔 pnl 必须等于 卖出净额 - (买入成本 + 买入佣金)"""
        for t in self.result.trades:
            expected = (t.shares * t.exit_price - t.commission - t.stamp_tax
                        - t.shares * t.entry_price - t.buy_commission)
            self.assertAlmostEqual(t.pnl, expected, places=6,
                                   msg=f'{t.symbol} {t.entry_date.date()} 的 pnl 与成本口径不符')
            self.assertGreater(t.buy_commission, 0.0, '买入佣金应计入每笔交易')

    def test_trade_sum_vs_equity(self):
        """明细合计与净值增长之间只剩现金利息, 不应再有"消失的钱" """
        pnl_sum = sum(t.pnl for t in self.result.trades)
        equity_gain = self.result.equity_curve.iloc[-1] - 100000.0
        # 现金利息使净值高于交易盈亏之和; 差额应远小于总盈亏规模
        self.assertGreater(equity_gain, pnl_sum)
        self.assertLess(abs(equity_gain - pnl_sum), abs(pnl_sum) * 0.25)


class TestWarmupWindow(unittest.TestCase):
    """评估窗口与预热窗口分离 (回归基线)"""

    def test_eval_start_keeps_warmup(self):
        data = make_data_dict()
        eval_start = data['SYN_A'].index[150]
        result, _ = build_result(eval_start=eval_start)

        self.assertEqual(result.equity_curve.index[0], eval_start)
        # 预热在窗口外完成后, 这一段的结果应当等于下面这组数字;
        # 若退回"先切数据再算信号", MA60 会重新累积, 结果必然偏离。
        self.assertEqual(result.metrics['total_trades'], 8)
        self.assertAlmostEqual(result.metrics['total_return'], -0.152550, places=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
