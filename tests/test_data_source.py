"""
数据源层边界测试 (data_source.py)

覆盖命门路径:
- validate_daily: 只修确定有害项, 其余仅报告 (OHLC矛盾/跳变/缺失日 不擅改)
- _request_with_retry: 重试/指数退避/全失败返回 None/未知接口报错
- 缓存新鲜度: _prev_weekday / _expected_bar_date / _is_cache_fresh / _drop_incomplete_bar

跑法: python -m unittest discover -s tests -v
时间相关逻辑用 mock 或"相对今天"构造, 保证确定性。
"""
import sys
import json
import unittest
from pathlib import Path
from unittest import mock
from datetime import datetime, timedelta, date

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import data_source
from data_source import DataSource, validate_daily


def _frame(closes, start='2024-01-01', open_=None, high=None, low=None):
    idx = pd.bdate_range(start, periods=len(closes), name='date')
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        'open': c if open_ is None else np.asarray(open_, dtype=float),
        'high': c if high is None else np.asarray(high, dtype=float),
        'low': c if low is None else np.asarray(low, dtype=float),
        'close': c,
        'volume': np.ones(len(c)),
    }, index=idx)


class TestValidateDailyBoundary(unittest.TestCase):
    """validate_daily: 报告但不擅改的边界情形"""

    def test_empty_df_no_crash(self):
        empty = _frame([]).iloc[:0]
        cleaned, issues = validate_daily(empty)
        self.assertEqual(len(cleaned), 0)

    def test_ohlc_inconsistent_reported_not_removed(self):
        # close > high: 只报告, 行数不减 (擅自删行会改变回测样本)
        df = _frame([10.0, 11.0, 12.0], high=[10.0, 10.5, 12.0])
        cleaned, issues = validate_daily(df)
        self.assertEqual(len(cleaned), 3)
        self.assertIn('ohlc_inconsistent', issues)
        self.assertEqual(issues['ohlc_inconsistent'], 1)

    def test_abnormal_jump_reported_with_dates(self):
        # +40% 单日跳变: 报告并给出前3个日期
        df = _frame([10.0, 10.2, 14.3, 14.4])
        _, issues = validate_daily(df)
        self.assertIn('abnormal_jumps', issues)
        self.assertEqual(issues['abnormal_jumps'], 1)
        self.assertEqual(len(issues['jump_dates']), 1)

    def test_normal_jump_not_flagged(self):
        df = _frame([10.0, 10.5, 11.0, 11.4])  # 每日 <5%
        _, issues = validate_daily(df)
        self.assertNotIn('abnormal_jumps', issues)

    def test_missing_days_vs_calendar(self):
        # 缺失检测只覆盖 df 首尾之间的内部缺口 (日历会被裁剪到数据范围)
        df = _frame([10.0, 11.0, 12.0], start='2024-01-01')
        df.index = pd.DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-05'])
        calendar = pd.bdate_range('2024-01-01', '2024-01-05')  # 内部缺 1/3,1/4
        _, issues = validate_daily(df, calendar=calendar)
        self.assertEqual(issues['missing_days'], 2)
        self.assertEqual(len(issues['missing_examples']), 2)

    def test_clean_data_no_issues(self):
        df = _frame([10.0, 10.1, 10.2])
        cleaned, issues = validate_daily(df)
        self.assertEqual(issues, {})
        self.assertEqual(len(cleaned), 3)


class TestRequestWithRetry(unittest.TestCase):
    """_request_with_retry: 重试与降级路径"""

    def setUp(self):
        self.ds = DataSource(use_cache=False)
        self.df = _frame([10.0, 11.0])
        # 睡眠全部短路, 测试不等待
        self.sleep_patcher = mock.patch('data_source.time.sleep')
        self.mock_sleep = self.sleep_patcher.start()

    def tearDown(self):
        self.sleep_patcher.stop()

    def _install_fake(self, behavior):
        """往 akshare 模块上挂临时接口"""
        import akshare as ak
        ak._test_fake = behavior

        def _cleanup():
            if hasattr(ak, '_test_fake'):
                delattr(ak, '_test_fake')
        self.addCleanup(_cleanup)

    def test_success_first_try(self):
        calls = []

        def fake(**kw):
            calls.append(kw)
            return self.df

        self._install_fake(fake)
        out = self.ds._request_with_retry('_test_fake', symbol='000001')
        self.assertEqual(len(out), 2)
        self.assertEqual(len(calls), 1)
        self.mock_sleep.assert_not_called()

    def test_fail_then_succeed(self):
        state = {'n': 0}

        def fake(**kw):
            state['n'] += 1
            if state['n'] < 3:
                raise ConnectionError('boom')
            return self.df

        self._install_fake(fake)
        out = self.ds._request_with_retry('_test_fake')
        self.assertEqual(len(out), 2)
        self.assertEqual(state['n'], 3)
        # 指数退避: 第1次失败睡2s, 第2次睡4s
        self.assertEqual([c.args[0] for c in self.mock_sleep.call_args_list], [2, 4])

    def test_all_fail_returns_none(self):
        def fake(**kw):
            raise ConnectionError('boom')

        self._install_fake(fake)
        out = self.ds._request_with_retry('_test_fake')
        self.assertIsNone(out)
        self.assertEqual(self.mock_sleep.call_count, 2)  # 最后一次失败不再睡

    def test_empty_df_counts_as_failure(self):
        def fake(**kw):
            return pd.DataFrame()

        self._install_fake(fake)
        self.assertIsNone(self.ds._request_with_retry('_test_fake'))

    def test_unknown_api_raises(self):
        with self.assertRaises(AttributeError):
            self.ds._request_with_retry('no_such_api_xyz')


class TestCalendarHelpers(unittest.TestCase):
    """交易日回退与"应有K线日期" """

    def test_prev_weekday(self):
        self.assertEqual(DataSource._prev_weekday(date(2024, 1, 6)), date(2024, 1, 5))   # 周六→周五
        self.assertEqual(DataSource._prev_weekday(date(2024, 1, 7)), date(2024, 1, 5))   # 周日→周五
        self.assertEqual(DataSource._prev_weekday(date(2024, 1, 5)), date(2024, 1, 5))   # 周五不动

    def test_expected_bar_date_historical_end(self):
        # 指定历史 end_date 时以请求区间为准, 不受"今天"影响
        got = DataSource._expected_bar_date('20240107')  # 周日
        self.assertEqual(got, date(2024, 1, 5))

    def test_expected_bar_date_never_future(self):
        got = DataSource._expected_bar_date(None)
        self.assertLessEqual(got, date.today())


class TestDropIncompleteBar(unittest.TestCase):
    """盘中"半根日线"必须丢弃 (防未来函数的执行层保障)"""

    def _run(self, fake_now, last_bar_date):
        df = _frame([10.0, 11.0], start='2024-01-04')
        # 重建索引让最后一根 = 指定日期
        df.index = pd.DatetimeIndex([last_bar_date - timedelta(days=1), last_bar_date])
        fake_dt = mock.Mock(wraps=datetime)
        fake_dt.now.return_value = fake_now
        with mock.patch.object(data_source, 'datetime', fake_dt):
            return self.ds._drop_incomplete_bar(df)

    def setUp(self):
        self.ds = DataSource(use_cache=False)

    def test_intraday_bar_dropped(self):
        # 2024-01-05 是周五; 上午10点拉数据, 当日K线未收盘 → 丢弃
        out = self._run(datetime(2024, 1, 5, 10, 0), pd.Timestamp('2024-01-05'))
        self.assertEqual(out.index[-1], pd.Timestamp('2024-01-04'))

    def test_after_close_bar_kept(self):
        # 收盘后 (16点) 当日K线完整 → 保留
        out = self._run(datetime(2024, 1, 5, 16, 0), pd.Timestamp('2024-01-05'))
        self.assertEqual(out.index[-1], pd.Timestamp('2024-01-05'))

    def test_historical_bar_kept(self):
        # 盘中拉数据但最后一根是历史日 → 保留
        out = self._run(datetime(2024, 1, 8, 10, 0), pd.Timestamp('2024-01-05'))
        self.assertEqual(out.index[-1], pd.Timestamp('2024-01-05'))


class TestCacheFreshness(unittest.TestCase):
    """_is_cache_fresh: 新鲜/过期/盘中重拉 判定"""

    def setUp(self):
        self.ds = DataSource(use_cache=False)
        self.tmp = Path(__file__).parent / '_tmp_cache_test'
        self.tmp.mkdir(exist_ok=True)
        self.csv = self.tmp / 'x.csv'
        self.csv.write_text('date,close\n2021-01-04,10.0\n', encoding='utf-8')

    def tearDown(self):
        for p in self.tmp.iterdir():
            p.unlink()
        self.tmp.rmdir()

    def _write_stamp(self, checked_at, last_bar):
        p = self.ds._stamp_path(self.csv)
        p.write_text(json.dumps({
            'checked_at': checked_at, 'last_bar_date': str(last_bar),
        }), encoding='utf-8')

    def test_no_stamp_stale(self):
        bar = pd.Timestamp('2021-01-04')
        self.assertFalse(self.ds._is_cache_fresh(self.csv, bar, date(2024, 1, 5)))

    def test_future_bar_fresh(self):
        # K线已覆盖应有日期且非今天 → 新鲜
        bar = pd.Timestamp('2021-01-04')
        self.assertTrue(self.ds._is_cache_fresh(self.csv, bar, date(2020, 1, 1)))

    def test_stale_but_checked_today_same_bar(self):
        # 今天已成功拉过且K线没动 → 数据源未更新(节假日), 当天不再重复请求
        bar = pd.Timestamp('2021-01-04')
        self._write_stamp(datetime.now().isoformat(timespec='seconds'), bar.date())
        fresh = self.ds._is_cache_fresh(self.csv, bar, date(2024, 1, 5))
        # checked==now → "盘中检查过且现已收盘"分支不可能成立 → 必为 True
        self.assertTrue(fresh)

    def test_checked_yesterday_stale(self):
        bar = pd.Timestamp('2021-01-04')
        yesterday = (datetime.now() - timedelta(days=1)).isoformat(timespec='seconds')
        self._write_stamp(yesterday, bar.date())
        self.assertFalse(self.ds._is_cache_fresh(self.csv, bar, date(2024, 1, 5)))

    def test_stamp_bar_mismatch_stale(self):
        # stamp 记录的K线日期与实际不符 (缓存被手工改过) → 不新鲜
        bar = pd.Timestamp('2021-01-04')
        self._write_stamp(datetime.now().isoformat(timespec='seconds'), date(2020, 12, 31))
        self.assertFalse(self.ds._is_cache_fresh(self.csv, bar, date(2024, 1, 5)))

    def test_corrupt_stamp_stale(self):
        bar = pd.Timestamp('2021-01-04')
        self.ds._stamp_path(self.csv).write_text('{broken json', encoding='utf-8')
        self.assertFalse(self.ds._is_cache_fresh(self.csv, bar, date(2024, 1, 5)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
