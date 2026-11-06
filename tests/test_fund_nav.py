"""
fund_rules_backtest.get_nav 三级降级策略测试

失败策略 (商业化口径):
1. 缓存新鲜            → 直接用, 不碰网络
2. 缓存损坏/过期 + 网络OK → 重拉并覆写缓存
3. 网络失败 + 有旧缓存   → 降级用旧缓存 (WARNING)
4. 网络失败 + 无缓存     → 抛 RuntimeError (带基金代码)
5. 接口返回空序列       → 视为失败, 走 3/4

网络与缓存目录全部 mock/临时目录, 测试不联网、不写仓库 data/cache。
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

import fund_rules_backtest as frb


def _make_nav(start='2024-01-01', n=5, base=1.0):
    idx = pd.bdate_range(start, periods=n)
    return pd.Series([base + i * 0.01 for i in range(n)], index=idx, name='nav')


def _ak_df(nav: pd.Series) -> pd.DataFrame:
    """模拟 ak.fund_open_fund_info_em 的返回格式 (中文列)"""
    return pd.DataFrame({
        '净值日期': [d.strftime('%Y-%m-%d') for d in nav.index],
        '单位净值': [str(v) for v in nav.values],
    })


class GetNavTestBase(unittest.TestCase):
    def setUp(self):
        # 临时缓存目录 (替换模块级 CACHE_DIR)
        self.tmp = Path(__file__).parent / '_tmp_fund_cache'
        self.tmp.mkdir(exist_ok=True)
        self._cache_dir = frb.CACHE_DIR
        frb.CACHE_DIR = self.tmp
        # 短路限速 sleep
        self.sleep_patcher = mock.patch.object(frb.time, 'sleep')
        self.sleep_patcher.start()
        self.addCleanup(self.sleep_patcher.stop)

    def tearDown(self):
        frb.CACHE_DIR = self._cache_dir
        for p in self.tmp.iterdir():
            p.unlink()
        self.tmp.rmdir()

    def _write_cache(self, code: str, nav: pd.Series):
        path = self.tmp / f'fund_{code}.csv'
        df = nav.rename_axis('date').reset_index()
        df.to_csv(path, index=False, encoding='utf-8-sig')
        return path

    def _patch_ak(self, fn):
        fake = mock.Mock()
        fake.fund_open_fund_info_em = fn
        patcher = mock.patch.object(frb, 'ak', fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        return fake


class TestCacheFresh(GetNavTestBase):
    def test_fresh_cache_skips_network(self):
        # 缓存K线到 2099 年: 必然 >= _expected_nav_date() → 不碰网络
        nav = _make_nav(start='2099-01-05', n=3)
        self._write_cache('000001', nav)

        def explode(**kw):
            raise AssertionError('缓存新鲜时不应触网')

        self._patch_ak(explode)
        out = frb.get_nav('000001')
        self.assertEqual(len(out), 3)
        self.assertAlmostEqual(out.iloc[-1], nav.iloc[-1])


class TestCacheCorrupt(GetNavTestBase):
    def test_corrupt_cache_refetch(self):
        path = self.tmp / 'fund_000002.csv'
        path.write_text('date,nav\nnot-a-date,xxx\n', encoding='utf-8')

        fresh = _make_nav(n=4)
        self._patch_ak(lambda **kw: _ak_df(fresh))
        out = frb.get_nav('000002')
        self.assertEqual(len(out), 4)
        # 缓存被覆写为可用内容
        reloaded = pd.read_csv(path, parse_dates=['date'], index_col='date')
        self.assertEqual(len(reloaded), 4)

    def test_empty_cache_file_refetch(self):
        path = self.tmp / 'fund_000003.csv'
        path.write_text('date,nav\n', encoding='utf-8')  # 只有表头

        fresh = _make_nav(n=3)
        self._patch_ak(lambda **kw: _ak_df(fresh))
        out = frb.get_nav('000003')
        self.assertEqual(len(out), 3)


class TestNetworkFallback(GetNavTestBase):
    def test_network_fail_with_stale_cache_degrades(self):
        # 旧缓存 (2020年, 必然过期) + 网络挂 → 降级用旧缓存
        stale = _make_nav(start='2020-01-01', n=5)
        self._write_cache('000004', stale)

        def down(**kw):
            raise ConnectionError('network down')

        self._patch_ak(down)
        out = frb.get_nav('000004')
        self.assertEqual(len(out), 5)
        self.assertEqual(out.index[-1].date(), stale.index[-1].date())

    def test_network_fail_no_cache_raises(self):
        def down(**kw):
            raise ConnectionError('network down')

        self._patch_ak(down)
        with self.assertRaises(RuntimeError) as ctx:
            frb.get_nav('000005')
        self.assertIn('000005', str(ctx.exception))  # 报错必须带基金代码

    def test_empty_api_response_counts_as_failure(self):
        # 接口返回空表 → 视为失败; 无缓存 → RuntimeError
        self._patch_ak(lambda **kw: pd.DataFrame({'净值日期': [], '单位净值': []}))
        with self.assertRaises(RuntimeError):
            frb.get_nav('000006')

    def test_empty_api_with_stale_cache_degrades(self):
        stale = _make_nav(start='2020-06-01', n=3)
        self._write_cache('000007', stale)
        self._patch_ak(lambda **kw: pd.DataFrame({'净值日期': [], '单位净值': []}))
        out = frb.get_nav('000007')
        self.assertEqual(len(out), 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
