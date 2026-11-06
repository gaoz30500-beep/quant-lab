"""
logging_setup 回归测试

守护的 bug: 模块 import 时 get_logger 的兜底 basicConfig 先挂一个 handler,
入口再调 setup_logging() 又挂一个 → 每条日志打印两遍 (2026-11-06 实际发生)。
"""
import sys
import logging
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import logging_setup


class TestSetupLogging(unittest.TestCase):

    def setUp(self):
        logging_setup._CONFIGURED = False
        self.root = logging.getLogger()
        self._saved = list(self.root.handlers)
        self.root.handlers.clear()

    def tearDown(self):
        self.root.handlers.clear()
        self.root.handlers.extend(self._saved)
        logging_setup._CONFIGURED = False

    def test_fallback_then_setup_no_duplicate(self):
        logging_setup.get_logger('some.module')   # 兜底 basicConfig 挂 1 个
        logging_setup.setup_logging()              # 必须清掉兜底再挂自己的
        self.assertEqual(len(self.root.handlers), 1,
                         '兜底 handler 与 setup_logging 叠加会导致日志重复打印')

    def test_setup_idempotent(self):
        logging_setup.setup_logging()
        logging_setup.setup_logging()
        self.assertEqual(len(self.root.handlers), 1)

    def test_noisy_libs_quieted(self):
        logging_setup.setup_logging()
        self.assertEqual(logging.getLogger('akshare').level, logging.WARNING)
        self.assertEqual(logging.getLogger('urllib3').level, logging.WARNING)


if __name__ == '__main__':
    unittest.main(verbosity=2)
