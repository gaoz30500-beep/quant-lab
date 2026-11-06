"""
统一日志配置

约定:
- 诊断信息 (数据源降级/缓存命中/重试) 走 logging, 不用 print
- 面向用户的报告输出 (回测摘要/表格) 仍可用 print, 那是 CLI 展示层
- 库模块只 getLogger(__name__), 不调 setup_logging; 入口 (main.py 等) 调一次

用法:
    from logging_setup import get_logger, setup_logging
    setup_logging()                 # 入口调用, 默认 INFO
    logger = get_logger(__name__)   # 各模块
"""
import logging
import os
import sys

_CONFIGURED = False

# 环境变量覆盖: QUANT_LOG_LEVEL=DEBUG python main.py ...
_DEFAULT_LEVEL = os.environ.get('QUANT_LOG_LEVEL', 'INFO').upper()

_FORMAT = '%(asctime)s %(levelname)-7s [%(name)s] %(message)s'
_DATEFMT = '%H:%M:%S'


def setup_logging(level: str = None, stream=sys.stderr) -> None:
    """
    入口调用一次; 重复调用幂等 (避免 argparse 子命令多次触发时叠加 handler)

    参数:
        level: 'DEBUG'/'INFO'/'WARNING'/'ERROR', 缺省读 QUANT_LOG_LEVEL 环境变量
        stream: 日志输出流, 默认 stderr (stdout 留给报告输出, 便于管道重定向)
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    # 清掉 get_logger 兜底 basicConfig 可能已挂的 handler, 否则每条日志打印两遍
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(getattr(logging, (level or _DEFAULT_LEVEL), logging.INFO))

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root.addHandler(handler)

    # 第三方库降噪: akshare/urllib3 的 DEBUG 日志非常吵
    for noisy in ('akshare', 'urllib3', 'requests', 'matplotlib'):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """模块级 logger; 若入口未调 setup_logging, 挂一个兜底 handler 保证消息不丢"""
    if not _CONFIGURED and not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=_FORMAT, datefmt=_DATEFMT)
    return logging.getLogger(name)
