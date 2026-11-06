"""
真实行情数据源模块
封装 AKShare，提供统一接口，含重试 + 本地CSV缓存
"""
import json
import time
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from datetime import datetime, timedelta, date

from logging_setup import get_logger

logger = get_logger(__name__)


def validate_daily(
    df: pd.DataFrame,
    calendar: Optional[pd.DatetimeIndex] = None,
    jump_thresh: float = 0.25,
) -> Tuple[pd.DataFrame, Dict]:
    """
    日线数据校验

    只修复确定有害的问题 (重复日期 / 非正价格), 其余 (OHLC 矛盾 / 异常跳变 /
    缺失交易日) 仅报告, 不擅自删改:
      - duplicate_dates   重复日期条数 (保留最后一条)
      - non_positive      非正价格行数 (已剔除)
      - ohlc_inconsistent 收盘价不在当日高低区间内的行数
      - abnormal_jumps    单日涨跌幅超过 jump_thresh 的次数 (疑似数据错误)
      - missing_days      交易日历中存在但个股缺失的交易日 (正常停牌也会计入)

    返回 (清洗后的数据, 问题报告); 无问题时报告为空 dict
    """
    issues: Dict = {}

    if df.index.has_duplicates:
        issues['duplicate_dates'] = int(df.index.duplicated().sum())
        df = df[~df.index.duplicated(keep='last')].sort_index()

    px_cols = [c for c in ('open', 'high', 'low', 'close') if c in df.columns]
    if px_cols:
        bad = (df[px_cols] <= 0).any(axis=1)
        if bad.any():
            issues['non_positive'] = int(bad.sum())
            df = df[~bad]

    if {'open', 'high', 'low', 'close'} <= set(df.columns) and len(df) > 0:
        inconsistent = (
            (df['high'] < df['low'])
            | (df['close'] > df['high'] * 1.0001)
            | (df['close'] < df['low'] * 0.9999)
        )
        if inconsistent.any():
            issues['ohlc_inconsistent'] = int(inconsistent.sum())

    if len(df) > 1:
        jumps = df['close'].pct_change().abs()
        abnormal = jumps[jumps > jump_thresh]
        if len(abnormal) > 0:
            issues['abnormal_jumps'] = len(abnormal)
            issues['jump_dates'] = [str(d.date()) for d in abnormal.index[:3]]

    if calendar is not None and len(df) > 0:
        lo, hi = df.index[0], df.index[-1]
        cal = pd.DatetimeIndex(calendar)
        cal = cal[(cal >= lo) & (cal <= hi)]
        missing = cal.difference(df.index)
        if len(missing) > 0:
            issues['missing_days'] = len(missing)
            issues['missing_examples'] = [str(d.date()) for d in missing[:3]]

    return df, issues


class DataSource:
    """行情数据源 (AKShare)"""

    CACHE_DIR = Path(__file__).parent / 'data' / 'cache'
    MARKET_CLOSE_HOUR = 15   # A股收盘时间, 收盘前当天K线视为未完成

    def __init__(self, use_cache: bool = True, max_retries: int = 3):
        self.use_cache = use_cache
        self.max_retries = max_retries
        if self.use_cache:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- 公共接口 ----------------

    def get_stock_daily(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = 'qfq',
    ) -> pd.DataFrame:
        """
        获取A股日线数据 (多源自动降级: 东方财富 -> 新浪)

        参数:
            symbol: 股票代码，如 '000001'
            start_date: 'YYYYMMDD' 或 'YYYY-MM-DD'
            end_date: 'YYYYMMDD' 或 'YYYY-MM-DD'
            adjust: 复权类型 'qfq'(前复权) / 'hfq'(后复权) / ''(不复权)

        返回:
            标准化 DataFrame，索引为date，列: open, high, low, close, volume
        """
        symbol = self._normalize_symbol(symbol)

        # 尝试缓存
        if self.use_cache:
            cached = self._load_cache(symbol, start_date, end_date, adjust)
            if cached is not None:
                return cached

        s = self._fmt_date(start_date) or '20100101'
        e = self._fmt_date(end_date) or datetime.now().strftime('%Y%m%d')

        # 数据源1: 东方财富 (stock_zh_a_hist)
        df = self._request_with_retry(
            'stock_zh_a_hist',
            symbol=symbol, period='daily',
            start_date=s, end_date=e, adjust=adjust,
        )

        # 数据源2: 新浪 (stock_zh_a_daily) — 需加市场前缀
        if df is None or df.empty:
            logger.warning("股票 %s: 东方财富失败, 切换至新浪源", symbol)
            sina_symbol = self._add_market_prefix(symbol)
            df = self._request_with_retry(
                'stock_zh_a_daily',
                symbol=sina_symbol,
                start_date=s, end_date=e, adjust=adjust,
            )

        if df is None or df.empty:
            raise RuntimeError(f"获取股票 {symbol} 数据失败 (所有数据源均不可用)")

        df = self._normalize_columns(df)

        if self.use_cache:
            self._save_cache(df, symbol, start_date, end_date, adjust)

        return df

    def get_stock_list(self) -> pd.DataFrame:
        """获取A股股票列表"""
        df = self._request_with_retry('stock_info_a_code_name')
        if df is None or df.empty:
            raise RuntimeError("获取股票列表失败")
        return df

    def get_realtime_quote(self, symbol: str) -> Optional[dict]:
        """
        获取实时行情快照 (东方财富 -> 新浪HTTP直连 降级)
        返回: {name, price, open, high, low, pre_close, volume, amount, pct_change}
        """
        symbol = self._normalize_symbol(symbol)

        # 数据源1: 东方财富实时行情
        try:
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            row = df[df['代码'] == symbol]
            if not row.empty:
                r = row.iloc[0]
                return {
                    'symbol': symbol,
                    'name': r.get('名称', ''),
                    'price': float(r.get('最新价', 0)),
                    'open': float(r.get('今开', 0)),
                    'high': float(r.get('最高', 0)),
                    'low': float(r.get('最低', 0)),
                    'pre_close': float(r.get('昨收', 0)),
                    'volume': float(r.get('成交量', 0)),
                    'amount': float(r.get('成交额', 0)),
                    'pct_change': float(r.get('涨跌幅', 0)),
                }
        except Exception as e:
            logger.warning("实时行情 %s: 东方财富获取失败: %s", symbol, str(e)[:60])

        # 数据源2: 新浪 HTTP 直连 (单只股票，轻量快速)
        logger.info("实时行情 %s: 切换至新浪直连接口", symbol)
        return self._sina_realtime_quote(symbol)

    def _sina_realtime_quote(self, symbol: str) -> Optional[dict]:
        """新浪实时行情 HTTP 直连"""
        import requests
        sina_symbol = self._add_market_prefix(symbol)
        url = f"http://hq.sinajs.cn/list={sina_symbol}"
        headers = {'Referer': 'https://finance.sina.com.cn'}

        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.encoding = 'gbk'
            text = resp.text.strip()
            # 格式: var hq_str_sz000001="名称,今开,昨收,现价,最高,最低,...,成交量,...";
            content = text.split('"')[1]
            if not content:
                return None
            fields = content.split(',')
            name = fields[0]
            open_price = float(fields[1])
            pre_close = float(fields[2])
            price = float(fields[3])
            high = float(fields[4])
            low = float(fields[5])
            volume = float(fields[8])
            amount = float(fields[9])
            pct_change = ((price - pre_close) / pre_close * 100) if pre_close > 0 else 0.0

            return {
                'symbol': symbol,
                'name': name,
                'price': price,
                'open': open_price,
                'high': high,
                'low': low,
                'pre_close': pre_close,
                'volume': volume,
                'amount': amount,
                'pct_change': pct_change,
            }
        except Exception as e:
            logger.error("实时行情 %s: 新浪直连获取失败: %s", symbol, e)
            return None

    def get_index_daily(
        self,
        symbol: str = '000001',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取指数日线 (如上证指数 000001, 深证成指 399001)
        东方财富 -> 新浪 自动降级; 与个股一致走本地缓存 + 新鲜度判定

        指数是市场闸门与交易日历的唯一来源, 取不到时闸门会退化为"全开"、
        日历退化为个股并集, 回测边界条件被静默改变。因此这里必须能离线复现:
        缓存键加 index_ 前缀, 与个股 CSV 分开存放
        """
        s = self._fmt_date(start_date) or '20100101'
        e = self._fmt_date(end_date) or datetime.now().strftime('%Y%m%d')
        cache_key = f'index_{symbol}'

        if self.use_cache:
            cached = self._load_cache(cache_key, s, e, 'raw')
            if cached is not None:
                return cached

        # 数据源1: 东方财富
        df = self._request_with_retry(
            'stock_zh_index_daily_em', symbol=symbol,
            start_date=s, end_date=e,
        )

        # 数据源2: 新浪 (需加 sh/sz 前缀，指数用 sh)
        if df is None or df.empty:
            logger.warning("指数 %s: 东方财富接口失败, 切换至新浪源", symbol)
            # 指数前缀: 上证指数 000001 → sh, 深证成指 399001 → sz
            sina_symbol = f'sz{symbol}' if symbol.startswith('399') else f'sh{symbol}'
            df = self._request_with_retry(
                'stock_zh_index_daily', symbol=sina_symbol,
            )

        if df is None or df.empty:
            raise RuntimeError(f"获取指数 {symbol} 数据失败 (所有数据源均不可用)")

        df = self._normalize_columns(df)

        # 新浪源返回全量历史, 统一按请求区间裁剪 (东财源在接口层已裁剪, 这里幂等)
        if start_date:
            df = df[df.index >= pd.to_datetime(s)]
        if end_date:
            df = df[df.index <= pd.to_datetime(e)]

        if self.use_cache and not df.empty:
            self._save_cache(df, cache_key, s, e, 'raw')

        return df

    def get_etf_daily(
        self,
        symbol: str = '510300',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取场内 ETF 日线 (如沪深300ETF 510300)
        新浪 -> 东方财富 自动降级

        注意: 新浪源为未复权价, ETF 分红会导致收益轻微低估
        """
        s = self._fmt_date(start_date) or '20100101'
        e = self._fmt_date(end_date) or datetime.now().strftime('%Y%m%d')

        cached = self._load_cache(f'etf_{symbol}', s, e, 'qfq')
        if cached is not None and not cached.empty:
            return cached

        # 数据源1: 新浪 (沪市 5 开头, 深市 1 开头)
        prefix = 'sh' if symbol.startswith('5') else 'sz'
        df = self._request_with_retry('fund_etf_hist_sina', symbol=f'{prefix}{symbol}')

        # 数据源2: 东方财富
        if df is None or df.empty:
            logger.warning("ETF %s: 新浪接口失败, 切换至东方财富", symbol)
            df = self._request_with_retry(
                'fund_etf_hist_em', symbol=symbol, period='daily',
                start_date=s, end_date=e, adjust='qfq',
            )

        if df is None or df.empty:
            raise RuntimeError(f"获取 ETF {symbol} 数据失败 (所有数据源均不可用)")

        df = self._normalize_columns(df)
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        if df.empty:
            raise RuntimeError(f"ETF {symbol} 在指定区间内无数据")

        self._save_cache(df, f'etf_{symbol}', s, e, 'qfq')
        return df

    # ---------------- 内部方法 ----------------

    def _request_with_retry(self, func_name: str, **kwargs) -> Optional[pd.DataFrame]:
        """带重试的请求"""
        import akshare as ak
        func = getattr(ak, func_name, None)
        if func is None:
            raise AttributeError(f"AKShare 无此接口: {func_name}")

        for attempt in range(1, self.max_retries + 1):
            try:
                df = func(**kwargs)
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                logger.warning("%s 第%d/%d次请求失败: %s",
                               func_name, attempt, self.max_retries, e)
                if attempt < self.max_retries:
                    time.sleep(2 * attempt)  # 指数退避
        return None

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """标准化股票代码，去掉前缀和后缀"""
        symbol = symbol.strip()
        # 去掉 sh/sz 前缀
        if symbol.lower().startswith(('sh', 'sz', 'bj')):
            symbol = symbol[2:]
        # 去掉 .SZ/.SH 后缀
        symbol = symbol.split('.')[0]
        return symbol

    @staticmethod
    def _add_market_prefix(symbol: str) -> str:
        """根据股票代码判断市场，添加新浪源需要的前缀"""
        if symbol.startswith(('60', '68', '90', '11', '13')):
            return f'sh{symbol}'
        elif symbol.startswith(('00', '30', '20')):
            return f'sz{symbol}'
        elif symbol.startswith(('4', '8')):
            return f'bj{symbol}'
        return f'sz{symbol}'  # 默认深市

    @staticmethod
    def _fmt_date(date_str: Optional[str]) -> Optional[str]:
        """日期格式化为 YYYYMMDD"""
        if not date_str:
            return None
        date_str = date_str.strip()
        if '-' in date_str:
            return date_str.replace('-', '')
        return date_str

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """将AKShare中文列名转为标准英文列名"""
        col_map = {
            '日期': 'date', 'date': 'date',
            '开盘': 'open', 'open': 'open',
            '收盘': 'close', 'close': 'close',
            '最高': 'high', 'high': 'high',
            '最低': 'low', 'low': 'low',
            '成交量': 'volume', 'volume': 'volume',
            '成交额': 'amount', 'amount': 'amount',
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if 'date' not in df.columns:
            # 有些接口 date 已经是索引
            if df.index.name in ('date', '日期'):
                df = df.reset_index()
            else:
                raise ValueError("数据中找不到日期列")

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')

        # 确保数值列
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['close'])
        return self._drop_incomplete_bar(df)

    def _drop_incomplete_bar(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        丢弃当日尚未收盘的 K 线

        盘中拉数据会拿到"半根日线"(收盘价 = 当前价), 若直接入库, 回测在当日
        就用到了当天最终价, 等于提前知道了当天走势。收盘后(15:00)才保留。
        """
        if df.empty:
            return df
        now = datetime.now()
        if now.hour < self.MARKET_CLOSE_HOUR and df.index[-1].date() == now.date():
            return df.iloc[:-1]
        return df

    # ---------------- 缓存 ----------------

    def _cache_path(self, symbol: str, start_date, end_date, adjust: str) -> Path:
        s = self._fmt_date(start_date) or 'all'
        e = self._fmt_date(end_date) or 'all'
        name = f"{symbol}_{s}_{e}_{adjust}.csv"
        return self.CACHE_DIR / name

    def _load_cache(self, symbol, start_date, end_date, adjust) -> Optional[pd.DataFrame]:
        path = self._cache_path(symbol, start_date, end_date, adjust)
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, parse_dates=['date'], index_col='date')
        except Exception:
            return None
        if df.empty:
            return None

        last_bar = df.index.max()
        expected = self._expected_bar_date(end_date)
        if self._is_cache_fresh(path, last_bar, expected):
            logger.info("缓存命中 %s (最新K线 %s, 应为 %s)", symbol, last_bar.date(), expected)
            return df

        if last_bar.date() >= expected:
            logger.info("缓存 %s 需更新 (当日K线可能未收盘), 重新拉取", symbol)
        else:
            logger.info("缓存 %s 已过期 (最新K线 %s < 应为 %s), 重新拉取",
                        symbol, last_bar.date(), expected)
        return None

    def _save_cache(self, df: pd.DataFrame, symbol, start_date, end_date, adjust):
        path = self._cache_path(symbol, start_date, end_date, adjust)
        try:
            df.to_csv(path, encoding='utf-8-sig')
            if len(df) > 0:
                self._write_stamp(path, df.index.max())
            logger.info("缓存保存 %s -> %s", symbol, path.name)
        except Exception as e:
            logger.warning("缓存保存失败 %s: %s", symbol, e)

    # ---------- 缓存新鲜度判定 ----------

    @staticmethod
    def _prev_weekday(day: date) -> date:
        """回退到最近的工作日 (周六/周日不算交易日)"""
        while day.weekday() >= 5:
            day -= timedelta(days=1)
        return day

    @classmethod
    def _expected_bar_date(cls, end_date: Optional[str]) -> date:
        """
        缓存里"本应已经存在"的最新日K日期。

        规则: 15点前当天未收盘 → 取上一工作日; 15点后 → 取当天(若为周末则回退)。
        本地没有交易日历, 法定节假日无法预先判断, 由"当日已尝试"标记兜底
        (节假日首次会多请求一次, 之后当天不再重复请求)。
        指定了历史 end_date 时以请求区间为准, 避免为了历史区间去拉最新数据。
        """
        now = datetime.now()
        today = now.date()
        day = today if now.hour >= cls.MARKET_CLOSE_HOUR else today - timedelta(days=1)
        day = cls._prev_weekday(day)
        if end_date:
            req = pd.to_datetime(cls._fmt_date(end_date)).date()
            day = min(day, cls._prev_weekday(req))
        return day

    def _stamp_path(self, cache_path: Path) -> Path:
        """与 CSV 同名的旁路标记文件, 不改变 CSV 字段结构"""
        return cache_path.with_suffix('.stamp')

    def _read_stamp(self, cache_path: Path) -> dict:
        p = self._stamp_path(cache_path)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            return {}

    def _write_stamp(self, cache_path: Path, last_bar: pd.Timestamp) -> None:
        p = self._stamp_path(cache_path)
        try:
            p.write_text(json.dumps({
                'checked_at': datetime.now().isoformat(timespec='seconds'),
                'last_bar_date': str(pd.Timestamp(last_bar).date()),
            }, ensure_ascii=False), encoding='utf-8')
        except Exception as e:
            logger.warning("缓存标记写入失败 %s: %s", cache_path.name, e)

    @staticmethod
    def _parse_check_time(stamp: dict) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(str(stamp.get('checked_at', '')))
        except ValueError:
            return None

    def _is_cache_fresh(self, cache_path: Path, last_bar: pd.Timestamp, expected: date) -> bool:
        now = datetime.now()
        bar_date = pd.Timestamp(last_bar).date()
        stamp = self._read_stamp(cache_path)
        checked = self._parse_check_time(stamp)
        same_bar = str(stamp.get('last_bar_date', '')) == str(bar_date)
        checked_today = bool(checked) and checked.date() == date.today()

        if bar_date >= expected:
            # 当天K线可能是在收盘前拉到的盘中数据, 收盘后允许重拉一次
            intraday = (bar_date == now.date() and same_bar and checked_today
                        and checked.hour < self.MARKET_CLOSE_HOUR <= now.hour)
            return not intraday

        # 数据滞后于应有交易日: 若当天已经成功拉取过且K线仍停在同一根,
        # 说明是节假日/数据源未更新, 当天不再重复请求 (次日会重新判断)
        if not same_bar or not checked_today:
            return False
        # 盘中检查过、现在已收盘 → 允许再试一次, 以拿到当天K线
        if checked.hour < self.MARKET_CLOSE_HOUR <= now.hour:
            return False
        return True
