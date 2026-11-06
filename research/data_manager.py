"""
增强版数据管理模块
- 数据质量检查和清洗
- 增量数据更新
- 多数据源支持
- 数据缓存和版本管理
- 异常检测和处理
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import hashlib
import json
import logging
from datetime import datetime, timedelta
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import warnings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        
    def check_price_data(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """检查价格数据质量"""
        issues = []
        
        # 检查必要列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"缺少必要列: {missing_columns}")
        
        # 检查数据类型
        for col in ['open', 'high', 'low', 'close']:
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"列 {col} 不是数值类型")
        
        # 检查价格合理性
        for col in ['open', 'high', 'low', 'close']:
            if (df[col] < 0).any():
                issues.append(f"列 {col} 包含负值")
            if df[col].isna().any():
                issues.append(f"列 {col} 包含空值")
        
        # 检查价格逻辑关系
        invalid_prices = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ).sum()
        if invalid_prices > 0:
            issues.append(f"发现 {invalid_prices} 条价格逻辑错误")
        
        # 检查成交量
        if (df['volume'] < 0).any():
            issues.append("成交量包含负值")
        if df['volume'].isna().any():
            issues.append("成交量包含空值")
        
        # 检查价格连续性
        price_gaps = df['close'].diff().abs() > df['close'] * 0.1  # 10%以上跳空
        if price_gaps.sum() > 0:
            issues.append(f"发现 {price_gaps.sum()} 个价格跳空")
        
        # 检查异常值 (使用IQR方法)
        for col in ['open', 'high', 'low', 'close']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > len(df) * 0.05:  # 超过5%认为是异常
                issues.append(f"列 {col} 可能有异常值: {outliers} 个")
        
        return {'issues': issues, 'passed': len(issues) == 0}
    
    def check_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查缺失数据"""
        missing_info = {}
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_ratio = missing_count / len(df)
            missing_info[col] = {
                'missing_count': missing_count,
                'missing_ratio': missing_ratio,
                'is_continuous': col in ['open', 'high', 'low', 'close']
            }
        
        # 检查连续缺失
        for col in ['open', 'high', 'low', 'close']:
            if col in missing_info:
                consecutive_missing = self._find_consecutive_missing(df[col])
                if consecutive_missing:
                    missing_info[col]['consecutive_missing'] = consecutive_missing
        
        return missing_info
    
    def _find_consecutive_missing(self, series: pd.Series) -> List[int]:
        """查找连续缺失值"""
        missing_mask = series.isna()
        consecutive_missing = []
        current_streak = 0
        
        for is_missing in missing_mask:
            if is_missing:
                current_streak += 1
            else:
                if current_streak > 0:
                    consecutive_missing.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            consecutive_missing.append(current_streak)
        
        return consecutive_missing


class DataManager:
    """增强版数据管理器"""
    
    def __init__(self, data_dir: Path, cache_dir: Path, db_path: Path):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.db_path = Path(db_path)
        self.quality_checker = DataQualityChecker()
        
        # 创建必要的目录
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 初始化数据库
        self._init_database()
        
        # 数据源配置
        self.data_sources = {
            'local_csv': self._load_local_csv,
            'api': self._load_from_api,
            'database': self._load_from_database,
        }
        
        # 缓存配置
        self.cache_expiry = timedelta(days=7)  # 缓存7天过期
        
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_metadata (
                symbol TEXT,
                data_type TEXT,
                start_date TEXT,
                end_date TEXT,
                record_count INTEGER,
                file_hash TEXT,
                quality_score REAL,
                last_updated TEXT,
                PRIMARY KEY (symbol, data_type)
            )
        ''')
        
        # 创建缓存表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_cache (
                cache_id TEXT PRIMARY KEY,
                symbol TEXT,
                data_type TEXT,
                cache_data BLOB,
                created_at TEXT,
                expires_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_data(self, symbol: str, data_type: str = 'daily',
                  start_date: Optional[str] = None, end_date: Optional[str] = None,
                  source: str = 'local_csv', **kwargs) -> pd.DataFrame:
        """
        加载数据
        
        参数:
            symbol: 代码
            data_type: 数据类型 (daily, minute, etc.)
            start_date: 开始日期
            end_date: 结束日期
            source: 数据源
            kwargs: 其他参数
        """
        # 检查缓存
        cached_data = self._get_from_cache(symbol, data_type, start_date, end_date)
        if cached_data is not None:
            logger.info(f"从缓存加载数据: {symbol}")
            return cached_data
        
        # 从数据源加载
        if source not in self.data_sources:
            raise ValueError(f"不支持的数据源: {source}")
        
        df = self.data_sources[source](symbol, data_type, start_date, end_date, **kwargs)
        
        # 数据质量检查
        quality_result = self.quality_checker.check_price_data(df)
        if not quality_result['passed']:
            logger.warning(f"数据质量检查发现问题: {quality_result['issues']}")
        
        # 数据清洗
        df = self._clean_data(df)
        
        # 更新元数据
        self._update_metadata(symbol, data_type, df, quality_result)
        
        # 缓存数据
        self._cache_data(df, symbol, data_type, start_date, end_date)
        
        return df
    
    def _load_local_csv(self, symbol: str, data_type: str,
                       start_date: Optional[str], end_date: Optional[str],
                       **kwargs) -> pd.DataFrame:
        """从本地CSV文件加载数据"""
        # 查找文件
        possible_paths = [
            self.data_dir / f"{symbol}.csv",
            self.data_dir / f"{symbol}_{data_type}.csv",
            self.data_dir / f"{symbol.lower()}.csv",
        ]
        
        file_path = None
        for path in possible_paths:
            if path.exists():
                file_path = path
                break
        
        if file_path is None:
            raise FileNotFoundError(f"未找到数据文件: {symbol}")
        
        # 加载数据
        df = pd.read_csv(file_path)
        
        # 数据预处理
        df.columns = [col.strip().lower() for col in df.columns]
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        
        # 日期过滤
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        return df
    
    def _load_from_api(self, symbol: str, data_type: str,
                      start_date: Optional[str], end_date: Optional[str],
                      **kwargs) -> pd.DataFrame:
        """从API加载数据 (示例实现)"""
        # 这里可以接入各种数据源API
        # 例如: Tushare, Wind, 同花顺等
        
        # 示例：生成模拟数据
        logger.warning("API数据加载为示例实现")
        return self._generate_sample_data(symbol, start_date, end_date)
    
    def _load_from_database(self, symbol: str, data_type: str,
                           start_date: Optional[str], end_date: Optional[str],
                           **kwargs) -> pd.DataFrame:
        """从数据库加载数据"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM data_cache WHERE symbol = ? AND data_type = ?"
        params = [symbol, data_type]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        # 删除完全空白的行
        df = df.dropna(how='all')
        
        # 处理价格数据
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                # 填充空值为前值
                df[col] = df[col].fillna(method='ffill')
                # 如果前值也为空，用后值填充
                df[col] = df[col].fillna(method='bfill')
        
        # 处理成交量
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)
        
        # 处理异常值 (用IQR方法)
        for col in price_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 将异常值替换为边界值
                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        
        return df
    
    def _generate_sample_data(self, symbol: str, start_date: Optional[str], 
                            end_date: Optional[str]) -> pd.DataFrame:
        """生成示例数据"""
        import numpy as np
        
        if start_date is None:
            start_date = '2020-01-01'
        if end_date is None:
            end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
        
        dates = pd.bdate_range(start=start_date, end=end_date)
        n = len(dates)
        
        # 生成价格数据
        np.random.seed(hash(symbol) % 1000)
        returns = np.random.normal(0.001, 0.02, n)
        prices = 100 * np.cumprod(1 + returns)
        
        # 生成OHLC数据
        high = prices * (1 + np.random.uniform(0, 0.01, n))
        low = prices * (1 - np.random.uniform(0, 0.01, n))
        opens = np.concatenate([[100], prices[:-1]])
        
        df = pd.DataFrame({
            'date': dates,
            'open': opens,
            'high': high,
            'low': low,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, n),
        })
        
        return df.set_index('date')
    
    def _update_metadata(self, symbol: str, data_type: str, 
                        df: pd.DataFrame, quality_result: Dict) -> None:
        """更新数据元数据"""
        file_hash = self._calculate_hash(df)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 计算质量分数 (0-100)
        quality_score = 100 - len(quality_result['issues']) * 10
        quality_score = max(0, quality_score)
        
        cursor.execute('''
            INSERT OR REPLACE INTO data_metadata 
            (symbol, data_type, start_date, end_date, record_count, 
             file_hash, quality_score, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol, data_type, 
            df.index.min().strftime('%Y-%m-%d'),
            df.index.max().strftime('%Y-%m-%d'),
            len(df),
            file_hash,
            quality_score,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        
        conn.commit()
        conn.close()
    
    def _cache_data(self, df: pd.DataFrame, symbol: str, 
                   data_type: str, start_date: Optional[str], 
                   end_date: Optional[str]) -> None:
        """缓存数据"""
        cache_id = self._generate_cache_id(symbol, data_type, start_date, end_date)
        expires_at = datetime.now() + self.cache_expiry
        
        # 序列化数据
        cache_data = df.to_json().encode('utf-8')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO data_cache 
            (cache_id, symbol, data_type, cache_data, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            cache_id, symbol, data_type, cache_data,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            expires_at.strftime('%Y-%m-%d %H:%M:%S')
        ))
        
        conn.commit()
        conn.close()
    
    def _get_from_cache(self, symbol: str, data_type: str,
                        start_date: Optional[str], end_date: Optional[str]) -> Optional[pd.DataFrame]:
        """从缓存获取数据"""
        cache_id = self._generate_cache_id(symbol, data_type, start_date, end_date)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT cache_data, expires_at FROM data_cache WHERE cache_id = ?', (cache_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result is None:
            return None
        
        cache_data, expires_at = result
        
        # 检查缓存是否过期
        if datetime.now() > datetime.strptime(expires_at, '%Y-%m-%d %H:%M:%S'):
            return None
        
        # 反序列化数据
        df = pd.read_json(cache_data.decode('utf-8'))
        return df
    
    def _generate_cache_id(self, symbol: str, data_type: str,
                          start_date: Optional[str], end_date: Optional[str]) -> str:
        """生成缓存ID"""
        key_parts = [symbol, data_type]
        if start_date:
            key_parts.append(start_date)
        if end_date:
            key_parts.append(end_date)
        
        key = '|'.join(key_parts)
        return hashlib.md5(key.encode()).hexdigest()
    
    def _calculate_hash(self, df: pd.DataFrame) -> str:
        """计算数据哈希值"""
        return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
    
    def update_data_incrementally(self, symbol: str, data_type: str,
                                 new_data: pd.DataFrame) -> None:
        """增量更新数据"""
        # 加载现有数据
        existing_data = self.load_data(symbol, data_type)
        
        # 合并数据
        combined_data = pd.concat([existing_data, new_data]).sort_index()
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        
        # 重新保存
        self._save_to_file(combined_data, symbol, data_type)
        
        # 更新缓存
        self._cache_data(combined_data, symbol, data_type, None, None)
        
        logger.info(f"增量更新数据: {symbol}, 新增 {len(new_data)} 条记录")
    
    def _save_to_file(self, df: pd.DataFrame, symbol: str, data_type: str) -> None:
        """保存数据到文件"""
        file_path = self.data_dir / f"{symbol}_{data_type}.csv"
        df.to_csv(file_path, encoding='utf-8-sig')
    
    def get_data_quality_report(self, symbol: str, data_type: str) -> Dict[str, Any]:
        """获取数据质量报告"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM data_metadata WHERE symbol = ? AND data_type = ?
        ''', (symbol, data_type))
        
        result = cursor.fetchone()
        conn.close()
        
        if result is None:
            return {'error': '数据未找到'}
        
        columns = ['symbol', 'data_type', 'start_date', 'end_date', 
                  'record_count', 'file_hash', 'quality_score', 'last_updated']
        
        metadata = dict(zip(columns, result))
        
        # 加载数据进行详细检查
        df = self.load_data(symbol, data_type)
        quality_info = self.quality_checker.check_missing_data(df)
        
        return {
            'metadata': metadata,
            'quality_info': quality_info,
            'data_preview': df.head().to_dict('records'),
            'data_summary': {
                'total_records': len(df),
                'date_range': f"{df.index.min()} to {df.index.max()}",
                'columns': list(df.columns),
            }
        }
    
    def cleanup_expired_cache(self) -> int:
        """清理过期缓存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 删除过期缓存
        cursor.execute('''
            DELETE FROM data_cache WHERE expires_at < ?
        ''', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        logger.info(f"清理了 {deleted_count} 条过期缓存")
        return deleted_count
    
    def batch_load_data(self, symbols: List[str], data_type: str = 'daily',
                       source: str = 'local_csv', max_workers: int = 4) -> Dict[str, pd.DataFrame]:
        """批量加载数据"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.load_data, symbol, data_type, source=source): symbol
                for symbol in symbols
            }
            
            for future in futures:
                symbol = futures[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    logger.error(f"加载 {symbol} 数据失败: {e}")
                    results[symbol] = None
        
        return results