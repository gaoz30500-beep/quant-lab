"""
清理本地行情缓存 (data/cache/)

缓存按"新鲜度"自动复用, 一般无需手动清理; 以下场景才需要:
- 怀疑缓存数据有误 (如复权方式变更/数据源修数)
- 切换复权口径后强制重拉

用法:
    python clean_cache.py                 # 列出将删除的文件并询问确认
    python clean_cache.py --symbol 000001 # 只清某只股票/基金/指数/ETF
    python clean_cache.py --dry-run       # 只看会删什么, 不删
    python clean_cache.py --yes           # 跳过确认直接删
"""
import argparse
import sys
from pathlib import Path

CACHE_DIR = Path(__file__).parent / 'data' / 'cache'


def collect(symbol: str = None):
    if not CACHE_DIR.exists():
        return []
    patterns = ('*.csv', '*.stamp')
    files = [p for pat in patterns for p in CACHE_DIR.glob(pat)]
    if symbol:
        # 缓存命名: {symbol}_{start}_{end}_{adjust}.csv / fund_{code}.csv / index_{code}...
        files = [p for p in files
                 if p.stem.split('_')[0] == symbol
                 or p.stem == f'fund_{symbol}'
                 or p.stem == f'index_{symbol}'
                 or p.stem == f'etf_{symbol}'
                 or f'_{symbol}_' in p.stem]
    return sorted(files)


def main():
    ap = argparse.ArgumentParser(description='清理 data/cache 行情缓存')
    ap.add_argument('--symbol', help='只清理该代码 (股票/基金/指数/ETF)')
    ap.add_argument('--dry-run', action='store_true', help='只列出, 不删除')
    ap.add_argument('--yes', '-y', action='store_true', help='跳过确认')
    args = ap.parse_args()

    files = collect(args.symbol)
    if not files:
        print('没有匹配的缓存文件')
        return 0

    total_kb = sum(p.stat().st_size for p in files) / 1024
    print(f'匹配 {len(files)} 个文件, 共 {total_kb:.0f} KB:')
    for p in files:
        print(f'  {p.name}')

    if args.dry_run:
        print('\n[dry-run] 未删除任何文件')
        return 0

    if not args.yes:
        ans = input('\n确认删除? [y/N] ').strip().lower()
        if ans != 'y':
            print('已取消')
            return 0

    for p in files:
        p.unlink()
    print(f'已删除 {len(files)} 个缓存文件')
    return 0


if __name__ == '__main__':
    sys.exit(main())
