"""
生成示例行情数据到 CSV (用于测试真实数据加载流程)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import DataLoader


def main():
    output_dir = Path(__file__).parent / 'data'
    output_dir.mkdir(exist_ok=True)

    # 生成3只模拟股票
    stocks = {
        'SAMPLE_A': {'seed': 42, 'volatility': 0.02, 'drift': 0.0003},
        'SAMPLE_B': {'seed': 123, 'volatility': 0.03, 'drift': 0.0001},
        'SAMPLE_C': {'seed': 999, 'volatility': 0.015, 'drift': 0.0005},
    }

    for symbol, params in stocks.items():
        df = DataLoader.generate_sample_data(
            days=500,
            start_price=50 + hash(symbol) % 100,
            **params,
        )
        # 重置索引以便 CSV 包含 date 列
        df_reset = df.reset_index()
        filepath = output_dir / f'{symbol}.csv'
        df_reset.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"已生成: {filepath} ({len(df)} 条记录)")

    print(f"\n示例数据已保存至: {output_dir}")


if __name__ == '__main__':
    main()
