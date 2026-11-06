"""
基金趋势分析脚本 - 直接分析
分析基金014409和025942的趋势，使用双均线和布林带策略
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from strategies.ma_cross import MACrossStrategy
from strategies.bollinger import BollingerStrategy
from engine import BacktestEngine
from config import BacktestConfig

def load_fund_data(fund_code: str) -> pd.DataFrame:
    """
    直接加载基金净值数据并转换为OHLCV格式
    """
    fund_file = Path(__file__).parent / 'data' / 'cache' / f'fund_{fund_code}.csv'
    
    if not fund_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {fund_file}")
    
    # 读取基金数据
    df = pd.read_csv(fund_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    
    # 将nav作为close价格
    df['close'] = df['nav']
    
    # 生成其他OHLC列（对于基金数据，我们简化处理）
    df['open'] = df['close'].shift(1).fillna(df['close'])
    df['high'] = df['close']  # 基金没有高低价，使用净值
    df['low'] = df['close']   # 基金没有高低价，使用净值
    
    # 生成成交量（模拟）
    df['volume'] = 1000000  # 模拟成交量
    
    # 删除nav列
    df = df.drop('nav', axis=1)
    
    return df

def analyze_fund(fund_code: str, fund_name: str):
    """分析单个基金的趋势"""
    print(f"\n{'='*60}")
    print(f"分析基金: {fund_name} ({fund_code})")
    print(f"{'='*60}")
    
    try:
        # 加载基金数据
        fund_data = load_fund_data(fund_code)
        print(f"数据时间范围: {fund_data.index.min()} 至 {fund_data.index.max()}")
        print(f"数据点数: {len(fund_data)}")
        
        # 设置回测配置
        config = BacktestConfig(
            initial_capital=100000.0,
            commission_rate=0.001,  # 基金手续费率
            slippage=0.001,
            stamp_tax=0.0,  # 基金没有印花税
            min_commission=1.0,
            position_size_pct=0.95,
            max_positions=1,
            stop_loss_pct=0.05,  # 基金止损5%
            take_profit_pct=0.10,  # 基金止盈10%
            trailing_stop_pct=0.0,
            trailing_activation_pct=0.0,
            risk_free_rate=0.03,
            cash_yield=0.02,
            enable_stamp_tax=False
        )
        
        # 使用双均线策略分析
        ma_strategy = MACrossStrategy(fast_window=5, slow_window=20)
        ma_engine = BacktestEngine(fund_data, ma_strategy, config)
        ma_result = ma_engine.run()
        
        # 使用布林带策略分析
        bollinger_strategy = BollingerStrategy(window=20, num_std=2.0)
        bollinger_engine = BacktestEngine(fund_data, bollinger_strategy, config)
        bollinger_result = bollinger_engine.run()
        
        # 分析结果
        print(f"\n双均线策略 ({fund_code})")
        print(f"总收益率: {ma_result.metrics['total_return']:.2%}")
        print(f"年化收益率: {ma_result.metrics['annual_return']:.2%}")
        print(f"最大回撤: {ma_result.metrics['max_drawdown']:.2%}")
        print(f"夏普比率: {ma_result.metrics['sharpe_ratio']:.2f}")
        print(f"交易次数: {len(ma_result.trades)}")
        print(f"胜率: {ma_result.metrics['win_rate']:.2%}")
        
        print(f"\n布林带策略 ({fund_code})")
        print(f"总收益率: {bollinger_result.metrics['total_return']:.2%}")
        print(f"年化收益率: {bollinger_result.metrics['annual_return']:.2%}")
        print(f"最大回撤: {bollinger_result.metrics['max_drawdown']:.2%}")
        print(f"夏普比率: {bollinger_result.metrics['sharpe_ratio']:.2f}")
        print(f"交易次数: {len(bollinger_result.trades)}")
        print(f"胜率: {bollinger_result.metrics['win_rate']:.2%}")
        
        # 获取最近的价格和趋势
        recent_data = fund_data.tail(30)  # 最近30个交易日
        latest_price = recent_data['close'].iloc[-1]
        prev_price = recent_data['close'].iloc[-2]
        price_change = (latest_price - prev_price) / prev_price
        
        print(f"\n最近表现 ({fund_code})")
        print(f"最新净值: {latest_price:.4f}")
        print(f"涨跌幅: {price_change:.2%}")
        
        # 计算简单移动平均线
        ma5 = recent_data['close'].rolling(5).mean().iloc[-1]
        ma20 = recent_data['close'].rolling(20).mean().iloc[-1]
        current_price = recent_data['close'].iloc[-1]
        
        print(f"MA5: {ma5:.4f}")
        print(f"MA20: {ma20:.4f}")
        print(f"价格位置: {current_price/ma20:.2%} of MA20")
        
        # 趋势判断
        if current_price > ma5 > ma20:
            trend = "上升趋势"
        elif current_price < ma5 < ma20:
            trend = "下降趋势"
        else:
            trend = "震荡趋势"
        
        print(f"趋势判断: {trend}")
        
        return {
            'fund_code': fund_code,
            'fund_name': fund_name,
            'latest_price': latest_price,
            'price_change': price_change,
            'ma5': ma5,
            'ma20': ma20,
            'ma_total_return': ma_result.metrics['total_return'],
            'ma_sharpe': ma_result.metrics['sharpe_ratio'],
            'bollinger_total_return': bollinger_result.metrics['total_return'],
            'bollinger_sharpe': bollinger_result.metrics['sharpe_ratio'],
            'trend': trend
        }
        
    except Exception as e:
        print(f"分析基金 {fund_code} 时出错: {e}")
        return None

def main():
    """主函数"""
    print("基金趋势分析系统")
    print("=" * 60)
    
    # 基金信息
    funds = {
        '014409': '易方达蓝筹精选混合',
        '025942': '富国天惠成长混合'
    }
    
    results = []
    
    for fund_code, fund_name in funds.items():
        result = analyze_fund(fund_code, fund_name)
        if result:
            results.append(result)
    
    # 综合建议
    print(f"\n{'='*60}")
    print("综合投资建议")
    print(f"{'='*60}")
    
    if results:
        # 按最新表现排序
        results.sort(key=lambda x: x['price_change'], reverse=True)
        
        print("\n排名（按近期表现）:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['fund_name']} ({result['fund_code']})")
            print(f"   近期涨跌: {result['price_change']:.2%}")
            print(f"   趋势: {result['trend']}")
            print(f"   双均线策略收益: {result['ma_total_return']:.2%}")
            print(f"   布林带策略收益: {result['bollinger_total_return']:.2%}")
        
        # 投资建议
        best_fund = results[0]
        print(f"\n投资建议:")
        print(f"推荐关注: {best_fund['fund_name']} ({best_fund['fund_code']})")
        print(f"理由: 近期表现较好，{best_fund['trend']}")
        
        # 风险提示
        print(f"\n风险提示:")
        print("1. 基金投资有风险，过往业绩不代表未来表现")
        print("2. 建议分散投资，不要把所有资金投入单一基金")
        print("3. 定期关注基金持仓变化和市场环境")
        print("4. 建议设置合理的止损止盈点")
        
        # 操作建议
        print(f"\n操作建议:")
        for result in results:
            if result['price_change'] > 0.02:  # 涨幅超过2%
                print(f"短期强势，可考虑适当加仓: {result['fund_name']}")
            elif result['price_change'] < -0.02:  # 跌幅超过2%
                print(f"短期调整，可考虑逢低布局: {result['fund_name']}")
            else:
                print(f"震荡整理，可继续观望: {result['fund_name']}")

if __name__ == '__main__':
    main()