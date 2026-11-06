"""
多基金趋势分析脚本
分析多个基金的趋势，使用双均线和布林带策略
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
            'trend': trend,
            'data_points': len(fund_data)
        }
        
    except Exception as e:
        print(f"分析基金 {fund_code} 时出错: {e}")
        return None

def main():
    """主函数"""
    print("多基金趋势分析系统")
    print("=" * 60)
    
    # 基金信息
    funds = {
        '018036': '长城全球新能源车',
        '012922': '易方达全球成长',
        '021190': '南方亚太精选',
        '014409': '易方达蓝筹精选混合',
        '025942': '富国天惠成长混合',
        '017992': '广发稳健增长混合',
        '018168': '华宝有色行业',
        '009478': '华安黄金ETF',
        '004253': '汇添富黄金ETF',
        '015831': '南方纯债债券'
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
        
        print("\n📊 排名（按近期表现）:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['fund_name']} ({result['fund_code']})")
            print(f"   近期涨跌: {result['price_change']:.2%}")
            print(f"   趋势: {result['trend']}")
            print(f"   双均线策略收益: {result['ma_total_return']:.2%}")
            print(f"   布林带策略收益: {result['bollinger_total_return']:.2%}")
            print(f"   数据点数: {result['data_points']}")
        
        # 投资建议
        print(f"\n🏆 推荐关注:")
        top_3 = results[:3]
        for i, result in enumerate(top_3, 1):
            print(f"{i}. {result['fund_name']} ({result['fund_code']})")
            print(f"   理由: 近期表现{result['price_change']:.2%}，{result['trend']}")
        
        # 风险提示
        print(f"\n⚠️ 风险提示:")
        print("1. 基金投资有风险，过往业绩不代表未来表现")
        print("2. 建议分散投资，不要把所有资金投入单一基金")
        print("3. 定期关注基金持仓变化和市场环境")
        print("4. 建议设置合理的止损止盈点")
        
        # 操作建议
        print(f"\n🎯 操作建议:")
        for result in results:
            if result['price_change'] > 0.03:  # 涨幅超过3%
                print(f"📈 强势推荐: {result['fund_name']} - 近期涨幅{result['price_change']:.2%}，{result['trend']}")
            elif result['price_change'] > 0.01:  # 涨幅超过1%
                print(f"📊 关注: {result['fund_name']} - 近期涨幅{result['price_change']:.2%}，{result['trend']}")
            elif result['price_change'] < -0.03:  # 跌幅超过3%
                print(f"📉 观望: {result['fund_name']} - 近期跌幅{abs(result['price_change']):.2%}，{result['trend']}")
            else:
                print(f"➡️ 观察: {result['fund_name']} - 近期表现平稳，{result['trend']}")
        
        # 策略表现对比
        print(f"\n📈 策略表现对比:")
        print("基金类型 | 双均线策略 | 布林带策略 | 近期表现")
        print("-" * 50)
        for result in results:
            fund_type = result['fund_name'].split(' ')[0] if ' ' in result['fund_name'] else result['fund_name']
            print(f"{fund_type:8} | {result['ma_total_return']:8.2%} | {result['bollinger_total_return']:8.2%} | {result['price_change']:8.2%}")

if __name__ == '__main__':
    main()