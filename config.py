"""
全局配置
"""
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 100000.0          # 初始资金
    commission_rate: float = 0.0003            # 手续费率 (万分之三)
    slippage: float = 0.001                    # 滑点 (千分之一)
    stamp_tax: float = 0.001                   # 印花税 (卖出千一，仅A股)
    min_commission: float = 5.0                # 最低手续费
    position_size_pct: float = 0.95            # 单只股票最大仓位比例
    max_positions: int = 5                     # 最大持仓数量
    stop_loss_pct: float = 0.08                # 止损比例 -8%
    take_profit_pct: float = 0.20              # 止盈比例 +20%
    # 移动止损默认关闭: 实测在日线级别会把正常波动误判为破位,
    # 16笔移动止损全部亏损(合计-12686), 是组合最大亏损源
    trailing_stop_pct: float = 0.0             # 移动止损比例 (0=关闭)
    # 盈利后才启动移动止损: 浮盈达到该比例后, 才用移动止损保护利润
    # 这样早期正常波动不会被打断, 又能在盈利后锁定收益
    trailing_activation_pct: float = 0.10      # 浮盈达到10%后启用 (0=立即启用)
    risk_free_rate: float = 0.03               # 无风险利率 (年化3%)
    cash_yield: float = 0.02                   # 空仓资金年化收益 (货币基金约2%)
    enable_stamp_tax: bool = True              # 是否启用印花税

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class StrategyConfig:
    """策略参数基类"""
    pass


@dataclass
class MACrossConfig(StrategyConfig):
    """双均线策略参数"""
    fast_window: int = 5
    slow_window: int = 20


@dataclass
class BollingerConfig(StrategyConfig):
    """布林带均值回归策略参数"""
    window: int = 20
    num_std: float = 2.0
