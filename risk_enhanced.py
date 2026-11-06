"""
增强版风控模块
- 动态风控参数调整
- 投资组合风险管理
- 极端行情保护
- 压力测试功能
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from enum import Enum
import warnings
from datetime import datetime, timedelta

from config import BacktestConfig


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Position:
    """持仓"""
    symbol: str
    shares: int
    entry_price: float
    entry_date: pd.Timestamp
    highest_price: float           # 持仓期间最高价 (用于移动止损)
    stop_loss_price: float
    take_profit_price: float
    unrealized_pnl: float = 0.0    # 浮动盈亏
    unrealized_pnl_pct: float = 0.0 # 浮动盈亏比例
    drawdown_from_peak: float = 0.0 # 从最高点回撤比例

    def update(self, price: float) -> None:
        """更新持仓状态"""
        # 更新最高价
        if price > self.highest_price:
            self.highest_price = price
            self.drawdown_from_peak = 0.0
        else:
            # 计算从最高点的回撤
            if self.highest_price > 0:
                self.drawdown_from_peak = (self.highest_price - price) / self.highest_price
        
        # 更新浮动盈亏
        self.unrealized_pnl = (price - self.entry_price) * self.shares
        self.unrealized_pnl_pct = self.unrealized_pnl / (self.entry_price * self.shares)


@dataclass
class PortfolioRiskMetrics:
    """投资组合风险指标"""
    total_value: float
    total_exposure: float          # 总风险敞口
    cash_ratio: float              # 现金比例
    portfolio_var: float            # 投资组合VaR
    portfolio_volatility: float     # 投资组合波动率
    max_drawdown: float            # 最大回撤
    concentration_risk: float      # 集中度风险
    correlation_risk: float         # 相关性风险
    risk_level: RiskLevel

    def to_dict(self) -> Dict:
        return {
            'total_value': self.total_value,
            'total_exposure': self.total_exposure,
            'cash_ratio': self.cash_ratio,
            'portfolio_var': self.portfolio_var,
            'portfolio_volatility': self.portfolio_volatility,
            'max_drawdown': self.max_drawdown,
            'concentration_risk': self.concentration_risk,
            'correlation_risk': self.correlation_risk,
            'risk_level': self.risk_level.value,
        }


class EnhancedRiskManager:
    """增强版风控管理器"""

    def __init__(
        self,
        base_config: BacktestConfig,
        enable_dynamic_risk: bool = True,
        enable_portfolio_risk: bool = True,
        enable_circuit_breaker: bool = True,
    ):
        """
        初始化增强风控管理器
        
        参数:
            base_config: 基础配置
            enable_dynamic_risk: 是否启用动态风控
            enable_portfolio_risk: 是否启用组合风控
            enable_circuit_breaker: 是否启用熔断机制
        """
        self.base_config = base_config
        
        # 动态风控参数
        self.enable_dynamic_risk = enable_dynamic_risk
        self.volatility_window = 20  # 波动率计算窗口
        self.volatility_multiplier = 2.0  # 波动率调整倍数
        
        # 组合风控参数
        self.enable_portfolio_risk = enable_portfolio_risk
        self.max_portfolio_var = 0.10  # 投资组合VaR上限 (10%)
        self.max_portfolio_volatility = 0.20  # 投资组合波动率上限 (20%)
        self.max_concentration = 0.30  # 单个持仓最大比例 (30%)
        
        # 熔断机制参数
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_breaker_threshold = 0.15  # 单日最大亏损 (15%)
        self.daily_loss_reset = True  # 每日重置亏损计数器
        
        # 风险状态
        self.risk_metrics = PortfolioRiskMetrics(
            total_value=0.0,
            total_exposure=0.0,
            cash_ratio=1.0,
            portfolio_var=0.0,
            portfolio_volatility=0.0,
            max_drawdown=0.0,
            concentration_risk=0.0,
            correlation_risk=0.0,
            risk_level=RiskLevel.LOW
        )
        
        # 动态风控状态
        self.daily_losses = []  # 每日亏损记录
        self.volatility_history = []  # 波动率历史
        self.risk_adjustment_factor = 1.0  # 风险调整因子

    def update_risk_metrics(self, positions: Dict[str, Position], 
                          cash: float, current_prices: Dict[str, float]) -> None:
        """更新投资组合风险指标"""
        if not positions:
            self.risk_metrics = PortfolioRiskMetrics(
                total_value=cash,
                total_exposure=0.0,
                cash_ratio=1.0,
                portfolio_var=0.0,
                portfolio_volatility=0.0,
                max_drawdown=0.0,
                concentration_risk=0.0,
                correlation_risk=0.0,
                risk_level=RiskLevel.LOW
            )
            return

        # 计算总价值和风险敞口
        total_value = cash + sum(pos.shares * current_prices.get(pos.symbol, pos.entry_price) 
                               for pos in positions.values())
        total_exposure = sum(pos.shares * current_prices.get(pos.symbol, pos.entry_price) 
                           for pos in positions.values())
        cash_ratio = cash / total_value if total_value > 0 else 1.0

        # 计算集中度风险
        position_values = {pos.symbol: pos.shares * current_prices.get(pos.symbol, pos.entry_price) 
                          for pos in positions.values()}
        max_position_ratio = max(position_values.values()) / total_value if total_value > 0 else 0.0
        concentration_risk = max_position_ratio / self.max_concentration

        # 计算投资组合VaR (简化计算)
        portfolio_var = self._calculate_portfolio_var(positions, current_prices)
        
        # 计算投资组合波动率
        portfolio_volatility = self._calculate_portfolio_volatility(positions)

        # 计算最大回撤
        max_drawdown = max([pos.drawdown_from_peak for pos in positions.values()], default=0.0)

        # 确定风险等级
        risk_level = self._determine_risk_level(
            total_value, total_exposure, portfolio_var, portfolio_volatility, max_drawdown
        )

        self.risk_metrics = PortfolioRiskMetrics(
            total_value=total_value,
            total_exposure=total_exposure,
            cash_ratio=cash_ratio,
            portfolio_var=portfolio_var,
            portfolio_volatility=portfolio_volatility,
            max_drawdown=max_drawdown,
            concentration_risk=concentration_risk,
            correlation_risk=0.0,  # 简化版本，实际需要相关性矩阵
            risk_level=risk_level
        )

    def check_circuit_breaker(self, daily_pnl: float) -> bool:
        """检查是否触发熔断"""
        if not self.enable_circuit_breaker:
            return False

        # 记录每日亏损
        if self.daily_loss_reset:
            # 简单实现：如果当日亏损超过阈值，触发熔断
            if daily_pnl < -self.circuit_breaker_threshold * self.base_config.initial_capital:
                return True
        
        return False

    def get_dynamic_risk_params(self) -> Dict[str, float]:
        """获取动态风控参数"""
        if not self.enable_dynamic_risk:
            return {
                'stop_loss_pct': self.base_config.stop_loss_pct,
                'take_profit_pct': self.base_config.take_profit_pct,
                'position_size_pct': self.base_config.position_size_pct,
            }

        # 根据市场波动率调整参数
        current_volatility = self._get_current_volatility()
        volatility_factor = min(current_volatility / 0.02, 3.0)  # 相对于2%基准波动率的倍数

        # 高波动率时收紧风控
        adjusted_stop_loss = self.base_config.stop_loss_pct * (1 + 0.5 * volatility_factor)
        adjusted_take_profit = self.base_config.take_profit_pct * (1 - 0.3 * volatility_factor)
        adjusted_position_size = self.base_config.position_size_pct * (1 - 0.4 * volatility_factor)

        return {
            'stop_loss_pct': min(adjusted_stop_loss, 0.20),  # 最大20%止损
            'take_profit_pct': max(adjusted_take_profit, 0.05),  # 最小5%止盈
            'position_size_pct': max(adjusted_position_size, 0.3),  # 最小30%仓位
        }

    def check_portfolio_risk(self, positions: Dict[str, Position], 
                           cash: float, current_prices: Dict[str, float]) -> Tuple[bool, str]:
        """检查投资组合风险"""
        if not self.enable_portfolio_risk:
            return False, "组合风控已禁用"

        self.update_risk_metrics(positions, cash, current_prices)

        # 检查VaR
        if self.risk_metrics.portfolio_var > self.max_portfolio_var:
            return True, f"投资组合VaR ({self.risk_metrics.portfolio_var:.2%}) 超过上限 ({self.max_portfolio_var:.2%})"

        # 检查波动率
        if self.risk_metrics.portfolio_volatility > self.max_portfolio_volatility:
            return True, f"投资组合波动率 ({self.risk_metrics.portfolio_volatility:.2%}) 超过上限 ({self.max_portfolio_volatility:.2%})"

        # 检查集中度
        if self.risk_metrics.concentration_risk > 1.0:
            return True, f"投资组合集中度风险过高 ({self.risk_metrics.concentration_risk:.2f})"

        return False, "组合风险可控"

    def get_position_limits(self, symbol: str, current_positions: Dict[str, Position],
                          cash: float, current_price: float) -> Dict[str, int]:
        """获取单个持仓的限制"""
        limits = {
            'max_shares': 0,
            'min_shares': 0,
            'recommended_position_size': 0,
        }

        # 基础限制
        base_max_shares = self.base_config.max_positions

        # 动态风控调整
        if self.enable_dynamic_risk:
            dynamic_params = self.get_dynamic_risk_params()
            max_allocation = cash * dynamic_params['position_size_pct']
            max_shares = int(max_allocation / current_price)
            base_max_shares = min(base_max_shares, max_shares // 100) * 100  # 100股整数倍

        # 组合风控调整
        if self.enable_portfolio_risk:
            current_total_positions = sum(pos.shares for pos in current_positions.values())
            if current_total_positions >= base_max_shares:
                limits['max_shares'] = 0
            else:
                remaining_slots = base_max_shares - current_total_positions
                limits['max_shares'] = remaining_slots

        # 推荐仓位大小
        if limits['max_shares'] > 0:
            recommended_size = int(cash * 0.2 / current_price) * 100  # 推荐使用20%资金
            limits['recommended_position_size'] = min(recommended_size, limits['max_shares'])

        return limits

    def stress_test(self, positions: Dict[str, Position], cash: float,
                   price_scenarios: Dict[str, float]) -> Dict[str, float]:
        """压力测试"""
        stress_results = {}

        for scenario_name, price_changes in price_scenarios.items():
            scenario_value = cash
            for symbol, position in positions.items():
                new_price = position.entry_price * (1 + price_changes.get(symbol, 0))
                scenario_value += position.shares * new_price

            pnl = scenario_value - (cash + sum(pos.shares * pos.entry_price for pos in positions.values()))
            pnl_pct = pnl / (cash + sum(pos.shares * pos.entry_price for pos in positions.values()))
            
            stress_results[scenario_name] = {
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'final_value': scenario_value,
            }

        return stress_results

    def _calculate_portfolio_var(self, positions: Dict[str, Position], 
                               current_prices: Dict[str, float]) -> float:
        """计算投资组合VaR (简化版本)"""
        # 简化计算：假设各持仓独立，使用历史波动率
        if not positions:
            return 0.0

        total_value = sum(pos.shares * current_prices.get(pos.symbol, pos.entry_price) 
                        for pos in positions.values())
        
        # 简单VaR计算：95%置信度，1.65倍标准差
        weighted_volatility = 0.0
        for pos in positions.values():
            position_weight = pos.shares * current_prices.get(pos.symbol, pos.entry_price) / total_value
            # 假设20%年化波动率
            position_volatility = 0.20 / np.sqrt(252)  # 日波动率
            weighted_volatility += position_weight * position_volatility

        var = 1.65 * weighted_volatility * total_value
        return var / total_value if total_value > 0 else 0.0

    def _calculate_portfolio_volatility(self, positions: Dict[str, Position]) -> float:
        """计算投资组合波动率"""
        if not positions:
            return 0.0

        # 简化计算：加权平均波动率
        total_value = sum(pos.shares * pos.entry_price for pos in positions.values())
        weighted_volatility = 0.0
        
        for pos in positions.values():
            position_weight = pos.shares * pos.entry_price / total_value
            # 假设20%年化波动率
            position_volatility = 0.20 / np.sqrt(252)  # 日波动率
            weighted_volatility += position_weight * position_volatility

        return weighted_volatility

    def _get_current_volatility(self) -> float:
        """获取当前市场波动率"""
        if not self.volatility_history:
            return 0.02  # 默认2%

        # 使用最近20天的平均波动率
        recent_volatility = np.mean(self.volatility_history[-20:])
        return recent_volatility

    def _determine_risk_level(self, total_value: float, total_exposure: float,
                            portfolio_var: float, portfolio_volatility: float,
                            max_drawdown: float) -> RiskLevel:
        """确定风险等级"""
        # 计算综合风险评分
        risk_score = 0.0

        # VaR贡献
        if portfolio_var > self.max_portfolio_var:
            risk_score += (portfolio_var - self.max_portfolio_var) / self.max_portfolio_var

        # 波动率贡献
        if portfolio_volatility > self.max_portfolio_volatility:
            risk_score += (portfolio_volatility - self.max_portfolio_volatility) / self.max_portfolio_volatility

        # 回撤贡献
        if max_drawdown > 0.10:  # 10%回撤
            risk_score += max_drawdown / 0.10

        # 确定风险等级
        if risk_score < 0.5:
            return RiskLevel.LOW
        elif risk_score < 1.0:
            return RiskLevel.MEDIUM
        elif risk_score < 2.0:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def record_daily_performance(self, daily_pnl: float, portfolio_value: float) -> None:
        """记录每日表现数据"""
        self.daily_losses.append(daily_pnl)
        
        # 保持最近100天的记录
        if len(self.daily_losses) > 100:
            self.daily_losses.pop(0)

        # 更新波动率历史
        daily_return = daily_pnl / portfolio_value if portfolio_value > 0 else 0
        self.volatility_history.append(abs(daily_return))
        
        if len(self.volatility_history) > 252:  # 保持一年数据
            self.volatility_history.pop(0)

    def get_risk_report(self) -> Dict[str, any]:
        """生成风险报告"""
        return {
            'current_risk_level': self.risk_metrics.risk_level.value,
            'risk_metrics': self.risk_metrics.to_dict(),
            'dynamic_params': self.get_dynamic_risk_params(),
            'recent_volatility': self._get_current_volatility(),
            'daily_loss_stats': {
                'avg_daily_loss': np.mean(self.daily_losses) if self.daily_losses else 0.0,
                'max_daily_loss': min(self.daily_losses) if self.daily_losses else 0.0,
                'volatility': np.std(self.daily_losses) if self.daily_losses else 0.0,
            },
            'risk_recommendations': self._get_risk_recommendations(),
        }

    def _get_risk_recommendations(self) -> List[str]:
        """获取风险建议"""
        recommendations = []

        if self.risk_metrics.risk_level == RiskLevel.HIGH:
            recommendations.append("建议降低仓位，收紧风控参数")
        elif self.risk_metrics.risk_level == RiskLevel.CRITICAL:
            recommendations.append("建议立即减仓，考虑暂停交易")

        if self.risk_metrics.concentration_risk > 1.0:
            recommendations.append("建议分散投资，降低集中度风险")

        if self.risk_metrics.portfolio_volatility > self.max_portfolio_volatility:
            recommendations.append("建议降低投资组合波动率")

        return recommendations