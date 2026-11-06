"""
增强版监控和报告系统
- 实时监控
- 性能分析
- 归因分析
- 预警机制
- 专业报告生成
- 可视化展示
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import json
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """预警级别"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    """预警类型"""
    DRAWDOWN = "DRAWDOWN"          # 回撤预警
    VOLATILITY = "VOLATILITY"      # 波动率预警
    DRAWDOWN_DURATION = "DRAWDOWN_DURATION"  # 回撤持续时间预警
    WIN_RATE = "WIN_RATE"          # 胜率预警
    SHARPE_RATIO = "SHARPE_RATIO"  # 夏普比率预警
    POSITION_CONCENTRATION = "POSITION_CONCENTRATION"  # 持仓集中度预警
    CASH_RATIO = "CASH_RATIO"      # 现金比例预警
    STRATEGY_PERFORMANCE = "STRATEGY_PERFORMANCE"  # 策略表现预警


@dataclass
class Alert:
    """预警"""
    alert_id: str
    alert_type: AlertType
    level: AlertLevel
    message: str
    timestamp: datetime
    strategy_id: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'level': self.level.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'strategy_id': self.strategy_id,
            'value': self.value,
            'threshold': self.threshold,
            'is_resolved': self.is_resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
        }


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: datetime
    total_return: float
    daily_return: float
    cumulative_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    win_rate: float
    profit_factor: float
    trades_count: int
    positions_count: int
    cash_ratio: float
    portfolio_value: float
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_return': self.total_return,
            'daily_return': self.daily_return,
            'cumulative_return': self.cumulative_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'trades_count': self.trades_count,
            'positions_count': self.positions_count,
            'cash_ratio': self.cash_ratio,
            'portfolio_value': self.portfolio_value,
        }


@dataclass
class AttributionResult:
    """归因分析结果"""
    strategy_id: str
    period_start: datetime
    period_end: datetime
    total_return: float
    attribution: Dict[str, float]  # 各因子贡献
    residual_return: float
    r_squared: float
    factor_exposures: Dict[str, float]


class MonitoringSystem:
    """监控系统"""
    
    def __init__(self, db_path: Path, alert_thresholds: Dict = None):
        self.db_path = Path(db_path)
        self.alert_thresholds = alert_thresholds or self._default_thresholds()
        self.alerts = []
        self.metrics_history = []
        self.is_running = False
        
        # 初始化数据库
        self._init_database()
        
    def _default_thresholds(self) -> Dict:
        """默认预警阈值"""
        return {
            AlertType.DRAWDOWN: {'critical': 0.15, 'warning': 0.10},
            AlertType.VOLATILITY: {'critical': 0.30, 'warning': 0.20},
            AlertType.DRAWDOWN_DURATION: {'critical': 30, 'warning': 15},
            AlertType.WIN_RATE: {'critical': 0.40, 'warning': 0.50},
            AlertType.SHARPE_RATIO: {'critical': 0.3, 'warning': 0.5},
            AlertType.POSITION_CONCENTRATION: {'critical': 0.4, 'warning': 0.3},
            AlertType.CASH_RATIO: {'critical': 0.1, 'warning': 0.2},
            AlertType.STRATEGY_PERFORMANCE: {'critical': -0.05, 'warning': -0.02},
        }
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建指标表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                timestamp TEXT PRIMARY KEY,
                strategy_id TEXT,
                total_return REAL,
                daily_return REAL,
                cumulative_return REAL,
                volatility REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                current_drawdown REAL,
                win_rate REAL,
                profit_factor REAL,
                trades_count INTEGER,
                positions_count INTEGER,
                cash_ratio REAL,
                portfolio_value REAL
            )
        ''')
        
        # 创建预警表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                alert_type TEXT,
                level TEXT,
                message TEXT,
                timestamp TEXT,
                strategy_id TEXT,
                value REAL,
                threshold REAL,
                is_resolved BOOLEAN,
                resolved_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_monitoring(self):
        """启动监控"""
        self.is_running = True
        logger.info("监控系统已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        logger.info("监控系统已停止")
    
    def record_metrics(self, metrics: PerformanceMetrics, strategy_id: str = 'default'):
        """记录性能指标"""
        metrics.strategy_id = strategy_id
        
        # 添加到历史记录
        self.metrics_history.append(metrics)
        
        # 保存到数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO performance_metrics 
            (timestamp, strategy_id, total_return, daily_return, cumulative_return,
             volatility, sharpe_ratio, max_drawdown, current_drawdown, win_rate,
             profit_factor, trades_count, positions_count, cash_ratio, portfolio_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp.isoformat(),
            strategy_id,
            metrics.total_return,
            metrics.daily_return,
            metrics.cumulative_return,
            metrics.volatility,
            metrics.sharpe_ratio,
            metrics.max_drawdown,
            metrics.current_drawdown,
            metrics.win_rate,
            metrics.profit_factor,
            metrics.trades_count,
            metrics.positions_count,
            metrics.cash_ratio,
            metrics.portfolio_value,
        ))
        
        conn.commit()
        conn.close()
        
        # 检查预警
        self._check_alerts(metrics, strategy_id)
    
    def _check_alerts(self, metrics: PerformanceMetrics, strategy_id: str):
        """检查预警条件"""
        # 检查回撤
        if metrics.current_drawdown > self.alert_thresholds[AlertType.DRAWDOWN]['critical']:
            self.create_alert(
                AlertType.DRAWDOWN,
                AlertLevel.CRITICAL,
                f"当前回撤 {metrics.current_drawdown:.2%} 超过临界值",
                metrics.current_drawdown,
                self.alert_thresholds[AlertType.DRAWDOWN]['critical'],
                strategy_id
            )
        elif metrics.current_drawdown > self.alert_thresholds[AlertType.DRAWDOWN]['warning']:
            self.create_alert(
                AlertType.DRAWDOWN,
                AlertLevel.WARNING,
                f"当前回撤 {metrics.current_drawdown:.2%} 超过警告值",
                metrics.current_drawdown,
                self.alert_thresholds[AlertType.DRAWDOWN]['warning'],
                strategy_id
            )
        
        # 检查波动率
        if metrics.volatility > self.alert_thresholds[AlertType.VOLATILITY]['critical']:
            self.create_alert(
                AlertType.VOLATILITY,
                AlertLevel.CRITICAL,
                f"当前波动率 {metrics.volatility:.2%} 超过临界值",
                metrics.volatility,
                self.alert_thresholds[AlertType.VOLATILITY]['critical'],
                strategy_id
            )
        
        # 检查夏普比率
        if metrics.sharpe_ratio < self.alert_thresholds[AlertType.SHARPE_RATIO]['critical']:
            self.create_alert(
                AlertType.SHARPE_RATIO,
                AlertLevel.CRITICAL,
                f"夏普比率 {metrics.sharpe_ratio:.2f} 低于临界值",
                metrics.sharpe_ratio,
                self.alert_thresholds[AlertType.SHARPE_RATIO]['critical'],
                strategy_id
            )
        
        # 检查胜率
        if metrics.win_rate < self.alert_thresholds[AlertType.WIN_RATE]['critical']:
            self.create_alert(
                AlertType.WIN_RATE,
                AlertLevel.CRITICAL,
                f"胜率 {metrics.win_rate:.2%} 低于临界值",
                metrics.win_rate,
                self.alert_thresholds[AlertType.WIN_RATE]['critical'],
                strategy_id
            )
        
        # 检查持仓集中度
        if metrics.positions_count > 0:
            concentration = 1.0 / metrics.positions_count  # 简化计算
            if concentration > self.alert_thresholds[AlertType.POSITION_CONCENTRATION]['critical']:
                self.create_alert(
                    AlertType.POSITION_CONCENTRATION,
                    AlertLevel.CRITICAL,
                    f"持仓集中度 {concentration:.2%} 超过临界值",
                    concentration,
                    self.alert_thresholds[AlertType.POSITION_CONCENTRATION]['critical'],
                    strategy_id
                )
    
    def create_alert(self, alert_type: AlertType, level: AlertLevel, 
                    message: str, value: float, threshold: float, 
                    strategy_id: str = 'default'):
        """创建预警"""
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            alert_type=alert_type,
            level=level,
            message=message,
            timestamp=datetime.now(),
            strategy_id=strategy_id,
            value=value,
            threshold=threshold,
        )
        
        self.alerts.append(alert)
        
        # 保存到数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts 
            (alert_id, alert_type, level, message, timestamp, strategy_id, value, threshold, is_resolved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert.alert_id,
            alert.alert_type.value,
            alert.level.value,
            alert.message,
            alert.timestamp.isoformat(),
            alert.strategy_id,
            alert.value,
            alert.threshold,
            alert.is_resolved,
        ))
        
        conn.commit()
        conn.close()
        
        logger.warning(f"预警 [{level.value}] {message}")
    
    def resolve_alert(self, alert_id: str):
        """解决预警"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.is_resolved = True
                alert.resolved_at = datetime.now()
                
                # 更新数据库
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE alerts SET is_resolved = 1, resolved_at = ? WHERE alert_id = ?
                ''', (alert.resolved_at.isoformat(), alert.alert_id))
                
                conn.commit()
                conn.close()
                
                logger.info(f"预警已解决: {alert_id}")
                break
    
    def get_active_alerts(self, strategy_id: Optional[str] = None) -> List[Alert]:
        """获取活跃预警"""
        alerts = [alert for alert in self.alerts if not alert.is_resolved]
        if strategy_id:
            alerts = [alert for alert in alerts if alert.strategy_id == strategy_id]
        return alerts
    
    def get_metrics_history(self, strategy_id: str = 'default', 
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> List[PerformanceMetrics]:
        """获取指标历史"""
        metrics = [m for m in self.metrics_history if m.strategy_id == strategy_id]
        
        if start_date:
            metrics = [m for m in metrics if m.timestamp >= start_date]
        if end_date:
            metrics = [m for m in metrics if m.timestamp <= end_date]
        
        return sorted(metrics, key=lambda x: x.timestamp)
    
    def generate_performance_report(self, strategy_id: str = 'default',
                                 period: str = '1M') -> Dict:
        """生成性能报告"""
        # 获取历史数据
        end_date = datetime.now()
        if period == '1D':
            start_date = end_date - timedelta(days=1)
        elif period == '1W':
            start_date = end_date - timedelta(weeks=1)
        elif period == '1M':
            start_date = end_date - timedelta(days=30)
        elif period == '3M':
            start_date = end_date - timedelta(days=90)
        elif period == '1Y':
            start_date = end_date - timedelta(days=365)
        else:
            start_date = end_date - timedelta(days=30)
        
        metrics = self.get_metrics_history(strategy_id, start_date, end_date)
        
        if not metrics:
            return {'error': '无数据'}
        
        # 计算统计指标
        total_return = metrics[-1].cumulative_return
        annual_return = (1 + total_return) ** (365 / (end_date - start_date).days) - 1
        volatility = np.mean([m.volatility for m in metrics])
        sharpe_ratio = np.mean([m.sharpe_ratio for m in metrics])
        max_drawdown = max([m.max_drawdown for m in metrics])
        current_drawdown = metrics[-1].current_drawdown
        win_rate = np.mean([m.win_rate for m in metrics])
        profit_factor = np.mean([m.profit_factor for m in metrics])
        
        # 生成趋势分析
        return_trend = self._calculate_trend([m.cumulative_return for m in metrics])
        drawdown_trend = self._calculate_trend([-m.current_drawdown for m in metrics])
        
        # 获取预警
        active_alerts = self.get_active_alerts(strategy_id)
        
        return {
            'strategy_id': strategy_id,
            'period': period,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'summary': {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'current_drawdown': current_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'data_points': len(metrics),
            },
            'trends': {
                'return_trend': return_trend,
                'drawdown_trend': drawdown_trend,
            },
            'alerts': [alert.to_dict() for alert in active_alerts],
            'recommendations': self._generate_recommendations(metrics, active_alerts),
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 2:
            return 'insufficient_data'
        
        # 简单线性回归
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    def _generate_recommendations(self, metrics: List[PerformanceMetrics], 
                                alerts: List[Alert]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于回撤的建议
        current_drawdown = metrics[-1].current_drawdown
        if current_drawdown > 0.1:
            recommendations.append("当前回撤较大，建议考虑减仓或暂停交易")
        
        # 基于夏普比率的建议
        avg_sharpe = np.mean([m.sharpe_ratio for m in metrics])
        if avg_sharpe < 0.5:
            recommendations.append("夏普比率偏低，建议优化策略或降低风险")
        
        # 基于胜率的建议
        avg_win_rate = np.mean([m.win_rate for m in metrics])
        if avg_win_rate < 0.5:
            recommendations.append("胜率偏低，建议检查策略逻辑或参数")
        
        # 基于预警的建议
        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        if critical_alerts:
            recommendations.append("存在严重预警，建议立即处理")
        
        return recommendations
    
    def create_visualization(self, strategy_id: str = 'default', 
                           period: str = '1M') -> Dict:
        """创建可视化图表"""
        metrics = self.get_metrics_history(strategy_id, 
                                         datetime.now() - timedelta(days=30),
                                         datetime.now())
        
        if not metrics:
            return {'error': '无数据'}
        
        # 创建图表
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('净值曲线', '回撤', '滚动夏普比率', '每日收益分布', 
                          '持仓数量', '现金比例'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 净值曲线
        fig.add_trace(
            go.Scatter(
                x=[m.timestamp for m in metrics],
                y=[1 + m.cumulative_return for m in metrics],
                mode='lines',
                name='净值曲线',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # 回撤
        fig.add_trace(
            go.Scatter(
                x=[m.timestamp for m in metrics],
                y=[-m.current_drawdown for m in metrics],
                mode='lines',
                name='回撤',
                fill='tozeroy',
                fillcolor='red',
                line=dict(color='red')
            ),
            row=1, col=2
        )
        
        # 滚动夏普比率
        rolling_sharpe = [m.sharpe_ratio for m in metrics]
        fig.add_trace(
            go.Scatter(
                x=[m.timestamp for m in metrics],
                y=rolling_sharpe,
                mode='lines',
                name='夏普比率',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        
        # 每日收益分布
        daily_returns = [m.daily_return for m in metrics]
        fig.add_trace(
            go.Histogram(
                x=daily_returns,
                nbinsx=30,
                name='每日收益分布',
                marker_color='lightblue'
            ),
            row=2, col=2
        )
        
        # 持仓数量
        positions = [m.positions_count for m in metrics]
        fig.add_trace(
            go.Scatter(
                x=[m.timestamp for m in metrics],
                y=positions,
                mode='lines+markers',
                name='持仓数量',
                line=dict(color='orange')
            ),
            row=3, col=1
        )
        
        # 现金比例
        cash_ratio = [m.cash_ratio for m in metrics]
        fig.add_trace(
            go.Scatter(
                x=[m.timestamp for m in metrics],
                y=cash_ratio,
                mode='lines',
                name='现金比例',
                line=dict(color='purple')
            ),
            row=3, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title=f'{strategy_id} 策略表现分析',
            height=800,
            showlegend=True
        )
        
        return {
            'chart': fig.to_html(include_plotlyjs='cdn'),
            'metrics': [m.to_dict() for m in metrics],
        }
    
    def export_report(self, strategy_id: str = 'default', 
                     period: str = '1M', format: str = 'json') -> str:
        """导出报告"""
        report = self.generate_performance_report(strategy_id, period)
        
        if format == 'json':
            return json.dumps(report, ensure_ascii=False, indent=2, default=str)
        elif format == 'html':
            return self._generate_html_report(report)
        else:
            return str(report)
    
    def _generate_html_report(self, report: Dict) -> str:
        """生成HTML报告"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>量化交易策略报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }
                .alert { padding: 10px; margin: 5px 0; border-radius: 3px; }
                .alert-critical { background-color: #ffebee; border-left: 4px solid #f44336; }
                .alert-warning { background-color: #fff3e0; border-left: 4px solid #ff9800; }
                .recommendation { background-color: #e8f5e8; padding: 10px; margin: 5px 0; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>量化交易策略报告</h1>
                <p><strong>策略ID:</strong> {strategy_id}</p>
                <p><strong>分析期间:</strong> {period}</p>
                <p><strong>生成时间:</strong> {timestamp}</p>
            </div>
            
            <div class="section">
                <h2>绩效摘要</h2>
                <div class="metric">
                    <strong>总收益率:</strong> {total_return:.2%}
                </div>
                <div class="metric">
                    <strong>年化收益率:</strong> {annual_return:.2%}
                </div>
                <div class="metric">
                    <strong>夏普比率:</strong> {sharpe_ratio:.2f}
                </div>
                <div class="metric">
                    <strong>最大回撤:</strong> {max_drawdown:.2%}
                </div>
                <div class="metric">
                    <strong>胜率:</strong> {win_rate:.2%}
                </div>
                <div class="metric">
                    <strong>盈亏比:</strong> {profit_factor:.2f}
                </div>
            </div>
            
            <div class="section">
                <h2>趋势分析</h2>
                <div class="metric">
                    <strong>收益趋势:</strong> {return_trend}
                </div>
                <div class="metric">
                    <strong>回撤趋势:</strong> {drawdown_trend}
                </div>
            </div>
            
            <div class="section">
                <h2>预警信息</h2>
                {alerts_html}
            </div>
            
            <div class="section">
                <h2>改进建议</h2>
                {recommendations_html}
            </div>
        </body>
        </html>
        """
        
        # 格式化预警
        alerts_html = ""
        for alert in report.get('alerts', []):
            alert_class = f"alert-{alert['level'].lower()}"
            alerts_html += f"""
            <div class="alert {alert_class}">
                <strong>{alert['level']}:</strong> {alert['message']}
                <br><small>时间: {alert['timestamp']}</small>
            </div>
            """
        
        # 格式化建议
        recommendations_html = ""
        for rec in report.get('recommendations', []):
            recommendations_html += f"""
            <div class="recommendation">
                {rec}
            </div>
            """
        
        return html_template.format(
            strategy_id=report['strategy_id'],
            period=report['period'],
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            total_return=report['summary']['total_return'],
            annual_return=report['summary']['annual_return'],
            sharpe_ratio=report['summary']['sharpe_ratio'],
            max_drawdown=report['summary']['max_drawdown'],
            win_rate=report['summary']['win_rate'],
            profit_factor=report['summary']['profit_factor'],
            return_trend=report['trends']['return_trend'],
            drawdown_trend=report['trends']['drawdown_trend'],
            alerts_html=alerts_html,
            recommendations_html=recommendations_html,
        )


# 导入必要的模块
import uuid
import statsmodels.api as sm

# 添加缺失的统计功能
def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算R²"""
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0