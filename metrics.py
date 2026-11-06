"""
绩效指标计算模块
"""
import numpy as np
import pandas as pd
from typing import Dict, Any


class Metrics:
    """绩效指标计算器"""

    @staticmethod
    def calc_total_return(equity: pd.Series) -> float:
        """总收益率"""
        if len(equity) < 2:
            return 0.0
        return equity.iloc[-1] / equity.iloc[0] - 1.0

    @staticmethod
    def calc_annual_return(equity: pd.Series, periods_per_year: int = 252) -> float:
        """年化收益率"""
        if len(equity) < 2:
            return 0.0
        total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
        n_years = len(equity) / periods_per_year
        if n_years <= 0:
            return 0.0
        return (1 + total_ret) ** (1 / n_years) - 1.0

    @staticmethod
    def calc_annual_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
        """年化波动率"""
        if len(returns) < 2:
            return 0.0
        return returns.std() * np.sqrt(periods_per_year)

    @staticmethod
    def calc_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.03,
        periods_per_year: int = 252,
    ) -> float:
        """夏普比率"""
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free_rate / periods_per_year
        std = excess.std()
        if std is None or np.isnan(std) or std < 1e-10:
            return 0.0
        return np.sqrt(periods_per_year) * excess.mean() / std

    @staticmethod
    def calc_sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.03,
        periods_per_year: int = 252,
    ) -> float:
        """
        索提诺比率 = 年化超额收益 / 下行标准差

        分母必须用下行偏差 sqrt(mean(min(r - target, 0)^2)),
        只取"负收益日的标准差"会把 0 收益日排除在外, 系统性低估下行风险。
        """
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free_rate / periods_per_year
        downside = np.minimum(excess.values, 0.0)
        downside_dev = float(np.sqrt(np.mean(downside ** 2)))
        if downside_dev < 1e-10:
            return 0.0
        return np.sqrt(periods_per_year) * excess.mean() / downside_dev

    @staticmethod
    def calc_max_drawdown(equity: pd.Series) -> Dict[str, Any]:
        """最大回撤"""
        if len(equity) < 2:
            return {'max_drawdown': 0.0, 'dd_duration': 0}
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_dd = drawdown.min()

        # 回撤持续期 (最长连续低于前高的天数)
        below_peak = drawdown < 0
        duration = 0
        max_duration = 0
        for v in below_peak:
            if v:
                duration += 1
                max_duration = max(max_duration, duration)
            else:
                duration = 0

        return {
            'max_drawdown': abs(max_dd),
            'dd_duration': max_duration,
        }

    @staticmethod
    def calc_calmar_ratio(annual_return: float, max_drawdown: float) -> float:
        """卡玛比率 = 年化收益 / 最大回撤"""
        if max_drawdown == 0 or np.isnan(max_drawdown):
            return 0.0
        return annual_return / max_drawdown

    @staticmethod
    def calc_trade_stats(trades: list) -> Dict[str, Any]:
        """
        交易统计
        trades: list of dict, 每个dict包含: entry_price, exit_price, shares, pnl
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'avg_holding_days': 0.0,
                'max_win': 0.0,
                'max_loss': 0.0,
            }

        pnls = np.array([t['pnl'] for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0.0
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0

        gross_win = wins.sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
        profit_factor = gross_win / gross_loss if gross_loss > 0 else float('inf')

        holding_days = [t.get('holding_days', 0) for t in trades]

        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_days': np.mean(holding_days) if holding_days else 0.0,
            'max_win': pnls.max(),
            'max_loss': pnls.min(),
        }

    @staticmethod
    def calc_all(
        equity: pd.Series,
        trades: list,
        risk_free_rate: float = 0.03,
        periods_per_year: int = 252,
    ) -> Dict[str, Any]:
        """计算全部绩效指标"""
        returns = equity.pct_change().dropna()

        total_return = Metrics.calc_total_return(equity)
        annual_return = Metrics.calc_annual_return(equity, periods_per_year)
        annual_vol = Metrics.calc_annual_volatility(returns, periods_per_year)
        sharpe = Metrics.calc_sharpe_ratio(returns, risk_free_rate, periods_per_year)
        sortino = Metrics.calc_sortino_ratio(returns, risk_free_rate, periods_per_year)
        dd_info = Metrics.calc_max_drawdown(equity)
        calmar = Metrics.calc_calmar_ratio(annual_return, dd_info['max_drawdown'])
        trade_stats = Metrics.calc_trade_stats(trades)

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': dd_info['max_drawdown'],
            'max_drawdown_duration': dd_info['dd_duration'],
            'calmar_ratio': calmar,
            **trade_stats,
        }

    @staticmethod
    def format_metrics(metrics: Dict[str, Any]) -> str:
        """将指标格式化为可读字符串"""
        lines = [
            "=" * 50,
            "          回测绩效报告",
            "=" * 50,
            f"总收益率:       {metrics['total_return']:.2%}",
            f"年化收益率:     {metrics['annual_return']:.2%}",
            f"年化波动率:     {metrics['annual_volatility']:.2%}",
            f"夏普比率:       {metrics['sharpe_ratio']:.3f}",
            f"索提诺比率:     {metrics['sortino_ratio']:.3f}",
            f"最大回撤:       {metrics['max_drawdown']:.2%}",
            f"回撤持续天数:   {metrics['max_drawdown_duration']}",
            f"卡玛比率:       {metrics['calmar_ratio']:.3f}",
            "-" * 50,
            f"交易次数:       {metrics['total_trades']}",
            f"胜率:           {metrics['win_rate']:.2%}",
            f"平均盈利:       {metrics['avg_win']:.2f}",
            f"平均亏损:       {metrics['avg_loss']:.2f}",
            f"盈亏比:         {metrics['profit_factor']:.2f}",
            f"最大单笔盈利:   {metrics['max_win']:.2f}",
            f"最大单笔亏损:   {metrics['max_loss']:.2f}",
            f"平均持仓天数:   {metrics['avg_holding_days']:.1f}",
            "=" * 50,
        ]
        return "\n".join(lines)
