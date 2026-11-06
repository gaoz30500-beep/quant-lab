"""
可视化模块
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from engine import BacktestResult


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class Plotter:
    """回测结果可视化"""

    @staticmethod
    def plot_result(result: BacktestResult, save_path: Optional[str] = None) -> None:
        """
        绘制综合图表:
        1. 价格 + 买卖点 + 指标
        2. 净值曲线 + 回撤 + 基准
        3. 交易盈亏分布
        """
        fig, axes = plt.subplots(
            3, 1, figsize=(14, 12),
            gridspec_kw={'height_ratios': [3, 2, 1]},
        )
        fig.suptitle('回测绩效分析', fontsize=14, fontweight='bold')

        price = result.price_data['close'] if result.price_data is not None else None

        # ---- 子图1: 价格 + 指标 + 买卖点 ----
        ax1 = axes[0]
        if price is not None:
            ax1.plot(price.index, price.values, label='收盘价', color='#333333', linewidth=1)

            # 绘制指标线
            for col in result.indicator_data.columns:
                ax1.plot(
                    result.indicator_data.index,
                    result.indicator_data[col].values,
                    label=col, linewidth=0.9, alpha=0.8,
                )

            # 买卖点
            buy_mask = result.signals == 'BUY'
            sell_mask = result.signals == 'SELL'
            if buy_mask.any():
                ax1.scatter(
                    price.index[buy_mask], price[buy_mask],
                    marker='^', color='#e74c3c', s=90, label='买入', zorder=5,
                    edgecolors='darkred', linewidths=0.5,
                )
            if sell_mask.any():
                ax1.scatter(
                    price.index[sell_mask], price[sell_mask],
                    marker='v', color='#27ae60', s=90, label='卖出', zorder=5,
                    edgecolors='darkgreen', linewidths=0.5,
                )
            ax1.legend(loc='upper left', fontsize=9)
        ax1.set_title('价格走势与交易信号')
        ax1.set_ylabel('价格')
        ax1.grid(True, alpha=0.3)

        # ---- 子图2: 净值曲线 + 回撤 + 基准 ----
        ax2 = axes[1]
        equity = result.equity_curve
        ax2.plot(equity.index, equity.values, label='策略净值', color='#2980b9', linewidth=1.3)

        # 买入持有基准
        if price is not None:
            benchmark = price / price.iloc[0] * result.config.initial_capital
            ax2.plot(
                benchmark.index, benchmark.values,
                label='买入持有基准', color='#95a5a6', linewidth=0.9, alpha=0.7, linestyle='--',
            )

        # 回撤填充
        cummax = equity.cummax()
        ax2.fill_between(
            equity.index, equity.values, cummax.values,
            color='#e74c3c', alpha=0.15, label='回撤',
        )
        ax2.legend(loc='upper left', fontsize=9)
        ax2.set_title(f"净值曲线 (最终净值: {equity.iloc[-1]:,.2f})")
        ax2.set_ylabel('净值')
        ax2.grid(True, alpha=0.3)

        # ---- 子图3: 每笔交易盈亏 ----
        ax3 = axes[2]
        if result.trades:
            pnls = [t.pnl for t in result.trades]
            colors = ['#e74c3c' if p >= 0 else '#27ae60' for p in pnls]
            ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.75, edgecolor='white', linewidth=0.3)
            ax3.axhline(y=0, color='black', linewidth=0.6)
            ax3.set_title(f'每笔交易盈亏 (共{len(pnls)}笔, 胜率: {sum(1 for p in pnls if p>0)/len(pnls):.1%})')
            ax3.set_xlabel('交易序号')
            ax3.set_ylabel('盈亏金额')
        else:
            ax3.text(0.5, 0.5, '无交易记录', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('每笔交易盈亏')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        else:
            plt.show()
        plt.close()

    @staticmethod
    def plot_equity_only(result: BacktestResult, save_path: Optional[str] = None) -> None:
        """仅绘制净值曲线"""
        fig, ax = plt.subplots(figsize=(12, 5))
        equity = result.equity_curve
        ax.plot(equity.index, equity.values, label='策略净值', color='#2980b9', linewidth=1.3)
        ax.fill_between(
            equity.index, equity.values, equity.cummax().values,
            color='#e74c3c', alpha=0.15, label='回撤',
        )
        ax.legend()
        ax.set_title('净值曲线')
        ax.set_ylabel('净值')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
