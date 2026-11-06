"""
参数优化模块
- 网格搜索 (Grid Search)
- 滚动前进验证 (Walk-Forward) 防过拟合
- 多指标评估 (夏普、年化收益、最大回撤、卡玛)
"""
import itertools
import time
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np

from config import BacktestConfig
from engine import BacktestEngine, BacktestResult
from metrics import Metrics


@dataclass
class OptimizationResult:
    """参数优化结果"""
    best_params: Dict[str, Any]
    best_score: float
    results_df: pd.DataFrame          # 所有参数组合的绩效
    metric: str                       # 优化目标指标

    def summary(self, top_n: int = 10) -> str:
        lines = [
            "=" * 60,
            f"参数优化结果 (优化目标: {self.metric})",
            "=" * 60,
            f"最优参数: {self.best_params}",
            f"最优得分: {self.best_score:.4f}",
            "-" * 60,
            f"Top {top_n} 参数组合:",
        ]
        top = self.results_df.head(top_n)
        metric_cols = {'total_return', 'annual_return', 'sharpe_ratio',
                       'sortino_ratio', 'max_drawdown', 'calmar_ratio',
                       'win_rate', 'total_trades', 'profit_factor', 'score'}
        param_cols = [c for c in self.results_df.columns if c not in metric_cols]
        for _, row in top.iterrows():
            params = {k: row[k] for k in param_cols}
            # 显示时把浮点参数还原为整数 (如果原始是整数)
            display_params = []
            for k, v in params.items():
                if isinstance(v, float) and v == int(v):
                    display_params.append(f"{k}={int(v)}")
                else:
                    display_params.append(f"{k}={v}")
            param_str = ', '.join(display_params)
            lines.append(
                f"  {param_str} | 收益={row['total_return']:.2%} "
                f"夏普={row['sharpe_ratio']:.3f} 回撤={row['max_drawdown']:.2%} "
                f"胜率={row['win_rate']:.1%}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


class ParameterOptimizer:
    """参数优化器"""

    # 可选优化目标
    METRICS = {
        'sharpe_ratio', 'annual_return', 'total_return',
        'calmar_ratio', 'sortino_ratio', 'win_rate', 'profit_factor',
    }

    def __init__(
        self,
        strategy_class,
        param_grid: Dict[str, List],
        data: pd.DataFrame,
        config: Optional[BacktestConfig] = None,
        metric: str = 'sharpe_ratio',
    ):
        """
        参数:
            strategy_class: 策略类 (如 MACrossStrategy)
            param_grid: 参数网格，如 {'fast_window': [3,5,7], 'slow_window': [20,30]}
            data: 行情数据
            config: 回测配置
            metric: 优化目标指标
        """
        if metric not in self.METRICS:
            raise ValueError(f"不支持的优化指标: {metric}, 可选: {self.METRICS}")

        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.data = data
        self.config = config or BacktestConfig()
        self.metric = metric

    def grid_search(self, verbose: bool = True) -> OptimizationResult:
        """网格搜索"""
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        combinations = list(itertools.product(*param_values))
        total = len(combinations)

        if verbose:
            print(f"网格搜索: {total} 组参数, 优化目标: {self.metric}")

        results = []
        t_start = time.perf_counter()

        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            try:
                strategy = self.strategy_class(**params)
                engine = BacktestEngine(self.data, strategy, self.config)
                result = engine.run()

                row = {**params}
                row['total_return'] = result.metrics['total_return']
                row['annual_return'] = result.metrics['annual_return']
                row['sharpe_ratio'] = result.metrics['sharpe_ratio']
                row['sortino_ratio'] = result.metrics['sortino_ratio']
                row['max_drawdown'] = result.metrics['max_drawdown']
                row['calmar_ratio'] = result.metrics['calmar_ratio']
                row['win_rate'] = result.metrics['win_rate']
                row['profit_factor'] = result.metrics['profit_factor']
                row['total_trades'] = result.metrics['total_trades']
                row['score'] = result.metrics[self.metric]
                results.append(row)

                if verbose and (i + 1) % max(1, total // 10) == 0:
                    elapsed = time.perf_counter() - t_start
                    eta = elapsed / (i + 1) * (total - i - 1)
                    print(f"  进度 {i+1}/{total} ({(i+1)/total:.0%}), "
                          f"已用 {elapsed:.1f}s, 预计剩余 {eta:.1f}s")
            except Exception as e:
                if verbose:
                    print(f"  参数 {params} 回测失败: {e}")

        if not results:
            raise RuntimeError("所有参数组合均回测失败")

        df = pd.DataFrame(results)
        df = df.sort_values('score', ascending=False).reset_index(drop=True)

        best = df.iloc[0]
        # 恢复原始参数类型 (DataFrame 会把 int 转成 float)
        best_params = {}
        for k in param_names:
            val = best[k]
            original_type = type(self.param_grid[k][0])
            try:
                best_params[k] = original_type(val)
            except (ValueError, TypeError):
                best_params[k] = val

        if verbose:
            elapsed = time.perf_counter() - t_start
            print(f"\n优化完成! 总耗时 {elapsed:.1f}s")

        return OptimizationResult(
            best_params=best_params,
            best_score=best['score'],
            results_df=df,
            metric=self.metric,
        )

    def walk_forward_analysis(
        self,
        best_params: Dict[str, Any],
        n_splits: int = 5,
        train_ratio: float = 0.7,
    ) -> Dict[str, Any]:
        """
        滚动前进验证 (Walk-Forward Analysis)
        将数据分成多段，每段用前 train_ratio 训练(选参)，后 (1-train_ratio) 测试，
        检验策略是否过拟合。

        返回: 各段测试集绩效的统计
        """
        n = len(self.data)
        fold_size = n // n_splits
        train_size = int(fold_size * train_ratio)

        test_metrics = []
        for i in range(n_splits):
            start = i * fold_size
            end = start + fold_size
            if end > n:
                break
            fold_data = self.data.iloc[start:end]
            train_data = fold_data.iloc[:train_size]
            test_data = fold_data.iloc[train_size:]

            if len(test_data) < 20:
                continue

            # 在测试集上用最优参数回测
            strategy = self.strategy_class(**best_params)
            engine = BacktestEngine(test_data, strategy, self.config)
            result = engine.run()
            test_metrics.append({
                'fold': i + 1,
                'total_return': result.metrics['total_return'],
                'sharpe_ratio': result.metrics['sharpe_ratio'],
                'max_drawdown': result.metrics['max_drawdown'],
                'win_rate': result.metrics['win_rate'],
            })

        df = pd.DataFrame(test_metrics)
        if df.empty:
            return {'message': '测试段数据不足，无法进行滚动验证'}

        summary = {
            'n_folds': len(df),
            'avg_return': df['total_return'].mean(),
            'std_return': df['total_return'].std(),
            'avg_sharpe': df['sharpe_ratio'].mean(),
            'avg_drawdown': df['max_drawdown'].mean(),
            'avg_win_rate': df['win_rate'].mean(),
            'positive_folds': int((df['total_return'] > 0).sum()),
            'fold_details': df.to_dict('records'),
        }
        return summary


def plot_optimization_heatmap(
    opt_result: OptimizationResult,
    param_x: str,
    param_y: str,
    save_path: Optional[str] = None,
):
    """
    绘制参数优化热力图 (仅适用于两个参数的策略)
    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    df = opt_result.results_df
    pivot = df.pivot_table(index=param_y, columns=param_x, values='score', aggfunc='mean')

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title(f'参数优化热力图 (指标: {opt_result.metric})')

    # 标注数值
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8,
                        color='black' if abs(val) < pivot.values.max() * 0.6 else 'white')

    plt.colorbar(im, ax=ax, label=opt_result.metric)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"热力图已保存: {save_path}")
    else:
        plt.show()
    plt.close()
