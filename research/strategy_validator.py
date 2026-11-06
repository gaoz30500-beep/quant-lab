"""
增强版策略验证模块
- 多维度过拟合检测
- 策略稳定性分析
- 市场regime变化测试
- 统计显著性检验
- 蒙特卡洛模拟
- 压力测试
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class ValidationMetric(Enum):
    """验证指标枚举"""
    SHARPE_RATIO = "sharpe_ratio"
    CALMAR_RATIO = "calmar_ratio"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    MAX_DRAWDOWN = "max_drawdown"
    STABILITY = "stability"
    ROBUSTNESS = "robustness"
    OVERFITTING_SCORE = "overfitting_score"


@dataclass
class ValidationResult:
    """验证结果"""
    strategy_name: str
    metric_name: ValidationMetric
    value: float
    confidence_interval: Tuple[float, float]
    p_value: float
    significance: bool
    interpretation: str


@dataclass
class StrategyValidationReport:
    """策略验证报告"""
    strategy_name: str
    validation_period: str
    sample_period: str
    out_of_sample_period: str
    metrics: Dict[str, float]
    overfitting_analysis: Dict[str, Any]
    stability_analysis: Dict[str, Any]
    regime_analysis: Dict[str, Any]
    monte_carlo_results: Dict[str, Any]
    recommendations: List[str]
    overall_score: float


class StrategyValidator:
    """策略验证器"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def comprehensive_validation(self, strategy, data: pd.DataFrame, 
                              config: Dict, n_splits: int = 5) -> StrategyValidationReport:
        """全面策略验证"""
        logger.info(f"开始全面验证策略: {strategy.name}")
        
        # 1. 基础性能测试
        base_metrics = self._test_base_performance(strategy, data, config)
        
        # 2. 过拟合检测
        overfitting_analysis = self._detect_overfitting(strategy, data, config, n_splits)
        
        # 3. 策略稳定性分析
        stability_analysis = self._analyze_stability(strategy, data, config)
        
        # 4. 市场regime分析
        regime_analysis = self._analyze_regime_robustness(strategy, data, config)
        
        # 5. 蒙特卡洛模拟
        monte_carlo_results = self._monte_carlo_simulation(strategy, data, config)
        
        # 6. 综合评分和建议
        overall_score = self._calculate_overall_score(base_metrics, overfitting_analysis, 
                                                     stability_analysis, regime_analysis)
        recommendations = self._generate_recommendations(base_metrics, overfitting_analysis, 
                                                      stability_analysis, regime_analysis)
        
        return StrategyValidationReport(
            strategy_name=strategy.name,
            validation_period=f"{data.index.min()} to {data.index.max()}",
            sample_period="训练集",
            out_of_sample_period="测试集",
            metrics=base_metrics,
            overfitting_analysis=overfitting_analysis,
            stability_analysis=stability_analysis,
            regime_analysis=regime_analysis,
            monte_carlo_results=monte_carlo_results,
            recommendations=recommendations,
            overall_score=overall_score
        )
    
    def _test_base_performance(self, strategy, data: pd.DataFrame, 
                              config: Dict) -> Dict[str, float]:
        """基础性能测试"""
        try:
            # 这里需要调用回测引擎
            # 简化实现，返回示例数据
            returns = np.random.normal(0.001, 0.02, len(data))
            equity = 100000 * np.cumprod(1 + returns)
            
            metrics = {
                'total_return': (equity[-1] / equity[0] - 1),
                'annual_return': (equity[-1] / equity[0]) ** (252/len(data)) - 1,
                'sharpe_ratio': np.sqrt(252) * np.mean(returns) / np.std(returns),
                'max_drawdown': self._calculate_max_drawdown(equity),
                'win_rate': np.mean(returns > 0),
                'profit_factor': np.sum(returns[returns > 0]) / np.sum(np.abs(returns[returns < 0])),
                'calmar_ratio': ((equity[-1] / equity[0]) ** (252/len(data)) - 1) / self._calculate_max_drawdown(equity),
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"基础性能测试失败: {e}")
            return {}
    
    def _detect_overfitting(self, strategy, data: pd.DataFrame, 
                           config: Dict, n_splits: int = 5) -> Dict[str, Any]:
        """过拟合检测"""
        logger.info("开始过拟合检测")
        
        # 时间序列交叉验证
        cv_results = self._time_series_cross_validation(strategy, data, config, n_splits)
        
        # 参数敏感性分析
        sensitivity_analysis = self._parameter_sensitivity_analysis(strategy, data, config)
        
        # 样本外测试
        oos_results = self._out_of_sample_test(strategy, data, config)
        
        # 计算过拟合分数
        overfitting_score = self._calculate_overfitting_score(cv_results, oos_results)
        
        return {
            'cross_validation': cv_results,
            'sensitivity_analysis': sensitivity_analysis,
            'out_of_sample': oos_results,
            'overfitting_score': overfitting_score,
            'is_overfitting': overfitting_score > 0.3,  # 阈值可调
        }
    
    def _time_series_cross_validation(self, strategy, data: pd.DataFrame, 
                                    config: Dict, n_splits: int) -> Dict[str, Any]:
        """时间序列交叉验证"""
        n = len(data)
        fold_size = n // n_splits
        
        cv_scores = []
        train_metrics = []
        test_metrics = []
        
        for i in range(n_splits):
            # 分割数据
            train_end = i * fold_size
            test_start = train_end
            test_end = min((i + 1) * fold_size, n)
            
            train_data = data.iloc[:train_end]
            test_data = data.iloc[test_start:test_end]
            
            if len(train_data) < 100 or len(test_data) < 50:  # 最小数据量要求
                continue
            
            # 训练和测试
            try:
                train_result = self._run_backtest(strategy, train_data, config)
                test_result = self._run_backtest(strategy, test_data, config)
                
                train_metrics.append(train_result['metrics'])
                test_metrics.append(test_result['metrics'])
                cv_scores.append(test_result['metrics']['sharpe_ratio'])
                
            except Exception as e:
                logger.warning(f"交叉验证第{i+1}折失败: {e}")
                continue
        
        if not cv_scores:
            return {'error': '交叉验证失败'}
        
        return {
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'stability_ratio': np.mean(cv_scores) / (np.std(cv_scores) + 1e-6),
        }
    
    def _parameter_sensitivity_analysis(self, strategy, data: pd.DataFrame, 
                                       config: Dict) -> Dict[str, Any]:
        """参数敏感性分析"""
        logger.info("进行参数敏感性分析")
        
        # 获取策略参数
        strategy_params = getattr(strategy, 'params', {})
        if not strategy_params:
            return {'error': '策略无参数'}
        
        sensitivity_results = {}
        
        # 对每个参数进行扰动分析
        for param_name, param_value in strategy_params.items():
            if isinstance(param_value, (int, float)):
                # 扰动参数值
                perturbations = [
                    param_value * 0.8,  # 减少20%
                    param_value * 0.9,  # 减少10%
                    param_value,        # 原值
                    param_value * 1.1,  # 增加10%
                    param_value * 1.2,  # 增加20%
                ]
                
                param_results = []
                for perturbed_value in perturbations:
                    try:
                        # 创建扰动后的策略
                        perturbed_strategy = self._create_perturbed_strategy(
                            strategy, param_name, perturbed_value
                        )
                        
                        result = self._run_backtest(perturbed_strategy, data, config)
                        param_results.append({
                            'param_value': perturbed_value,
                            'sharpe_ratio': result['metrics']['sharpe_ratio'],
                            'total_return': result['metrics']['total_return'],
                            'max_drawdown': result['metrics']['max_drawdown'],
                        })
                        
                    except Exception as e:
                        logger.warning(f"参数{param_name}值{perturbed_value}测试失败: {e}")
                        continue
                
                sensitivity_results[param_name] = param_results
        
        # 计算敏感性指标
        sensitivity_metrics = {}
        for param_name, results in sensitivity_results.items():
            if results:
                sharpe_ratios = [r['sharpe_ratio'] for r in results]
                sensitivity_metrics[param_name] = {
                    'mean_sharpe': np.mean(sharpe_ratios),
                    'std_sharpe': np.std(sharpe_ratios),
                    'sensitivity_index': np.std(sharpe_ratios) / (np.mean(sharpe_ratios) + 1e-6),
                    'robustness_score': 1 - np.std(sharpe_ratios) / (np.max(sharpe_ratios) - np.min(sharpe_ratios) + 1e-6),
                }
        
        return {
            'sensitivity_results': sensitivity_results,
            'sensitivity_metrics': sensitivity_metrics,
            'most_sensitive_param': max(sensitivity_metrics.items(), 
                                      key=lambda x: x[1]['sensitivity_index'])[0] if sensitivity_metrics else None,
        }
    
    def _out_of_sample_test(self, strategy, data: pd.DataFrame, 
                           config: Dict) -> Dict[str, Any]:
        """样本外测试"""
        logger.info("进行样本外测试")
        
        # 使用前70%作为训练集，后30%作为测试集
        split_point = int(len(data) * 0.7)
        train_data = data.iloc[:split_point]
        test_data = data.iloc[split_point:]
        
        try:
            # 训练集表现
            train_result = self._run_backtest(strategy, train_data, config)
            
            # 测试集表现
            test_result = self._run_backtest(strategy, test_data, config)
            
            # 比较分析
            performance_gap = {
                'sharpe_gap': test_result['metrics']['sharpe_ratio'] - train_result['metrics']['sharpe_ratio'],
                'return_gap': test_result['metrics']['total_return'] - train_result['metrics']['total_return'],
                'drawdown_gap': test_result['metrics']['max_drawdown'] - train_result['metrics']['max_drawdown'],
            }
            
            # 统计显著性检验
            t_stat, p_value = ttest_ind(
                train_result['returns'], 
                test_result['returns'], 
                equal_var=False
            )
            
            return {
                'train_metrics': train_result['metrics'],
                'test_metrics': test_result['metrics'],
                'performance_gap': performance_gap,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_difference': p_value < self.alpha,
            }
            
        except Exception as e:
            logger.error(f"样本外测试失败: {e}")
            return {'error': str(e)}
    
    def _analyze_stability(self, strategy, data: pd.DataFrame, 
                          config: Dict) -> Dict[str, Any]:
        """策略稳定性分析"""
        logger.info("分析策略稳定性")
        
        # 滚动窗口分析
        window_sizes = [20, 60, 120]  # 1个月、3个月、6个月
        stability_results = {}
        
        for window_size in window_sizes:
            rolling_metrics = []
            
            for i in range(window_size, len(data), 30):  # 每月滚动一次
                window_data = data.iloc[i-window_size:i]
                
                try:
                    result = self._run_backtest(strategy, window_data, config)
                    rolling_metrics.append({
                        'window_end': window_data.index[-1],
                        'sharpe_ratio': result['metrics']['sharpe_ratio'],
                        'total_return': result['metrics']['total_return'],
                        'max_drawdown': result['metrics']['max_drawdown'],
                    })
                    
                except Exception as e:
                    continue
            
            if rolling_metrics:
                stability_results[f'{window_size}_day'] = {
                    'metrics': rolling_metrics,
                    'volatility': np.std([m['sharpe_ratio'] for m in rolling_metrics]),
                    'consistency': 1 - np.std([m['sharpe_ratio'] for m in rolling_metrics]) / (np.mean([m['sharpe_ratio'] for m in rolling_metrics]) + 1e-6),
                    'trend': self._calculate_trend([m['sharpe_ratio'] for m in rolling_metrics]),
                }
        
        # 计算总体稳定性指标
        overall_stability = np.mean([s['consistency'] for s in stability_results.values()])
        
        return {
            'rolling_analysis': stability_results,
            'overall_stability': overall_stability,
            'is_stable': overall_stability > 0.7,  # 阈值可调
        }
    
    def _analyze_regime_robustness(self, strategy, data: pd.DataFrame, 
                                  config: Dict) -> Dict[str, Any]:
        """市场regime稳健性分析"""
        logger.info("分析市场regime稳健性")
        
        # 定义市场regime
        regimes = self._identify_market_regimes(data)
        
        regime_performance = {}
        for regime_name, regime_mask in regimes.items():
            regime_data = data[regime_mask]
            
            if len(regime_data) < 50:  # 最小数据量要求
                continue
            
            try:
                result = self._run_backtest(strategy, regime_data, config)
                regime_performance[regime_name] = {
                    'data_points': len(regime_data),
                    'sharpe_ratio': result['metrics']['sharpe_ratio'],
                    'total_return': result['metrics']['total_return'],
                    'max_drawdown': result['metrics']['max_drawdown'],
                    'win_rate': result['metrics']['win_rate'],
                }
                
            except Exception as e:
                logger.warning(f"Regime {regime_name} 测试失败: {e}")
                continue
        
        # 分析regime适应性
        if regime_performance:
            sharpe_ratios = [p['sharpe_ratio'] for p in regime_performance.values()]
            regime_adaptability = 1 - np.std(sharpe_ratios) / (np.mean(sharpe_ratios) + 1e-6)
            
            # 找出最佳和最差regime
            best_regime = max(regime_performance.items(), key=lambda x: x[1]['sharpe_ratio'])
            worst_regime = min(regime_performance.items(), key=lambda x: x[1]['sharpe_ratio'])
            
            return {
                'regime_performance': regime_performance,
                'regime_adaptability': regime_adaptability,
                'best_regime': best_regime,
                'worst_regime': worst_regime,
                'regime_consistency': regime_adaptability > 0.6,
            }
        
        return {'error': '无法分析regime稳健性'}
    
    def _monte_carlo_simulation(self, strategy, data: pd.DataFrame, 
                               config: Dict, n_simulations: int = 1000) -> Dict[str, Any]:
        """蒙特卡洛模拟"""
        logger.info("进行蒙特卡洛模拟")
        
        # 获取历史收益率
        returns = data['close'].pct_change().dropna()
        
        # 模拟结果
        simulation_results = []
        
        for _ in range(n_simulations):
            # 生成随机收益率序列
            simulated_returns = np.random.choice(returns, size=len(returns), replace=True)
            
            # 计算模拟净值
            simulated_equity = 100000 * np.cumprod(1 + simulated_returns)
            
            # 计算指标
            total_return = simulated_equity[-1] / simulated_equity[0] - 1
            annual_return = (1 + total_return) ** (252/len(returns)) - 1
            max_drawdown = self._calculate_max_drawdown(simulated_equity)
            sharpe_ratio = np.sqrt(252) * np.mean(simulated_returns) / np.std(simulated_returns)
            
            simulation_results.append({
                'total_return': total_return,
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
            })
        
        # 统计分析
        results_df = pd.DataFrame(simulation_results)
        
        return {
            'simulation_count': n_simulations,
            'statistics': {
                'mean_total_return': results_df['total_return'].mean(),
                'std_total_return': results_df['total_return'].std(),
                'mean_sharpe': results_df['sharpe_ratio'].mean(),
                'std_sharpe': results_df['sharpe_ratio'].std(),
                'max_drawdown_p95': results_df['max_drawdown'].quantile(0.95),
            },
            'percentiles': {
                'total_return_5th': results_df['total_return'].quantile(0.05),
                'total_return_95th': results_df['total_return'].quantile(0.95),
                'sharpe_ratio_5th': results_df['sharpe_ratio'].quantile(0.05),
                'sharpe_ratio_95th': results_df['sharpe_ratio'].quantile(0.95),
            },
            'probability_of_positive_return': (results_df['total_return'] > 0).mean(),
        }
    
    def _calculate_overfitting_score(self, cv_results: Dict, oos_results: Dict) -> float:
        """计算过拟合分数"""
        if 'error' in cv_results or 'error' in oos_results:
            return 1.0  # 无法计算，认为是过拟合
        
        # 计算交叉验证和样本外表现的差异
        cv_sharpe = cv_results.get('mean_score', 0)
        oos_sharpe = oos_results.get('test_metrics', {}).get('sharpe_ratio', 0)
        
        if cv_sharpe == 0:
            return 1.0
        
        performance_decay = max(0, (cv_sharpe - oos_sharpe) / cv_sharpe)
        
        # 计算稳定性因素
        cv_stability = cv_results.get('stability_ratio', 0)
        stability_factor = max(0, 1 - cv_stability)
        
        # 综合过拟合分数
        overfitting_score = 0.5 * performance_decay + 0.5 * stability_factor
        
        return min(1.0, overfitting_score)
    
    def _calculate_overall_score(self, base_metrics: Dict, overfitting_analysis: Dict,
                               stability_analysis: Dict, regime_analysis: Dict) -> float:
        """计算综合评分"""
        scores = []
        
        # 基础性能评分 (0-1)
        if base_metrics:
            sharpe_score = min(1.0, max(0.0, (base_metrics.get('sharpe_ratio', 0) + 0.5) / 1.0))
            scores.append(sharpe_score * 0.3)
        
        # 过拟合评分 (反向，0-1)
        overfitting_score = overfitting_analysis.get('overfitting_score', 0.5)
        scores.append((1 - overfitting_score) * 0.3)
        
        # 稳定性评分 (0-1)
        if stability_analysis.get('overall_stability') is not None:
            stability_score = stability_analysis['overall_stability']
            scores.append(stability_score * 0.2)
        
        # Regime适应性评分 (0-1)
        if regime_analysis.get('regime_adaptability') is not None:
            regime_score = regime_analysis['regime_adaptability']
            scores.append(regime_score * 0.2)
        
        return np.mean(scores) if scores else 0.0
    
    def _generate_recommendations(self, base_metrics: Dict, overfitting_analysis: Dict,
                               stability_analysis: Dict, regime_analysis: Dict) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 过拟合相关建议
        if overfitting_analysis.get('is_overfitting', False):
            recommendations.append("策略可能过拟合，建议简化模型或增加正则化")
        
        # 稳定性相关建议
        if stability_analysis.get('is_stable', False) is False:
            recommendations.append("策略稳定性不足，建议优化参数或调整策略逻辑")
        
        # Regime适应性建议
        if regime_analysis.get('regime_consistency', False) is False:
            recommendations.append("策略在不同市场regime下表现差异较大，建议增加自适应机制")
        
        # 性能建议
        if base_metrics.get('sharpe_ratio', 0) < 0.5:
            recommendations.append("策略夏普比率偏低，建议优化风险调整收益")
        
        if base_metrics.get('max_drawdown', 0) > 0.2:
            recommendations.append("策略最大回撤较大，建议加强风险管理")
        
        return recommendations
    
    def _identify_market_regimes(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """识别市场regime"""
        returns = data['close'].pct_change().dropna()
        
        # 基于波动率regime
        rolling_vol = returns.rolling(20).std() * np.sqrt(252)
        high_vol_mask = rolling_vol > rolling_vol.quantile(0.75)
        low_vol_mask = rolling_vol < rolling_vol.quantile(0.25)
        
        # 基于趋势regime
        ma_20 = data['close'].rolling(20).mean()
        ma_50 = data['close'].rolling(50).mean()
        uptrend_mask = (ma_20 > ma_50) & (ma_20 > ma_20.shift(1))
        downtrend_mask = (ma_20 < ma_50) & (ma_20 < ma_20.shift(1))
        
        return {
            'high_volatility': high_vol_mask,
            'low_volatility': low_vol_mask,
            'uptrend': uptrend_mask,
            'downtrend': downtrend_mask,
        }
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """计算最大回撤"""
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        return abs(drawdown.min())
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 2:
            return 'insufficient_data'
        
        # 简单线性回归判断趋势
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    def _run_backtest(self, strategy, data: pd.DataFrame, config: Dict) -> Dict:
        """运行回测（简化实现）"""
        # 这里应该调用实际的回测引擎
        # 返回模拟结果
        returns = np.random.normal(0.001, 0.02, len(data))
        equity = 100000 * np.cumprod(1 + returns)
        
        return {
            'returns': returns,
            'equity': equity,
            'metrics': {
                'total_return': (equity[-1] / equity[0] - 1),
                'annual_return': (equity[-1] / equity[0]) ** (252/len(data)) - 1,
                'sharpe_ratio': np.sqrt(252) * np.mean(returns) / np.std(returns),
                'max_drawdown': self._calculate_max_drawdown(equity),
                'win_rate': np.mean(returns > 0),
                'profit_factor': np.sum(returns[returns > 0]) / np.sum(np.abs(returns[returns < 0])),
                'calmar_ratio': ((equity[-1] / equity[0]) ** (252/len(data)) - 1) / self._calculate_max_drawdown(equity),
            }
        }
    
    def _create_perturbed_strategy(self, strategy, param_name: str, param_value: Any):
        """创建扰动后的策略"""
        # 这里应该实现策略参数扰动逻辑
        # 简化实现，返回原策略
        return strategy
    
    def save_validation_report(self, report: StrategyValidationReport, 
                              filepath: Path) -> None:
        """保存验证报告"""
        report_dict = {
            'strategy_name': report.strategy_name,
            'validation_period': report.validation_period,
            'metrics': report.metrics,
            'overfitting_analysis': report.overfitting_analysis,
            'stability_analysis': report.stability_analysis,
            'regime_analysis': report.regime_analysis,
            'monte_carlo_results': report.monte_carlo_results,
            'recommendations': report.recommendations,
            'overall_score': report.overall_score,
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"验证报告已保存: {filepath}")
    
    def generate_validation_summary(self, report: StrategyValidationReport) -> str:
        """生成验证摘要"""
        summary = []
        summary.append(f"策略验证报告: {report.strategy_name}")
        summary.append("=" * 50)
        summary.append(f"综合评分: {report.overall_score:.2f}/1.00")
        summary.append("")
        
        # 关键指标
        if report.metrics:
            summary.append("关键性能指标:")
            for metric, value in report.metrics.items():
                summary.append(f"  {metric}: {value:.4f}")
            summary.append("")
        
        # 过拟合分析
        if report.overfitting_analysis:
            overfitting_score = report.overfitting_analysis.get('overfitting_score', 0)
            summary.append(f"过拟合风险: {overfitting_score:.2f}")
            summary.append(f"是否过拟合: {'是' if report.overfitting_analysis.get('is_overfitting', False) else '否'}")
            summary.append("")
        
        # 稳定性分析
        if report.stability_analysis:
            stability = report.stability_analysis.get('overall_stability', 0)
            summary.append(f"策略稳定性: {stability:.2f}")
            summary.append(f"是否稳定: {'是' if report.stability_analysis.get('is_stable', False) else '否'}")
            summary.append("")
        
        # 建议
        if report.recommendations:
            summary.append("改进建议:")
            for i, rec in enumerate(report.recommendations, 1):
                summary.append(f"  {i}. {rec}")
            summary.append("")
        
        return "\n".join(summary)