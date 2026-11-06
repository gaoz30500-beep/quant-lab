# research/ — 未接入主链路的实验模块

这里的代码**不在主链路上**（`main.py` / `engine.py` / `portfolio_engine.py` /
`fund_rules_backtest.py` 均不引用），是历史探索与实验性设计，保留供参考。

| 文件 | 内容 | 状态 |
|---|---|---|
| `engine_optimized.py` | 并行回测引擎原型 | 实验；并行分块与持仓状态合并的正确性未验证 |
| `risk_enhanced.py` | 动态风控/组合VaR/熔断原型 | 实验；VaR 用固定波动率假设，未经校准 |
| `data_manager.py` | SQLite 元数据/缓存管理原型 | 实验；与 `data_source.py` 的 CSV 缓存功能重叠 |
| `strategy_validator.py` | 过拟合检测/蒙特卡洛框架 | 实验；内部 `_run_backtest` 是随机数占位，未接真实引擎 |
| `trading_system.py` | 实盘订单管理/交易接口原型 | 实验；仅有 Mock 接口，未接任何券商通道 |
| `monitoring_system.py` | 实时监控/报告原型 | 实验；依赖 plotly/seaborn/statsmodels |
| `fund_analyzer.py` `fund_backtest.py` `analyze_*.py` | 基金分析探索脚本 | 已废弃；正式版本是根目录 `fund_rules_backtest.py` / `fund_batch.py` |

## 约定

- **不要**从主链路 import 这里的模块；如需其中能力，先移植 + 补测试再接入
- 运行这里的脚本需从仓库根目录执行并设置 `PYTHONPATH=.`（部分脚本的
  `sys.path` 按旧目录层级写死）
- 额外依赖见根目录 `requirements.txt` 注释，或 `pip install -e ".[research]"`
