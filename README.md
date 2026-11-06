# A 股量化回测框架 (quant_trading)

日线级别的多标组合回测：趋势跟踪（MA 交叉 + 大盘闸门）与均值回归两套策略，
配套参数网格、滚动验证、最低本金测算和场外基金规则回测。

## 环境

- Python ≥ 3.9（开发机为 3.11）
- 依赖：`pip install -r requirements.txt`，或可安装方式 `pip install -e .`（提供 `quant` 命令）
- 数据缓存目录 `data/cache/`、结果目录 `results/` 均为本地生成，不入库（见 `.gitignore`）
- 日志级别：环境变量 `QUANT_LOG_LEVEL=DEBUG`（默认 INFO；诊断走 stderr，报告走 stdout）

## 常用命令

```bat
python main.py portfolio --start 20180101 --mode trend      :: 组合回测 + 等权基准对比
python main.py portfolio --start 20180101 --mode mean_reversion
python production_gate.py                                   :: 生产验收闸门 (红灯=不可实盘)
python clean_cache.py                                        :: 清缓存 (如需强制重拉)
```

参数网格等脚本按需单独运行：

```bat
python optimize_trend.py       :: 趋势参数网格, 双段一致性 + 卡玛
python validate_trend.py       :: 4 段滚动验证
python optimize_steady.py      :: 风控参数扫描 (止盈/移动止损/持仓数)
python etf_portfolio.py        :: ETF 组合 + 小资金档位
python capital_scaling.py      :: 不同本金档位的真实收益
python fund_rules_backtest.py  :: 场外基金 6 条预注册规则
```

## 回测口径（这些约定比参数更重要）

| 项目 | 约定 |
| --- | --- |
| 成交时点 | T 日收盘产生信号 → T+1 开盘价成交；基金规则回测用 T+1 净值（QDII 为 T+2） |
| 涨跌停 | 开盘价触及涨停则买不进、触及跌停则卖不出；主板 10%，创业板/科创板 20%，ETF 无限制 |
| 停牌 | 当日无 K 线视为停牌，挂单顺延至复牌 |
| 交易日历 | 优先用上证指数日历（`index_` 缓存），缺失时退化为个股交易日并集 |
| 复权 | 统一**后复权（hfq）**。前复权会把历史价减成负数（茅台 2015 年前复权 -117.62），收益率失效 |
| 费用 | 佣金双边（默认万三、最低 5 元）、卖出印花税 0.1%、买入不收印花税、滑点按比例 |
| 现金 | 空仓资金按年化 2% 计息，逐日累乘 |
| 预热与评估 | `data` 可包含评估起点之前的历史，引擎用 `eval_start` 只在窗口内交易；信号在完整历史上计算 |
| 绩效 | 全部走 `metrics.Metrics` 一套公式（几何年化、夏普按日超额、索提诺用下行偏差） |
| 可复现 | 每次回测落地 `results/experiment_portfolio_*.json`（配置 + 数据指纹 + 评估窗口） |

### 关于 `eval_start`

"先切数据、再算信号"会让 MA150/MA200 在每段开头重新累积，段首数月被动空仓，
在下跌段等于自动躲过行情，稳定性被系统性高估。正确做法是：

```python
PortfolioEngine(data_with_history, factory, config,
                trading_days=calendar, eval_start='2018-01-01')
```

## 目录

| 文件 | 作用 |
| --- | --- |
| `main.py` | CLI 入口 |
| `config.py` | 回测配置（费率、风控、现金收益） |
| `data_source.py` | 数据层：akshare 拉取 + 本地缓存 + 新鲜度判定 + 脏数据修正 |
| `engine.py` | 单标的引擎 + `Trade`/`BacktestResult` + 涨跌停判定 |
| `portfolio_engine.py` | 多标的共享资金引擎（主链路实际使用） |
| `strategies/` | 策略（MA 交叉、组合信号、闸门包装） |
| `market_gate.py` | 大盘闸门（指数 MA + 动量过滤） |
| `risk.py` | 仓位与风控 |
| `metrics.py` | 绩效指标 |
| `portfolio.py` / `optimize_*.py` / `validate_trend.py` / `etf_portfolio.py` / `capital_scaling.py` / `fund_rules_backtest.py` | 回测与研究报告脚本 |
| `logging_setup.py` | 统一分级日志（入口调 `setup_logging()`，模块用 `get_logger(__name__)`） |
| `production_gate.py` | 生产验收闸门：可信/稳/回报 8 项硬指标，红灯禁止实盘 |
| `clean_cache.py` | 缓存清理工具（`--symbol` 过滤 / `--dry-run` 预览） |
| `tests/` | 合成数据 golden master + 规则单测 + 数据层边界测试（41 个，全离线） |
| `research/` | **未接入主链路**的实验模块（见其 README，勿从主链路 import） |
| `archive/` | 历史废弃脚本（仅存档，git 历史可追溯） |

`research/` 中 `engine_optimized.py` 按收盘价成交且没有涨跌停判断，**不要**把它当作
`engine.py` 的升级版；其余模块状态见 `research/README.md` 的逐文件标注。

## 测试

```bat
python -m unittest discover -s tests -v
```

测试用完全确定性的合成行情 + mock（全离线，CI 可直接跑），覆盖：

- 引擎口径：涨跌停判定、"破 MA60 走/回 MA20 进"的离场优先级、组合回测
  golden master（总收益/回撤/笔数/逐笔盈亏口径）、预热窗口与评估窗口分离
- 数据层边界：`validate_daily` 只报告不擅改、重试与指数退避、盘中半根 K 线丢弃、
  缓存新鲜度判定（无 stamp/过期/当日已查/stamp 损坏）
- 基金降级：`get_nav` 三级策略（新鲜缓存不触网 / 网络失败降级旧缓存 / 无缓存抛带代码的错）

CI：push/PR 触发 GitHub Actions（`.github/workflows/tests.yml`）。

## 已知限制

- ST/*ST 的 5% 涨跌幅限制未建模（需要股票名称数据，仅凭代码无法判定）。
- 分红送股只体现在复权价格里，没有单独的现金分红流水与再投资时点。
- 基准为等权买入持有，未做行业中性或市值加权基准。
- 参数网格仍是网格搜索 + 双段一致性，未做正式 walk-forward 再拟合。
