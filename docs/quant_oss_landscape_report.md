# 加密量化交易系统（Binance USDT-M 永续 · 纸面模拟 + ML 决策）开源横向调研报告

> **置信度声明（重要）**：本会话网络对 github.com 可达（`curl -I https://github.com` 返回 HTTP/1.1 200 OK，已核验），但**逐项实时核验（star 数、当前 API/文档细节）因工具调用中断而未完成**。下表 star 数为**量级估计**，链接为官方仓库地址，落地前请逐项复核官方文档。未标注"已核验"的内容来自既有领域知识，属**中等置信度**。

---

## 一、项目速览（star 为量级估计，未核验）

| 项目 | star | 核心设计 | 最值得抄的一点 |
|---|---|---|---|
| [freqtrade](https://github.com/freqtrade/freqtrade) | ~40k | `IStrategy` 接口（`populate_indicators/entry/exit` + `custom_stoploss/custom_exit/confirm_trade_entry`）、hyperopt、protections、链式 pairlist、`FreqAI` 自适应 ML | 与本系统形态最接近的完整参照物（含逐币种 ML + 期货资金费） |
| [Hummingbot](https://github.com/hummingbot/hummingbot) | ~12k | 连接器抽象 `ConnectorBase/PerpetualDerivativePyBase`、`InFlightOrder` 订单状态机、`BudgetChecker`、`RateOracle`、`PositionExecutor`（三屏障） | **订单生命周期状态机 + 速率预算器** |
| [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) | ~7k | Rust 核心 + Python 策略；`MessageBus/DataEngine/ExecutionEngine/RiskEngine` 分引擎；回测与实盘同一 `Strategy` 代码路径 | **执行/Risk Engine 分层的"回测-实盘同构"** |
| [Jesse](https://github.com/jesse-ai/jesse) | ~6k | 简化策略钩子 + Jupyter 研究模块 + GA 优化 + live | "诚实简单的蜡烛级回测 + 研究环境"的取舍范式 |
| [OctoBot](https://github.com/Drakkar-Software/OctoBot) | ~3.5k | 插件（tentacles）+ profile 配置 + `TraderSimulator` 与真实 trader 同基类 | 模拟器与实盘交易器**同一接口** |
| [Zenbot](https://github.com/DeviaVir/zenbot) / [Gekko](https://github.com/askmike/gekko) | ~8k / ~8k | 均已停维护 | **反面教材**：蜡烛收盘价成交、无订单簿、无永续/资金费、无状态对账 |
| [vnpy](https://github.com/vnpy/vnpy) | ~28k | `EventEngine` + `BaseGateway` + 应用层（CtaStrategy/PortfolioStrategy/**SpreadTrading**/AlgoTrading/RiskManager）；tick 级回测 | **网关抽象 + 风控规则插件 + 价差(基差)交易应用** |
| [WonderTrader](https://github.com/wondertrader/wondertrader) | ~5k | C++ 内核（Cta/Sel/Hft/Exec/Uft 引擎）+ `wtpy`；本地撮合器与仿真交易器独立于策略 | 毫秒级 tick 撮合 + 仿真交易器解耦 |
| [hftbacktest](https://github.com/nkaz001/hftbacktest) | ~2.5k | L2/L3 订单簿重放、**队列位置模型**、**分离的行情/下单延迟**、maker/taker 费率、Binance 永续数据采集器 | **撮合真实性天花板（本报告最强借鉴源）** |
| [ccxt](https://github.com/ccxt/ccxt) | ~35k | 100+ 交易所统一 REST/WS（含 `fetchFundingRate`/`fetchOpenInterest`/`watchOrders`） | 广度换取"最低公约数"的取舍清单 |
| [FinRL](https://github.com/AI4Finance-Foundation/FinRL) / [FinRL-Meta](https://github.com/AI4Finance-Foundation/FinRL-Meta) | ~11k / ~2k | 环境-数据-智能体三层解耦；`DataProcessor` + **turbulence index**（崩盘避险） | 数据层与 RL 智能体解耦、turbulence 风控因子 |
| [TradeMaster](https://github.com/TradeMaster-NTU/TradeMaster) | ~2k | 任务分类（alpha/组合/执行/做市/HFT）+ 统一 `Evaluator` | 统一评估器（ARR/SR/MDD/CR/SoR） |
| [Qlib](https://github.com/microsoft/qlib) | ~26k | `Expression` 因子引擎、**Alpha158/Alpha360**、`RollingGen` 走步训练、`risk_analysis` | **可直接复用的因子库 + 走步(walk-forward)训练流水线** |
| [LEAN](https://github.com/QuantConnect/Lean) | ~11k | **Fill/Fee/Slippage/Margin/BuyingPower/Settlement 六类可插拔模型**；Alpha→组合构建→风控→执行框架 | **撮合真实性"模型分解"最佳架构参考** |
| [vectorbt](https://github.com/polakowo/vectorbt) / [backtrader](https://github.com/mementum/backtrader) / [quantstats](https://github.com/ranaroussi/quantstats) | ~5k / ~15k / ~5.5k | 向量化参数扫描 + `Splitter` 走步切分 / Broker 抽象 / 全套绩效报告 | 参数网格热力图 + `probabilistic_sharpe_ratio` |
| [cryptofeed](https://github.com/bmoscon/cryptofeed) | ~2.3k | 归一化 L2/成交/**资金费/持仓量/强平**流 | 强平瀑布与 OI 数据源 |

---

## 二、A–F 六问回答

### A. 撮合引擎真实性：分级与差距

**分级（从高到真）**：hftbacktest（L2/L3 队列位置 + 行情/下单双向延迟 + maker/taker 费 + tick/lot 约束）＞ NautilusTrader（L2/L3 簿、FillModel 概率成交、延迟模型、OCO/OUO 条件单、保证金账户与资金费）≈ LEAN（模型分解完备，但默认 bar 驱动）＞ WonderTrader/vnpy（tick 级撮合 + 滑点/费率/pricetick，但无 L2 队列）＞ Hummingbot paper trade（用真实订单簿模拟成交 + BudgetChecker，非队列感知）＞ freqtrade（**蜡烛级**：次根开盘成交、`custom_slippage`、期货模式计入资金费；**无部分成交、无队列、无强平**）＞ Jesse/backtrader/vectorbt（蜡烛级，偏乐观）。

**本系统该补的（按 ROI 排序）**：① 行情延迟与下单延迟**分离建模**（hftbacktest 的 `feed_latency`/`order_entry_latency` 语义）；② 限价单**队列位置模型**（RiskAverse/Probability 队列模型），至少用盘口前 N 档快照校准；③ **部分成交 + IOC/FOK/GTD/post-only/reduce-only**；④ 每 8 小时按**实际资金费率**结算；⑤ **标记价 + 分层维持保证金率**触发强平（而非最新价），并计强平手续费；⑥ maker/taker 分档费率。当前"含保证金与强平的纸面撮合"若不区分标记价与最新价、不建模延迟，会系统性高估高频/挂单类策略收益。

### B. 回测-实盘代码同构：四种可抄模式

1. **同策略 + 可插拔客户端**（Nautilus）：`Strategy` 只发订单意图，回测用 `BacktestDataClient/ExecutionClient`，实盘换成 Binance 客户端，策略代码零改动。
2. **模型分解**（LEAN）：把成交、费率、滑点、保证金、购买力、结算拆成 6 个可替换类——这是本系统纸面撮合最该重构的方向（目前大概率是单文件 if-else）。
3. **网关接口**（vnpy `BaseGateway` / WonderTrader 仿真交易器 / OctoBot `TraderSimulator`）：定义 `ExecutionGateway` 抽象，`PaperGateway` 与 `LiveGateway` 同签名实现。
4. **共享订单状态机 + 对账**（Hummingbot `InFlightOrder` + ccxt.pro `watchOrders/watchPositions`）：订单 FSM 与幂等 `clientOrderId` 是纸面转实盘不炸的关键；SQLite 只做缓存，**定期与交易所对账**才是真相来源。

### C. 参数搜索与防过拟合

- **hyperopt + 损失函数 + protections**（freqtrade）：`--spaces buy sell roi stoploss trailing protection`，损失函数用 `SortinoHyperOptLoss/CalmarHyperOptLoss/MaxDrawdownHyperOptLoss`，支持 Optuna 采样器。
- **走步滚动重训**（FreqAI：`train_period_days` + `backtest_period_days`；Qlib `RollingGen`）：模型在每个 out-of-sample 窗口重训，杜绝"全样本训练"。
- **重叠标签的 purged CV + embargo**（López de Prado）：本系统的 meta-labeling 逐品种门槛若用三屏障标签，**标签区间高度重叠**，必须用 PurgedKFold + embargo，否则 CV 分数虚高。
- **多重检验校正**：`Deflated Sharpe Ratio`、**CSCV/PBO（回测过拟合概率）**、quantstats 的 `probabilistic_sharpe_ratio`；vectorbt `Splitter.rolling_split` 做参数热力图看"参数高原 vs 尖峰"。
- 工程纪律：记录**试验次数**并据此扣减显著性；保留独立 holdout 时间段。

### D. 加密特有：谁做得最好

- **资金费/永续**：freqtrade（期货模式回测计资金费）、Hummingbot（永续连接器 + 资金费入 PnL）、Nautilus（`FundingRateUpdate` + 保证金账户）最成熟。
- **强平瀑布**：[cryptofeed](https://github.com/bmoscon/cryptofeed) 的 `Liquidations` 通道 + Binance `!forceOrder@arr` 流是标准数据源；hftbacktest 可重放强平造成的深度真空与跳空。
- **基差/价差**：**vnpy `SpreadTrading` 应用**是唯一成熟的价差（含跨期/基差）交易框架，可直接参照其价差合约与腿管理设计。
- 结论：**资金费记账学 freqtrade，强平数据学 cryptofeed，队列真实性学 hftbacktest，基差组合学 vnpy**。

### E. 风控与组合层

- **LEAN**：Alpha→组合构建（`IPortfolioConstructionModel`）→风控（`MaximumDrawdownPercentPortfolio`、`TrailingStopRiskManagementModel`）→执行，四段式流水线最完整。
- **vnpy RiskManager**：规则插件化（`MaxOrderSizeRule`/`ActiveOrderRule`/`DailyLimitRule`/`CancelOrderRule`），跟踪"流控/活动委托数"。
- **Nautilus `RiskEngine`**：把风控做成消息流中的一等引擎（下单速率、最大名义、reduce-only 强制、余额校验）。
- **freqtrade protections**：`StoplossGuard`/`MaxDrawdown`/`LowProfitPairs`/`CooldownPeriod`（`lookback_period`+`trade_limit`+`stop_duration`+`only_per_pair`）——**最易移植到逐品种门槛体系**。
- **Hummingbot** `BudgetChecker` + `RateOracle`（按 limit_id 的令牌桶预算）。
- **FinRL turbulence index**：市场级风险指数触发"清仓避险"，比单品种风控更早感知系统性回撤。

### F. 可直接复用的特征/因子库

1. **Qlib Alpha158 / Alpha360**（`qlib.contrib.data.handler`，表达式引擎 `$close/Ref($close,1)-1`）——可直接在 1m/5m K 线上重建。
2. **FinRL-Meta `DataProcessor`**（`add_technical_indicator`、`add_turbulence`、`df_to_array`）。
3. **mlfinlab 系列**（微结构特征：Kyle lambda、Amihud lambda、Roll、Corwin-Schultz；分数阶差分）+ `timeseriescv` 的 `PurgedKFold`。
4. freqtrade `FreqAI` 的特征工程与 `DissimilarityIndex` 离群过滤；`TA-Lib`/`pandas-ta`；本系统已用的 Chronos-2 属 Nixtla/Amazon 系零样本预测器，可继续保留。

---

## 三、本系统最该借鉴的 10 条具体改进

| # | 来源 | 具体做法 | 本系统落地 |
|---|---|---|---|
| 1 | LEAN | Fill/Fee/Slippage/Margin/Latency 拆成可插拔模型类 | 把纸面撮合重构为 `execution/models/*.py`，Binance 分层 MMR 表做成配置 |
| 2 | NautilusTrader | 策略只发意图，`ExecutionClient` 可换（回测/实盘同一策略） | 抽 `ExecutionGateway` 接口，`PaperGateway`/`BinanceGateway` 双实现，ML 决策层不动 |
| 3 | hftbacktest | 行情延迟与下单延迟分离 + 队列位置模型 | 采集盘口前 N 档到 SQLite/Parquet，限价单按队列模型判定成交 |
| 4 | Hummingbot | `InFlightOrder` 状态机 + 幂等 clientOrderId | 建 `OrderState` FSM（NEW/PARTIAL/FILLED/CANCELED/REJECTED/EXPIRED），重启可续 |
| 5 | ccxt.pro + vnpy | `watchOrders/watchPositions` 对账 + `DataRecorder` | 每 N 秒拉取订单/持仓/余额与本地 SQLite 对账，差异告警并自动纠偏 |
| 6 | freqtrade（期货模式） | 8 小时资金费结算用实际费率 | 在纸面 PnL 中按结算时刻计资金费，并作为特征入模 |
| 7 | FreqAI + Qlib RollingGen | `train_period_days`/`backtest_period_days` 走步重训 | 用滚动窗口替掉当前全样本训练；LightGBM/CatBoost 每窗重拟 |
| 8 | López de Prado 系（PurgedKFold/embargo + DSR + PBO） | 重叠标签净化 + 多重检验扣减 | meta-labeling 的逐品种门槛必须用 purged CV；记录试验次数算 Deflated Sharpe，PBO 超阈值禁止模型上线 |
| 9 | freqtrade protections + vnpy RiskManager | `StoplossGuard/MaxDrawdown/LowProfitPairs/Cooldown` + 下单前规则校验 | 新增 `ProtectionManager`（对某品种熔断 N 小时）+ `RiskEngine`（最大名义/活动委托数/日内亏损上限/断连撤单） |
| 10 | Qlib Alpha158 + LEAN PCM + quantstats | 因子库 + 组合构建层 + 标准化绩效报告 | 建 `alpha158_crypto`（叠加资金费/OI/基差特征）；加组合层（波动率目标/相关性上限/总杠杆上限）；每次回测自动出 quantstats 报告并入库做 run 对比 |

---

## 四、未完成部分（明确标注）

1. **实时核验未完成**：github.com 在本会话内可达（HTTP 200 已确认），但 GitHub API 拉取被中断，**所有 star 数为量级估计，未逐一核实**。
2. **未逐项精读官方文档**：vnpy `SpreadTrading`、WonderTrader `UftEngine`、hftbacktest 队列模型具体 API 参数、Nautilus 延迟模型配置项、LEAN 各 Fee/Margin 模型类名，均为既有知识的概括，**落地前必须复核官方文档**。
3. **未覆盖**：Binance 官方 `binance-futures-connector-python` 的能力边界、`mlfinlab` 商业授权限制（社区开源 fork 可用性）、TradeMaster 与 FinRL 在加密永续上的实际适配度。
4. 第 10 类"其他值得借鉴项目"仅覆盖 LEAN 与 cryptofeed，**未系统搜索**近两年的加密 ML/paper-trading 新项目（如需可发起第二轮检索）。
