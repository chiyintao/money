# Binance Futures AI Research Agent

这是默认只做数据同步、预测与仿真交易的 Binance USDT-M 合约研究系统。默认不会发送真实订单。

## 快速开始

```bash
pip install -r requirements.txt
python -m app.main --mode paper
```

双击 `start_paper.bat` 启动纸面交易服务。脚本会清理旧的 8097 端口进程，在新窗口运行服务，等待接口就绪后自动打开浏览器。打开 http://127.0.0.1:8097 查看实时研究台；API 为 `/api/state`，存活探针为 `/health`，就绪探针为 `/ready`，Prometheus 指标为 `/metrics`。

`/health` 不再返回恒定的 `ok` 加五个计数——那些计数在真正需要告警的状态下全都是 0，一个死掉的行情源和一个健康的行情源看起来完全一样。现在它由 `app/ops/health.py` 给出十项带判定的检查（`market_data`/`feed`/`models`/`features`/`ood_gate`/`drift`/`reconciliation`/`venue`/`derivatives`/`components`）。新增的 `features` 回答的是最根本的那个问题——**“模型有没有拿到它训练时被喂的东西”**：`/health` 现在会报 `every_row_degraded` / `partially_degraded` / `collected_families_empty` / `all_inputs_measured`，并列出具体缺哪些列。此前这件事从未被检查过：占位符 0 落在训练边界之内，OOD 门禁看不见它，模型加载成功，所有聚合指标都合理。其中 `derivatives` 专门盯**唯一一份过期即永久丢失的数据**：交易所在滚动三十天窗口里发布未平仓量/多空比/taker 量比然后丢弃，没有存档也没有历史接口。采集器把逐品种的异常吞进 `stats['failed']`/`stats['last_error']` 后正常返回，调用点随即 `errors.ok(...)` 把该源标记为健康——于是曾出现 `/health` 答 `ok` 而 `/metrics` 的 `derivatives_failures_total` 在涨。现在这项检查读的是采集器内存里的同一份数字，两者不可能再各说各话：有失败 → `collection_failing`，跑过但零行 → `collection_empty`，超过 3 个采集周期未动 → `collection_overdue`。`status` 取其中**最差**的一项：`ok` / `degraded` / `down`。全部读取内存中已有的状态，不组装仪表盘 payload、不发网络请求、不查库，因此和原来的计数字段一样便宜（实测仍是毫秒级）。**HTTP 状态码始终 200**：容器编排器若因行情中断而重启进程，会在唯一一种"进程本身没问题、只是没法下单"的情况下反复重启。需要按状态码判断的探针请用 `/ready`，它在 `down` 时返回 503。使用 `stop_paper.bat` 停止 8097 端口服务。

## 当前能力

- 528 个左右 USDT 永续合约的实时市场快照
- 热门、涨幅、跌幅、新合约榜
- 最新价格、成交额、最高/最低价、成交笔数
- 标记价格、指数价格、最新资金费率、单币种持仓量接口
- 可重连的 Binance mark-price WebSocket 组件，REST 负责回补和校验
- 资金费率与持仓量历史 SQLite 表
- 闭合 K 线识别与 SQLite 幂等存储
- 模拟交易的每一次方向判断都来自真实已训练模型：LightGBM + CatBoost 集成，可选 Chronos-2 量化分位预测投票
- EMA/RSI/ATR 规则基线降级为显式可选回退（MODEL_RULE_FALLBACK，默认关闭）
- 含手续费和滑点的纸面撮合
- 含分档维持保证金率、强平费与穿仓记录的纸面撮合
- 订单/风险事件 SQLite 账本和账户状态恢复
- 挂单占用组合风险额度（挂单在成交前即计入 `max_positions`、单标/总杠杆与组合风险预算）
- 高水位回撤熔断、连亏减仓、波动率目标与相关性敞口上限
- 衍生品数据采集器（持仓量、大户/全局多空比、taker 买卖比），以及 K 线订单流字段（taker 买量、成交笔数、quote volume）
- 市场数据站点一致性自检（`VENUE_CHECK_POLICY`）
- 数据陈旧开仓拒绝与组合持仓限制
- WebSocket K线与 mark price 事件解析、REST K线回补
- 权益曲线、交易记录和基础绩效指标
- 训练数据集生成器
- 训练数据质量门禁、线性基线和 walk-forward 评估
- WebSocket 事件与重连状态监控
- 本地只读 Agent 上下文和结构化候选计划校验

## 代码分层

`app/` 原先是 79 个模块平铺在一个命名空间里，层与层之间唯一的隔离是“暂时还没人从错误的方向 import”。两个真实缺陷由此产生：行情层为了两个纯值对象 import 了下单层（`market` → `broker`/`margin`），服务层反向 import 了模型层的一个打分函数（`live_models` → `advanced_model`）。两处都在调用点用函数内 import 收场——这是把环藏起来而不是修掉的标准做法。

现在 `app/` 按依赖方向分成十层，**上层可以 import 下层，下层永远不能 import 上层**：

| 层 | 职责 | 允许 import |
|---|---|---|
| `core/` | 词汇与机制：配置、领域对象、订单状态机、合约规格、事件、指标 | **无**（不 import 任何 app 模块） |
| `storage/` | 持久化：SQLite、批量写入、对账、保留策略 | core |
| `market/` | 交易所：REST/WS 连接、行情归一化、资金费、衍生品采集 | core, storage |
| `features/` | 特征契约：训练与线上逐字共用的同一个定义 | core, storage, market |
| `trading/` | 订单机制：撮合、成交模型、保证金、风控、仓位、执行规则 | 以上各层 |
| `backtest/` | 回放与账户状态投影 | 以上各层 |
| `models/` | 拟合、评估、模型产物与晋升 | 以上各层 |
| `strategy/` | 模型输出 → 交易决策 | 以上各层 |
| `ops/`, `web/` | 可观测性与只读视图 | 以上各层 |

`app/main.py` 与 `app/runtime_context.py` 是组合根，负责装配，因此豁免。每个包的 `__init__.py` 里都写了该层的契约。

契约不是文档，是可执行的。`tests/test_architecture.py` 检查四件事，因为它们失败的方式不同：

- **模块级 import 了更高层**：真正的分层违规
- **函数内 import 了更高层**：隐藏的违规。它在导入期不报错，因此能活过评审
- **即使每条边都朝下仍构成环**：照样是解不开的死结
- **`core/` import 了任何 app 模块**：所有层都可以 import core，所以 core 谁都不能 import

延迟 import 的总数另有一条上限（当前 30，实测 25）。每一个延迟 import 都是一个“等着被写出来的环”，上限存在的意义是让它增长时必须有人解释，而不是顺手加一个。

## 真实模型接入

纸面模式下，tick 决策与闭合 K 线决策都走同一条真实模型链路（`app/strategy/live_models.py`）：

```text
闭合K线 -> **features-v4（30 个特征，四个族）** -> 真实模型集成（LightGBM + CatBoost [+ Chronos-2]）
     -> 方向一致性 + 最小边际过滤 -> 风控/保证金/盘口新鲜度 -> 纸面撮合
```

- 权重来源优先级：注册表已晋级的生产模型（含 isotonic 校准）→ `data/research_v3/candidates` 中最新的真实 LightGBM / CatBoost 产物 → 无可用权重时不下单。
- **实测量：第一优先级从未生效过，当前跑的是第二优先级。** `data/models` 为空，`ModelRegistry('data/models').current()` 返回 `None`，`load_calibrated_model` 返回 `fallback / no_production_model`，所以 `ProductionMember` 从未被构造。原因是晋级门禁要求的每一种证据都没有生产者：校准器没有任何代码会去拟合它，组合样本外证据没有回放来源，而且 `fit_advanced` 把 `feature_version` 写成常量之外的字面量 `features-v1`（代码早已是 `features-v3`），产出的 artifact 会被自己的门禁拒掉。上述生产者本轮已补齐，端到端路径由 `tests/test_calibration_pipeline.py::test_a_real_model_with_sufficient_evidence_is_promoted_and_then_loads` 覆盖；因为 `MODEL_REQUIRE_PROMOTED=0`，这件事此前是静默发生的。
- **实测量：当前加载的两个候选，留出集证据说自己没有方向边缘。** `data/research_v3/candidates/` 的 lightgbm 与 catboost manifest：188,640 行留出样本上方向准确率 **48.238% / 48.244%**（低于 50%）；逐品种 12 个品种**全部低于 50%**（lightgbm 46.15–49.51%，catboost 46.25–49.46%）。正期望只来自模型"愿意下手"的极小样本：269 个（占比 0.143%，命中 53.53%，净边缘 +2.08 bp）与 480 个（占比 0.254%，命中 57.29%，净边缘 +37.87 bp）；manifest 自己带着 `evaluation: independent_overlapping_samples_not_portfolio` 的注记，即这些样本时间上重叠、不是组合结果。系统自身重训管线的两次真实运行给出同向结论：4h/11 折/24,100 笔样本外成交，边缘 **-41.11 bp**、P(edge>0)=0.07、亏损折 7 : 盈利折 4；5m/21,984 笔，边缘 **-5.07 bp**、P=0.27、8 : 3。因此 `MODEL_MIN_EDGE_BPS`/`MODEL_MIN_AGREEMENT` 调参只能改变"在什么条件下采纳这个模型的方向"，改变不了这些数字本身。
- `/api/models` 现在返回这些字段与逐品种离散度（`test.active_net_edge_bps`/`active_share_pct`/`active_hit_rate_pct`/`evaluation`、`by_symbol: {symbols,min,median,max,below_coin_flip}`）。**准确率是一个没有边缘的模型也能通过的指标**，只展示它等于把可用信息丢掉一半。
- **features-v4：把“一直在采集、从来没有建模”的 20 个特征接上。** `app/features/order_flow.py` 早就能从已有数据算出三个族——K 线的 taker 失衡/成交笔数/报价额、未平仓量与多空持仓比、每分钟的主动方与强平桶——而**这些函数在 `app/` 与 `scripts/` 下的生产调用点数为 0**，只被自己的单测调用。采集是通的，特征是断的，和“标记价被取回来、存下来、从不用来算基差”是同一类缺陷。现在 `FeatureSource` 在同一个 `snapshot()` 里合并这四个族，`FEATURE_SETS` 按版本声明（`features-v3` → 10 列、`features-v4` → 30 列），而**服务侧按 artifact 自己声明的列取数**（`TabularPredictor` 读 manifest 的 `features`、`ModelDecision.required_features()` 取各成员声明的并集），所以加特征不再等于让所有已加载模型同时降级。诊断计数（`order_flow_bars`/`positioning_rows`/`flow_buckets`/`trades`/`measured`）**刻意不是特征**：它们描述“有多少数据可用”而不是“市场做了什么”，在一个采集正常的训练集里它们是常数，正好是 OOD 门禁看不见的那种退化边界。
- **`oi_price_quadrant` 的价格腿此前永远是 0.0。** `positioning_features` 从 `derivatives_detail` 行的 `price_change` 列读价格方向——而持仓量端点只返回未平仓量和账户多空比、**不返回任何价格**，schema 里也没有这一列。于是模块注释里写着“符号对就是信号的全部内容”的那个特征，在系统建过的每一行上都是 0.0。现在由持有 K 线的调用方传入，且与配对的 `oi_change_5m` 用同一个一步窗口。
- **`Store.flow` / `Store.derivatives_detail` 的 `since` 不是窗口。** 语句以 `ORDER BY event_time DESC LIMIT n` 结尾，只给下界时，在行数超过 n 的表上返回的是**最新的 n 行**。实测：对 1500 根 K 线里第 49 根那个时刻，`since=bar-1h, limit=240` 返回的 240 行里最早的一行在第 1211 根之后（约 4.2 天），可见行为 0 行；补上 `until` 后返回 49 行、可见 13 行。两端的界都要有。
- **同一个 bar 的特征值不能取决于缓存块对齐。** 块缓存是必需的（150,000 行 × 3 次查询不可接受），但第一版只按“不晚于 bar_time”过滤，同一个 bar 在块开头会看到 47 个 flow 桶、下一根只看到 13 个。现在窗口两端都有界（`_visible`），值只由 `(span, bar_time)` 决定。
- **训练集构建的实测吞吐**：`reuse=True`（离线回放，store 不会被写）把每个品种 150k 行的构建从 313 行/秒提到 5,136 行/秒（0.5 分钟/品种），退化列从 15 个降到 0 个。构建器现在按**列名**而不是一个总数报告退化（`degraded_features`），因为“970 行是退化的”不说明该重建资金费历史还是等采集器。
- **块缓存的向前延伸只对闭合历史成立。** 为了让缓存真的省下查询，块按锚点向后取 4 倍跨度。离线回放里这是对的（那些行当时就存在，是同一次查询取回来的）；实盘里 12:00 取回的块被缓存到 15:00，里面没有 13:00 的行，`_visible` 会从“存在的那少数陈旧桶”里取值——**得到一个错的数，而不是一个缺失的值**，后者会被上报、前者不会。实测同一段 80 根 bar 的行走：实盘语义下与全历史读取不一致的 bar 数 = 0，向前延伸下 = 52（第 61 根起）。现在 `reuse` 同时决定这件事。
- **`reuse=True` 的语义是“这个 store 在使用期间不会被写”。** 它同时关掉 ttl 复用和限制块向前延伸的反面：离线回放两个都用，实盘两个都不用。
- **`ood_gate` 此前对“它没看过的列”报 ok。** 线上实测 `informative, checked: 10`——但 `data/research_v3/training_dataset.jsonl` 只有 10 列而契约有 30 列，**另外 20 列没有任何边界**，`out_of_range` 无从测试它们，而 `checked: 10` 读起来像完备的答案。现在无边界集合会与已加载权重实际声明的列取交集：只有“有模型在读这一列、却没有任何东西能界定它”时才降级。当前实测状态是 `degraded / unreachable_for:mark_basis`（`mark_basis` 的边界仍等于截断范围，第 2 轮重跑 profile 后从 4 条降到 1 条），以及 profile 覆盖 10 列 vs 契约 30 列。两者都要重建数据集后重跑 `scripts/profile_dataset.py`。
- **即时点 universe 此前是算出来、存下来、打印出来，然后丢掉。** `app/features/universe_history.py` 是认真写的模块（文件头明确写着要解决幸存者偏差与前视偏差，13 个测试，三态报告），`TrainingJob._point_in_time` 调用它、把结果存进 `job.point_in_time`、打进日志、写进 `training_runs`——**然后没有任何一行代码用它做决定**：紧随其后的 `dataset` 阶段仍然用 `universe` 阶段从**今天的快照**里选出的 `symbols`。去接线时先量了一下它的输出，发现它读到的数据本身就是错的：`Store.candles_range` 的默认 `LIMIT 500000` 配 `ORDER BY open_time ASC`，超出时**丢掉最新的部分**。实测 14 个品种、366 天、5 分钟共 **1,391,212 行**（7.7 秒 / 峰值分配 1.64 GB），其中 **891,212 行（64%）被静默丢弃**，重建拿到的 36% 恰好是最旧的一段，于是它把「24 小时滚动窗口」量在数据末端之前 **234 天**的地方，再报告说整个 universe 都停止交易了。三处修复：`candles_range` 改为 keyset 分页到底、`limit` 默认 `None`、超预算时**抛 `candle_range_over_limit`** 而不是返回一个看起来完整的短答案；新增 `Store.candle_windows` 只读「每个时刻往前 24 小时」加首末 bar（重建真正要问的两件事），实测读全量 1.31 秒、horizon 与最新 K 线一致；末端状态改为在**数据终点**而非墙上时钟评估（训练档 K 线只在训练运行时刷新，在 `now` 处重建会把「我们的采集停了」报成「市场停了」）。顺带把三种被折叠成同一种「不在 universe 里」的情况拆开（`no_data` / `not_listed_yet` / `no_bars_in_window`），并把 `measurable`（跑出结论了吗）与 `tradeable`（有东西可交易吗）分成两个字段——空 counts 是前者失败，全 excluded 是后者失败，调用方的反应相反。新增 `TrainingJob._restrict` 把结论接到行上，四种拒绝各有名字；写测试时抓到它自己的一个 bug：`graded` 含被排除的品种，当白名单用会让「全体被排除」一个都不裁却报 `ok`。线上实测 `days=365` 现在给出 `diagnosis=request_reaches_before_the_store`（库里最早的 bar 就在取数边界上，年龄门槛量的是我们的数据边界），因此**不裁剪**，并且明说原因——假装裁过了比不裁更糟。
- **278 万根 K 线的 `trades` / `quote_volume` / `taker_buy_volume` 是 NULL，不是 0。** 这条把一个一直读作「市场判断」的东西翻了过来：即时点重建里每个品种都倒在投机档的第三条判据上，报告写的是 `too few trades to fill`——而实测 `SELECT SUM(trades) FROM candles` 返回 **NULL**（不是 0），2,781,239 行**全部**如此，写入路径本身没问题（新建库写一行三个值都正确落库），是这批行的 INSERT 里根本没有这三列。于是 `symbol_facts` 求和得 0，`0 < 20000` 成立，**一个从没被记录过的数被当成了一次测量**。同一个 NULL 让 `order_flow_features` 的 `usable` 判据（`trades > 0`）为空：线上实测 v4 契约 **30 个特征里 21 个是缺的**（20 个订单流/持仓/flow 列加 `mark_basis`），只有 14 个真正有值。五个环节原来都把「缺失」压成 0，现在逐环改成可表示：`ingest._number/_count` 短数组返回 `None`、`storage._optional_float/_optional_int` 写 NULL、`symbol_facts` 的 `trades` 可为 `None` 并附 `trades_measured`、`classify_one` 报 `unmeasured: trade count` 而不是 `too few trades to fill`、`TrainingJob` 新增 `diagnosis=criteria_unmeasured` 与 `unmeasured_criteria`。原注释写「零成交笔数把它标记为未知」，这句话不成立——一根真正安静的 K 线**也是**零笔，差别在有没有记录。顺带把拒绝变成可执行的：新增 `max_window_days`（线上 speculative 档 = 274 天，低于默认的 365；mainstream 档 = **-365**，即该档在当前库上不可能被建立），`_restrict` 的理由也沿用重建的诊断——被一列空的排除和被年龄门槛排除，结果一样、下一步相反（重新灌数据 vs 缩短窗口）。
- 两个模型族格式不同，不要互换：注册表存的是 `gradient_boosted_stumps`（JSON，`predict_advanced` 要求 `target=gross_return` 且有 `base_score`），candidates 目录存的是 LightGBM/CatBoost booster。tabular 成员**没有校准器**，所以它返回的 `probability_up` 是 `None` 而不是把预测收益线性拉伸成的一个数——后者落在 (0,1) 内，但不是任何东西的概率。
- 每个信号的 `probability_up` 是各成员已校准概率的均值，同时给出 `calibrated_members` / `uncalibrated_members`。它**只上报、不设门禁**：实盘门禁是一致性下限与边缘下限，把概率加成第三道门会改变成交集合，阈值需要来自 `fit_directional` 在 holdout 上算出的可靠性曲线。当前线上该字段恒为 `None`，`calibrated_members` 恒为 0。
- 让 `data/models` 非空的命令是 `python -m app.models.advanced_model data/research_v3/training_dataset.jsonl --register`（需要传 `store` 才能产出组合证据；`--no-store` 会跳过回放并因此被门禁拒绝）。**顺序是先重建数据集再训练**，见下方 `mark_basis` 一节。
- 推理永不阻塞事件循环：表格模型走 `model-fast` 线程池，Chronos-2 走独立线程池；两者共用同一个 (symbol, 收盘 K 线) 缓存键，Chronos 结果落地后会失效对应信号缓存，下一次 tick 即带上该投票。
- 训练特征域校验：加载权重时用 manifest 里的 `dataset_sha256` 校验 `data/research_v3/training_dataset.jsonl`，并用该数据集的分位数边界判断输入是否越界。边界取每特征观测分布的 0.1% / 99.9% 分位（直方图流式统计），**不是 min/max**：训练数据写入前已被 `FEATURE_CLIP` 截断，取 min/max 会让任何触及过截断值的特征得到一条等于截断范围的边界，而服务侧输入又被同一套截断夹回该范围——这道门禁因此**永远无法触发**。实测数据集里 `return_10`/`funding_z`/`mark_basis`/`rsi` 四条正是如此。`MODEL_OOD_POLICY=block`（默认）拦截越界方向单，`warn` 只标记；`/health` 的 `ood_gate` 检查（细节在 `/api/models` 的 `feature_space`）会明确指出哪些边界仍然无效。**数据集 profile 已按新算法重新生成**（`python scripts/profile_dataset.py`，578.6 MB / 1,258,344 行约 24 秒）：退化边界从 4 条（`return_10`/`funding_z`/`mark_basis`/`rsi`，边界恰好等于截断范围）降到 1 条，`return_10` 的门禁从 ±0.5 收紧到 ±0.039。
- **`mark_basis` 的定义此前是错的**。标记价与资金费共用同一个时间轴和同一个二分下标，而资金费 8 小时才结算一次，于是 `mark/price - 1` 算出来的是"距上次结算以来的涨跌幅"而不是基差。实测：1,258,344 行训练数据里有 **122,201 行（9.711%）`mark_basis` 压在 ±2% 截断值上**，其余任何特征最高只有 0.046%（211 倍差距）——十分之一的数据把一个常数喂给了模型。修复分两部分：`mark_series()` 从 `derivatives_detail` 读 5 分钟粒度的标记价（`open_interest_value / open_interest` 即该桶均价，采集器一直在取、从未使用），资金费与标记价各走各的时间轴；`MAX_BASIS_AGE_MS=15min` 为硬上界，超龄或缺失时该特征返回 `None` 而非 `0.0`——0 落在训练边界内部，下游无法把它与"真实零基差"区分。
- 晋升门禁同时要求特征版本与训练时一致（此前写死比对 `features-v1`，而代码产出 `features-v3`，门禁不可达），并要求校准器已拟合。
- **组合样本外证据必须计入资金费**。实盘每个 tick 对每个持仓调用 `account.apply_funding`；回放此前只计入手续费与滑点，从不计资金费，而证据块里的 `costs_included` 是硬编码的 `True`——门禁读的第一个条件正是这个标志位。现在 `costs_included = bool(funding)`：没有费率数据时它是 `False`，附 `costs_missing: ["funding"]`，且 `total_funding` 字段被**移除**而不是留成 0.0（0.0 无法区分"净额为零"与"没有建模"）。费率从 `derivatives` 表读，结算窗口按品种传入（交易所有 8h/4h/1h 三种），方向由账户决定：多头付费率为正时付钱，空头收钱。实测同一组 fixture 只改费率符号，`net_return` 从 1.486209（付）变到 1.488171（收）。
- 每个决策都记录 `source`、`model_mode`、`model_version`、成员投票、`edge_bps`、`expected_net_return`、`probability_up`；`/api/models` 与“模型中心”页面展示同样的信息，权重路径与 SHA256 也会列出。
- 当前候选权重尚未通过注册表晋级门禁（其 manifest 在 12bp 成本下 `active_samples=0`），所以系统明确标注为“真实候选权重（未通过晋级门禁）”，不构成任何盈利结论。
- `MODEL_RULE_FALLBACK=0` 为默认值：真实模型不可用时系统拒绝交易（FLAT），不会静默用规则下单。

调参项：`MODEL_MIN_EDGE_BPS`（方向噪声下限，默认 0.5bp）、`MODEL_MIN_AGREEMENT`（成员一致性下限，默认 0.6）、`MODEL_OOD_POLICY`（越界输入 block/warn，默认 block）、`MODEL_CHRONOS_ENABLED`、`MODEL_MAX_CHRONOS_SYMBOLS`、`MODEL_DEVICE`。把 `MODEL_MIN_EDGE_BPS` 提到 12（等于训练时的成本阈值）即可只接受覆盖成本的方向；新建模拟会话时可选择“模型训练品种”，候选列表来自与所有已加载权重的数据集哈希匹配的训练数据（当前为 BTCUSDT、ETHUSDT）。这不会修改已有会话，也不保证当前输入仍处于训练范围；实时特征越界时仍会拦截。`SYMBOLS` 不会覆盖已有会话的选币结果。

离线回测默认仍用规则基线以便 A/B 对比，接入真实模型：

```bash
python -m app.backtest.backtest data/BTCUSDT_1m.jsonl --symbol BTCUSDT --models
```

## 优势门槛与逐品种过滤（meta-labeling）

复盘 85 笔真实成交后发现的核心问题：模型方向命中率约 50%，但**品种间差异极大**——同期有品种 15 分钟命中 64%，也有品种只有 33%。此前所有品种用同样的规则和仓位交易，亏损品种由盈利品种补贴。

这一层让**主模型提出方向，第二层决定这次是否值得下手**（Lopez de Prado 的 meta-labeling）：

- 每个候选预测都会被记录，并在 N 根 K 线后按**扣费后**收益结算
- 观测来自**信号而非成交**，因此被拒绝的品种仍在积累证据，可以自己挣回交易资格（这也是"证据不足时默认拒绝"安全的前提）
- 门槛按配置的持有期（默认 3 根 = 15 分钟）判定

```bash
curl http://127.0.0.1:8101/api/edge/symbols      # 逐品种战绩 + 启动时实测的优势曲线
python -m app.strategy.edge_report --min-samples 20       # 逐品种 × 逐持有期滚动验证表（P2）
```

**品种选择与门槛联动**：选币按涨幅排名，门槛按实测优势拒绝，两者若各自独立就会静默失配——实测出现过会话选中 5 个品种、全部被门槛拒绝、一笔未成交的情况。因此选择时优先取通过门槛的品种，并**保留 1 个槽位给证据不足的品种**（否则永远无法发现新品种）。

**止盈目标按实测优势衰减对齐**：实测优势在 6 根（30 分钟）最强、之后转负，而原几何瞄准数小时持有。启动时自动把离场时间止损对齐到实测最佳持有期（策略里已显式设置时不覆盖）。

## 市场数据站点一致性（最高优先级）

`.env` 里 `BINANCE_BASE_URL` 指向生产 REST，而 `BINANCE_WS_URL` 指向 `wss://fstream.binancefuture.com`——**这不是生产站点，是测试网**。两者从未被比较，因此服务用测试网盘口生成全部特征、K 线与纸面成交，却用生产站点取标记价、资金费与合约过滤器。没有任何报错：错误的价格仍然是价格。

20 秒 BTCUSDT `bookTicker` 实测：

| 站点 | 20s 消息数 | 中位买卖价差 | 中位买一量 | 中位卖一量 |
| --- | --- | --- | --- | --- |
| `fstream.binancefuture.com`（原配置） | 129 | 0.54 bps | **0.002 BTC** | **353.8 BTC** |
| `fstream.binance.com`（生产） | 10,645 | 0.01 bps | 2.156 BTC | 10.864 BTC |

更新频率差 82 倍，价差差 54 倍，且买一挂单只有约 156 美元、卖一却有约 2750 万美元——真实盘口不会如此单边。`app/trading/execution.py` 的深度成交模型正是读这两个数字判断"成交在盘口还是穿价"，因此买卖两侧的成交规则完全不同，**纸面交易结果不成立**。

现在启动时会做一次一致性自检（`app/market/venue_check.py`）：

- 静态部分永远执行：比较两个配置主机是否属于同一站点，混用则告警；
- 采样部分按 `VENUE_CHECK_POLICY` 决定：`warn`（默认）在启动后台执行，报告成交价差、更新频率、买一/卖一量失衡与相对 REST 价格的偏离；`block` 在监听端口前同步执行并在不一致时拒绝启动；`off` 跳过。
- 采样不到数据时状态是 `unknown` 而**不是** `ok`——正是"没什么可比较"才让原配置看起来一直正常。
- 结果进入 `/health` 的 `venue` 字段。

## 组合级风控（此前只有单笔风控）

原实现的每一道额度都是**单笔**的，因此看不见仓位之间的关系：

- **高水位回撤熔断**（`MAX_DRAWDOWN_FROM_PEAK`，默认 25%）：日内熔断在 UTC 换日时重置，因此"每天亏掉当日上限"是被允许的路径，账户按台阶下滑而没有任何一次拒绝。高水位熔断不会跨日自动解除，必须显式重置（同时重设 `peak_equity`）。
- **连亏减仓**（`LOSS_STREAK_LIMIT`=4、`LOSS_STREAK_SCALE`=0.5）：连续亏损达到阈值后风险预算按比例缩减，而不是停止交易——判定模型是否失效是另一套机制，需要新信号才能工作。
- **波动率目标**（`TARGET_BAR_VOLATILITY`，默认 0=关闭）：按 `target/realized` 缩放风险预算，**上限为 1**，即只减不增；把 `k` 设为 0.004 一类值时，只有波动率数倍于常态的品种会被缩减。
- **晋级门禁从「版本名相等」改成「输入契约相容」**：原判据只有 `manifest['feature_version'] != FEATURE_VERSION`，两个方向都盲。实测：一个 `feature_version` 填当前版本、但 `features` 里写了本代码根本算不出来的列名的模型**直接晋级**；反过来，一个完全可服务的旧版本产物仅因版本字符串不同就被拒——等于宣布特征契约永远不许扩展，而契约分版本的全部意义就是允许扩展。更糟的是服务侧 `required_features()` 取交集，声明全部落在契约之外时**返回空元组，而空的需求列表会把退化检查整个关掉**——那个「拒绝模型去读谁也填不上的输入」的门禁，被它本该拦下的输入关掉了，`/health` 的 `features` 检查还报 `all_inputs_measured`。现在 `register()` 把 `features` 写进 manifest（门禁想比也没得比是原问题的一半），`promote()` 同时判「列名本代码算不算得出来」（`unproducible_declared_features`）、「版本本代码还认不认」（`incompatible_feature_version`）、「声明的列是否在该版本契约内」（`features_outside_declared_version`）；服务侧记录 `unusable_features` 并由 `/health` 报 `degraded / declared_features_unproducible`。**「空声明」（输入清单未知，运行时保守地要契约全集）与「声明了算不出来的列」是两回事**，门禁只拒绝后者。
- **撮合层按 LEAN 的方式分解**（`app/trading/fill_models.py`）：原来 `broker.execute()` 一个函数里连着算价格、滑点、手续费，于是两个错误共用同一段代码、谁也暴露不了谁。**一、成交量完全不进成交价**：`slip = price * slippage_bps / 10000` 是常量，实测 0.001 与 10,000 BTCUSDT 都成交在 50,011.00（对 50,001 的卖一各滑 10.00），名义价值 50 与 5 亿——**一个假设自己能吃掉任何数量的账户报不出容量上限，因为它没有上限**。**二、每一笔成交都是 taker**：`Fill.liquidity` 早就存在，`execute_depth()` 把它硬编码成 `taker`、`execute()` 走默认值，于是挂单成交按 taker 费率收费（标准档 0.0400% vs maker 0.0200%，**收了一倍**），且因为字段写着 taker，下游无从发现。现在 `FeeModel(maker_rate, taker_rate)` 按流动性收费、`FillModel(spread_bps, impact_coefficient_bps, reference_notional)` 按**参与率**收冲击（平方根形式：吃掉整个盘口 → 系数全额，吃掉四分之一 → 一半），参与率未知时退回名义价值比，**两者都未知时按系数全额收费而不是假装这单很小——「不知道多大」不是「很小」**。挂单成交既不过价差也不计冲击（`slip_bps(maker) == 0`）；`liquidity` 与 `participation` 现在写进 `Fill`（此前标签算出来了、用它选了费率、然后在构造时丢掉了，**记录和它自己旁边的数字互相矛盾**）。
- **实盘的成交成本模型第一次可以配置**：`main.py` 的 `PaperBroker(...)` 此前不传 `fee_rate`/`slippage_bps`，而 `config.py` 里**根本没有对应配置项**——每一笔成交都用硬编码的 0.0400% taker + 2bp 定价，任何一段历史 session 都无法用它假定的成本来解释。新增 `TAKER_FEE_RATE` / `MAKER_FEE_RATE` / `SPREAD_BPS` / `IMPACT_BPS`（`IMPACT_BPS` 默认 0 = 与旧行为逐位一致，有测试钉住），`snapshot` 新增 `cost_model`。
- **订单转移表现在真的被应用了**：`order_state.transition()` 在仓库里**一次都没被调用过**——三处代码直接构造 `PaperOrder` 写状态（`broker.cancel`、`broker.process`、`execution._fill_from_depth`），于是转移表只是文档。更深一层：`PaperOrder` 是 **frozen dataclass**，而 `transition()` 的实现是 `order.status = status`，对**这个模块唯一为之存在的类型根本跑不通**（`FrozenInstanceError`）；它没被发现，正是因为它从没被调用。表本身也错：`PENDING` 只允许 `OPEN/REJECTED/CANCELED`，而 `PENDING` 在 `OPEN_STATUSES` 里、会被成交、`status_for_fill` 返回 `FILLED`/`PARTIALLY_FILLED`——**系统里最常见的那个转移是表里唯一没有的**。现在表已修正（`PENDING -> FILLED/PARTIALLY_FILLED` 必需：交易所不承诺先确认再成交），`transition()` 对 frozen dataclass 用 `dataclasses.replace` 重建并接受 `changes`，`PaperBroker.advance()` 是**唯一应用点**（`PaperBroker` 里只剩 1 处 `PaperOrder(`，即 `submit`）。被拒的转移计入 `transition_refusals`，进对账 `findings`（`order_transition_refused`）与 `paper_order_transition_refused{reason=...}`——成交路径上的拒绝是**账本与交易所的分歧**，不是统计量。
- **生产模型不可用时不再回落候选**：注册表这一侧一直是对的（`current()` 区分 `active`/`expired`，`load_production()` 对过期抛错），但 `ModelRuntime.load()` 的回退判据只有一个 `if not promoted`——于是「注册表里根本没有晋级模型」和「有晋级模型但它过期了/校准器丢了/读不出来」走同一条路，静默加载**晋级门禁拒绝过的**候选权重。从「已批准但过期」降级到「已被拒绝」是一次 provenance 降级，外面看只是 `mode=candidate`。现在 `classify_reason()` 把失败分成 `absent`（回落是对的）与 `unusable`（回落是错的，含**任何不认识的失败**），后者不加载任何权重、`mode=unavailable`、并由 `_fallback()` 产出 FLAT 即拒绝交易；想在这种情况下用未审核权重需显式 `MODEL_ALLOW_CANDIDATE_FALLBACK=1`。被拒信号的 `reason_codes` 现在带具体原因（`production_unusable:<reason>`），空注册表/过期/文件损坏在审计里不再都是同一句 `model_unavailable`。
- **`/health` 的 `models` 检查曾从错误的对象上读晋级状态**：`Runtime` 没有 `promotion` 字段（它在 `runtime.models.promotion` 上），`getattr` 拿到 `None` 后退化成 `trading_unpromoted_weights` 且 `mode=None`——**一个正确晋级的模型也被报成「在交易未晋级权重」**。现在从真正持有它的对象读取，三态如实上报：已晋级 `ok/tradable`、不可用 `degraded/production_unusable:<reason>`、无运行时 `skipped/no_model_runtime`（「未配置」与「配了但做不到」不再同形）。顺带补上 `promotion.created_at`：`provenance_problem()` 里那段按 manifest 年龄判过期的代码读的就是它，而这个键从来没被写过。
- **每日交易次数上限**（`MAX_DAILY_TRADES`，默认 60）：风控里每一条限制回答的都是「这一笔最多亏多少」，没有一条回答「一天能开几次火」。而入场门的注释自己记着单个品种**十秒内 37 次往返**——日内亏损熔断在换日时故意宽容，「许多笔小亏」这条路径对当时引擎里的每条上限都是隐形的。检查在 `risk.approve()` 内、**计数在提交成功处**：`approve()` 对随后被组合上限/执行就绪/lot step 拒绝的计划同样会跑，在那里计数会让「系统说 no 越多、预算消耗越快」，变成一个自我收紧的上限。计数进 `snapshot()`/`restore()`（忘记当天已用多少就等于每次部署重启一次预算），随 UTC 换日重置，高水位熔断不随之重置。
- **换日逻辑曾有两份且已经不一致**：`approve()` 里给了 `event_time_ms` 的分支（**生产永远走这条**）是一份内联复制品，它只清 `halted`，把 `halt_reason`/`halt_equity`/`halt_threshold` 留在那里描述**昨天**——日内熔断触发后，第二天每次拒绝都带昨天的理由码，面板上写着账户 halted 而它并不是。两份合并为 `_roll_day(equity, key=None)`。
- **`DRIFT_POLICY=block` 曾经是个不接线的开关**：`blocks_entries()` 只被 `/health` 和测试调用，运营商设成 `block` 后面板显示「正在拒绝入场」、入场一次都没被拒过。现在接进共用入场门，理由码 `feature_drift:<列名>`；没有监控或监控尚无结论时**放行**（不能因为一次没跑过的检查停掉会话），监控抛异常时放行并记 `errors.note`。
- **drift 现在会让模型要求重训**：`retrain_required()` 原有三条判据全是关于产物的（多旧/哪个数据集/哪个契约），没有一条关于市场。新增 `drifted` 判据排在三者之前；`RetrainScheduler` 通过注入的 `drift_fn` **在评估时现取**监视器结论，未配置或尚无结论读作「没检查」而不是「干净」。`/health` 的 `drift` 检查增加 `retrain_on_drift`，并在「发现了但没人会行动」时报 `feature_drift_unactionable`。
- **相关性敞口上限**（`MAX_CORRELATED_LEVERAGE`、`CORRELATION_THRESHOLD`，默认 0=关闭）：由已存 K 线计算收益率相关矩阵，与候选品种相关性超过阈值的持仓名义价值单独设限。总杠杆上限无法替代它——九个同涨同跌的品种是一个仓位穿了九件马甲。无法测出相关性的品种按"未知"处理，既不假设安全也不计入。
- **挂单风险预留**：挂单在成交前就占用 `max_positions`、单标/总杠杆与组合风险预算，成交时再复核一次（`portfolio_risk_budget_exceeded` 会记入 `fill_refused_by_risk` 事件）。此前 `risk.approve` 与 `limits.approve` 只在提交时对当时的账面检查一次，多个品种可以各自对着空账本定量、挂单、再依次成交，把组合风险放大到预算的若干倍而审计里看不出任何规则被违反。

## 止损/止盈几何硬校验

复盘发现 **16 笔成交的止盈价落在开仓价错误一侧**且几乎贴着开仓价，下一个 tick 立即触发——某品种因此产生 37 笔不足 10 秒的空转，手续费 0.97 而盈亏 −0.32。

现在计划生成后与 reanchor 后都校验：

```
LONG   stop < entry < target
SHORT  target < entry < stop
```

违反则拒绝下单并记录 `bracket_invalid:<原因>`。reanchor 只会阻止"把本来合法的几何改坏"，本来就坏的留给下单环节报错，避免静默恢复坏价位。

## 风险档位（可配置的交易策略）

激进度是一个**可运行时切换的命名档位**，不是启动常量。之前它固定在 `.env`，改一次要编辑文件并重启，而且历史会话无法说明自己当时是在什么参数下交易的。

| 档位 | 单笔风险 | 组合风险 | 目标敞口 | 持仓数 | 单标杠杆 | 总杠杆 | 日内熔断 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `conservative` 稳健 | 0.25% | 1% | 1x | 4 | 2x | 3x | 15% |
| `balanced` 均衡（默认） | 0.5% | 2% | 2x | 5 | 3x | 5x | 20% |
| `assertive` 进取 | 1% | 4% | 3x | 6 | 4x | 8x | 25% |
| `aggressive` 激进 | 2% | 8% | 5x | 8 | 6x | 12x | 30% |
| `maximum` 极限 | 3% | 12% | 8x | 10 | 10x | 20x | 40% |

每个档位**同时**设定全部 sizing 参数，因此彼此始终自洽：`max_portfolio_risk` 一定覆盖它允许的单笔风险之和（全部仓位同时止损的损失就是组合风险，而不是单笔风险乘以槽位数），单标杠杆也不超过总杠杆。

```bash
curl http://127.0.0.1:8101/api/risk/profiles                    # 全部档位 + 当前生效参数
curl -X POST http://127.0.0.1:8101/api/risk/profile \
     -H 'Content-Type: application/json' -d '{"profile":"assertive"}'
curl -X POST http://127.0.0.1:8101/api/risk/profile \
     -H 'Content-Type: application/json' \
     -d '{"profile":"balanced","overrides":{"max_risk_per_trade":0.02}}'
```

数值超出硬边界会返回 400 并指明字段与范围，而不是静默截断——例如 `{"error": "risk_field_out_of_range:max_risk_per_trade:0.0-0.1"}`。

切换只改变 sizing 参数，**保留 `day_start_equity`、熔断标志与 `day_key`**：这些是状态而非策略，中途清空会静默解除日内熔断。

档位同时写入 `risk_profile` 与 `risk_profile_name` 运行时行，并在 `risk_profile_changed` 事件中留档，因此任一历史成交都能用当时的参数解释。

Kelly 参考：45% 胜率、2:1 盈亏比下全 Kelly 约为权益的 17.5%，实务取 1/2 或 1/4 以吸收参数估计误差。上表各档位即使按 1/4 Kelly 衡量也远在其内。

## 仓位与执行

仓位不再只由"单笔风险 ÷ 止损距离"决定，也不再被绝对美元上限悄悄否决会话杠杆。三个约束同时生效，并在每个 `risk_decision` 事件和实时行情行里回报**是哪一个约束生效**（`binding`）：

| binding | 含义 |
| --- | --- |
| `risk` | 触及 `MAX_RISK_PER_TRADE`（并受 `MAX_PORTFOLIO_RISK` 已用额度限制） |
| `target` | 触及 `TARGET_EXPOSURE`，风险额度还没用完 |
| `room` | 触及 `MAX_SYMBOL_LEVERAGE` / `MAX_GROSS_LEVERAGE` |

组合上限一律是**权益倍数**，不是绝对美元：同一个数字在 100 USDT 和 10000 USDT 账户上含义一致。旧的 `MAX_NOTIONAL_MULTIPLE=2` 会直接否决会话里设置的 3x/10x 杠杆，`MAX_SYMBOL_NOTIONAL`/`MAX_TOTAL_NOTIONAL` 的绝对美元值则不随权益缩放，两者都已移除。

执行侧的三条规则保证挂单不会在几小时后带着过期计划成交：

- 每张订单都有 `ORDER_TTL_MS` 期限（默认 90s），到期即撤单，不再无限期挂单排队；
- 行情已走完止损的 `MAX_ENTRY_ADVERSE_FRACTION`（逆行）或止盈的 `MAX_ENTRY_CHASE_FRACTION`（顺行）时放弃入场，阈值按各自的距离归一，因此对 4% ATR 的山寨币和 BTC 都适用；
- 成交后按实际成交价平移止损与止盈，使实际风险等于风控计算的 `risk_cash`，而不是继承旧价位导致止盈已在成交价之后。

同一标的同一时刻只允许一张挂单和一个持仓；任何成交（含延迟成交）都会触发 `ENTRY_COOLDOWN_MINUTES` 静默期。合约的最小变动价位、步长与最小名义价值来自交易所 `exchangeInfo`，不再对所有品种使用同一套硬编码默认值。

离线回测（`app/backtest/backtest.py`）复用与实盘相同的 `RiskEngine`、`PortfolioLimits` 与成交重锚逻辑，因此回测结果可以用于预测纸面账户的行为。

## 实时性与审计保留

实时性瓶颈经实测确认在数据库，不在渲染：

| 操作 | 修复前 | 修复后 |
| --- | --- | --- |
| `recent_events(100)` | **6273 ms** | 0.3 ms |
| `simulation_history(50)` | **562 ms** | 1.7 ms |
| `/api/state` 端到端 | **81 s** | 0.03 s |
| `/health` | 超时 | 0.002 s |

两个根因：

1. **`events` 表缺 `event_time` 索引**，`ORDER BY event_time DESC LIMIT 100` 退化为 `SCAN events` + `USE TEMP B-TREE FOR ORDER BY`，每次全表扫描并排序 107 万行 / 542 MB。已加 `idx_events_time` 与 `idx_events_type_time`。
2. **表体积本身失控**。107 万行只覆盖 0.92 天，其中 75.8 万行是 `strategy_decision`，而每 5000 行只对应 **32 个不同决策**（156 倍冗余）。数据量随 tick 频率而非时间增长，所以**按天数保留无法约束它**。

因此保留策略按**行数**而非天数：热表只保留最新 `EVENT_RETENTION_ROWS` 行，更早的先复制进 `events_archive` 再删除，不丢数据。成交、爆仓、会话生命周期事件属于 `DURABLE_EVENT_TYPES`，不受行数限制，`simulation_history` 因此始终完整。

```bash
python -m app.storage.retention --stats              # 表体积与事件计数
python -m app.storage.retention --prune --rows 50000 # 手动执行保留
python -m app.storage.retention --vacuum             # 回收磁盘（会重写文件）
python -m app.storage.retention --archive-session <session_id>
```

归档表本身此前**没有保留策略也没有读取接口**：热表搬走的每一行都在这里永久累积（实测 102 万行且持续增长），而里面的数据无法访问——"不丢数据"这句话在实际意义上是假的。现在 `EVENT_ARCHIVE_ROWS`（默认 100 万）约束归档行数，`Store.archived_events()` 提供按类型/品种/会话回读。

注意 `-wal` 文件可能远大于主库（实测曾达 **7.4 GB**）。长时间运行的连接需要定期 `PRAGMA wal_checkpoint(TRUNCATE)`，否则 WAL 会一直增长。

## 衍生品数据采集（有硬截止）

交易所的 `/futures/data/*`（持仓量、大户持仓/账户多空比、全局多空比、taker 买卖量比）**只保留 30 天且不提供历史归档**：今天不采，明天任何价格都买不回来。此前没有任何组件调用这些端点，`derivatives.open_interest` 实测全为 0。

`app/market/derivatives_collect.py` 按 `DERIVATIVES_COLLECT_INTERVAL_SECONDS`（默认 300s）采集全部五个端点，合并到同一时间轴后写入 `derivatives_detail`。该表刻意与 `derivatives` 分开：这些列**可空**，缺失端点必须能与"读数确实为 0"区分开，而原表的三列做不到。采集失败只计数不抛异常——因限频打断决策循环是用一种数据丢失换另一种。

K 线侧的订单流字段此前是被**丢弃**的：Binance `/fapi/v1/klines` 数组的 7/8/9 号位分别是 quote volume、成交笔数、taker 买量，解析路径只读了 0-6 号。现在 `candles` 表新增 `taker_buy_volume`/`quote_volume`/`trades` 三列（老库自动 `ALTER TABLE` 补齐），并用 `app/features/order_flow.py` 生成 taker 失衡及其 z 分数、成交笔数与平均单笔规模、持仓量变化与多空拥挤度等特征。成交笔数为 0 标记为"未测量"而非"均衡"——两者都是浮点数，只有笔数能把它们分开。

标记价轮询现在也会把 mark price 与资金费率写入 `derivatives`（每分钟一次），否则 `funding_*` 与 `mark_basis` 系列只停留在最后一次回填 CLI 运行的那一天。

## 标签时点与数据集

`LABEL_VERSION=label-v2`。旧标签是 `close[i+h] / close[i] - 1`，即从**产生信号那根 K 线的收盘价**起算——而实盘不可能在该价格成交：它在一根闭合 K 线上做决策，成交发生在市场给出的下一个价格，也就是**下一根 K 线的开盘价**。这个偏差方向与所有标签一致，且成本模型无法修正它，因为它不是成本，是另一个价格。

现在标签从 `bars[i+1].open` 起算、在 `bars[i+1+h].close` 结算，数据集行同时记录 `entry_time` 与 `entry_price`。标签版本与特征版本分开记录，因为两者的失效方式不同：特征变更让模型**无法服务**，标签变更让模型**看起来完全正常但是错的**。

## 研究数据与训练

实时进程将去重后的闭合 K 线写入 `data/research.sqlite3`，同时保存权益、交易和运行状态。生成监督学习数据集：

```bash
python -m app.models.train --data-dir data --symbols BTCUSDT,ETHUSDT --interval 1m
```

生成的数据集是按时间顺序构造的未来收益标签。可用 `python -m app.models.fit_model data/training_dataset.jsonl` 训练一个可解释的线性基线，输出 train/validation/test 的 MAE 和方向准确率到 `data/model_baseline.json`。正式训练必须使用时间切分、walk-forward、手续费/滑点压力测试，不能把当前规则策略的 confidence 当作校准概率。

## 安全边界

`TRADING_MODE=paper` 是唯一实现模式；真实交易执行器刻意未实现。API Key 不应写入仓库，且任何未来执行器都必须禁止提现、限制 IP、经过 Testnet 和人工审核。模型只产出方向与期望收益，仓位大小、保证金、组合限额与撮合仍由风控和纸面 broker 决定，模型没有任何下单能力。

## 测试

```bash
pytest -q
python -m compileall -q app
```

风险提示：合约交易可能导致全部本金损失。本项目不承诺盈利。

## 本地验证

```bash
pytest -q                      # 874 项
python -m compileall -q app
```

分层重构与后续优化新增的测试文件：

- `tests/test_architecture.py`：把上面那张分层表变成断言。检查模块级向上 import、函数内向上 import（隐藏违规）、跨包环、以及 `core/` 的零依赖，另外给延迟 import 的总数设了上限。
- `tests/test_cost_contract.py`：钉住训练与线上对“可交易”的同一定义。这是审计里代价最大的缺陷——训练用「预测值在预测方向上超过往返成本」判定，线上却用 `abs(expected)` 对一个 0.5bp 的地板判定，把符号和成本都丢了。在已发布的产物上实测：训练规则 188,640 行里 269 行活跃（0.14%，净 +2.08bp），线上规则 188,521 行活跃（99.94%，净 -13.95bp）——700 倍的活跃比例差异，加上期望值反号。
- `tests/test_equity_and_marks.py`：权益曲线采样稳定性，以及标记价与成交价的分工（资金费与维持保证金按标记价结算）。
- `tests/test_exit_source.py`：离场来源统一（默认 bar 确认，`exit_on_tick=True` 显式选择逐笔），强平仍然盘中触发。
- `tests/test_metrics_endpoint.py`：`/metrics` 在 `content_type` 里带 `charset` 会让 aiohttp 对每次抓取抛 `ValueError` —— 实测修复前 500、修复后 200 + 103 行。
- `tests/test_scale_out.py`：分批止盈（`close(quantity)` 按 lot FIFO 消耗、资金费按比例切分、分批与一次性的盈亏完全相等），以及订单台账与持仓的对账。`position_quantities()` 曾读错字段名而恒返回 0，使这个检查从未生效。
- `tests/test_edge_sizing.py`：仓位随 edge 单调、只能缩小不能放大、`edge_bps` 与 `probability_up` 两种说法在盈亏平衡点必须一致、`SIZING_REFERENCE_EDGE_BPS=0` 可整体关闭。
- `tests/test_latency_model.py`：延迟按波动率与 `sqrt(时间)` 缩放、默认关闭、挂单不付延迟、走盘口阶梯的成交路径不重复收价差。
- `tests/test_triple_barrier.py`：三重障碍的分辨与对称性、唯一性权重（含一条计时回归——第一版按毫秒分配数组，587 行耗时 134 秒）。
- `tests/test_validation.py`：Deflated Sharpe 随试验次数单调下降、随样本量上升、能区分噪声与信号；PBO 的方向参数。
- `tests/test_archive.py`：归档读取。列按表头名而非位置解析（`count` 与 `trades` 是同一字段的两个名字）；**404 必须重试后才可采信**（实测该 CDN 会对存在的文件连续数分钟返回 404）；连接失败**不得**被当成「归档不存在」；四个边缘地址要逐个尝试（其中一个拒连，而 `urllib` 只试第一个）；缓存命中不得触网；中断的下载不得在缓存里留下截断文件。
- `tests/test_walkforward_purge.py`：fit 集与早停集之间的 purge。gap 取**最大**标签跨度而非中位数；fit 的最后一条标签必须在验证集开启前结束；按时间戳切分（按行下标会把同一根 bar 的多品种行劈开）；历史不足时回退而不报错。
- `tests/test_session_scope.py`：**行情断线不是品种问题**。`without_market_data` 曾因 `book_seen_at` 为空而淘汰全部品种（62–85 秒一轮），而 `resolve()` 只对持仓品种打分，换币即丢弃 pending 预测 —— `symbol_edge` 因此恒无样本、恒拒一切。测试锁定「传输存活由调用方显式传入」，并保留原有的逐品种淘汰规则。同时锁定**会话结束不等于一种退出策略**：`end()` 曾把会话结束原因直接当平仓原因，使 9 笔市价标记被记成 `manual`。
- `tests/test_session_accounting.py`：跨会话盈亏**按各自本金折算**。起始资金由前端任填，库里混有 100 与 10000，`all_time_realized_pnl` 曾把它们直接相加。
- `tests/test_edge_seeding.py`：**闸门必须能从被拦的品种身上学习**。`symbol_edge.allows()` 在样本 < 30 时一律拒绝，而样本来自主模型**提出的方向**；审计行此前只写**过闸后的 `side`**，被拦品种因此全是 `FLAT`，从审计轨重建证据的 `seed_from_history()` 只还原出 1 个品种 19 个样本，其余 17 个永久 `insufficient_edge_samples`——而新品种要从 0 攒到 30 样本需连续存活 165 分钟，会话根本活不到。测试锁定 `proposed_side` 与 `market.bar_time` 必须写进审计行，以及旧行可从 `decision.votes` 的符号和反推方向。同时锁定 `symbol_untrained()`：模型只在 12 个主流币上训练过，而 `source=gainers` 选出的 5 个小币与训练集**交集为空**，`atr_pct` 实测 0.027–0.034 对训练上界 0.0148，OOD 拦截是正确的——错的是默认选币越出了训练域。
- `tests/test_mark_pump.py`：mark 价格新鲜度判据比的是**数据年龄**，不是**往返耗时**。旧写法取「收到响应之后」的时钟减 venue 时间戳，得到的就是这次请求的耗时；本机 `/fapi/v1/premiumIndex` 实测中位 5185ms，稳定超出 5s 窗口，于是每一行都被丢弃、`derivatives` 表 47 小时没写入、funding 四个特征永久降级、全部决策被判 FLAT。测试覆盖「慢请求不得丢弃整批」这条回归本身。
- `tests/test_web_assets.py`：看板**确实被提供**，而不只是路由注册过。重构把 `app/web.py` 搬成 `app/web/web.py` 之后，`WEB_ROOT` 的 `parent.parent` 静默指向了不存在的一层，`/` 与全部 `/assets/*` 一起 404，页面空白且无任何报错。测试直接从 `index.html` 解析它引用的资源再逐个查存在性，并断言 `WEB_ROOT` 不得落在 `app/` 内。

本轮优化新增/改写的测试集中在 `tests/`：`test_venue_check.py`（站点识别与判定阈值）、`test_portfolio_limits_round.py`（高水位熔断、连亏、波动率目标、相关性敞口、挂单预留）、`test_data_collection.py`（K 线订单流解析、衍生品采集器、特征家族）、`test_label_and_archive.py`（标签时点、归档保留与回读、OOD 门禁可达性）、`test_gate_significance.py`（晋升门禁、标准误与 FDR）、`test_fidelity_round.py`、`test_feature_parity.py`、`test_margin.py`。

审查文档：模拟与决策缺陷见 `docs/simulation-and-decision-audit.md`（含每条问题的 `file:line` 证据与复现脚本），完整多视角审查见 `docs/paper-trading-ml-review.md`，开源项目对标见 `docs/quant_oss_landscape_report.md`。
