# 模拟交易与交易决策系统 · 多视角对抗式审查 + 外部对标 + 数据方案

> 审查对象：`D:\BaiduNetdiskDownload\money`（Binance USDT-M 永续纸面模拟 + ML 决策，Python/FastAPI/SQLite，app/ 约 8160 行）
> 方法：8 个独立视角并行审查（6 个代码对抗视角 + 2 个外部调研视角），每条结论要求 file:line 证据；
> 主报告对最高严重度的结论做了二次独立核验，标注「已复核」。
> 外部调研受联网工具中断影响，链接与端点未经逐条在线校验，已单独标注。

---

## 一、总体判断

这套系统的**工程骨架是合格的**（单一决策入口、幂等 K 线存储、风控/撮合/审计分层、回测与实盘共用 RiskEngine），
但**研究结论目前不可信**，原因是三条独立的链路各自断在同一个地方——**离线与在线用的是两套世界**：

| 断裂点 | 离线（训练/回测） | 在线（纸面实盘） | 后果 |
| --- | --- | --- | --- |
| 特征 | 4 个 funding 特征有真实值 | 恒为 0（常量） | 模型 40% 的输入维度在实盘是死的 |
| 离场 | 用 K 线 high/low 保守判定 | 用 mark price 逐点判定，且按穿越价成交 | 插针止损被漏掉 + 成交价优于触发价 |
| 门槛 | 回测不接 meta-labeling 门槛 | 实盘门槛是真前置 | 回测收益系统性高于纸面账户 |

因此当前 10 个特征、125.8 万行数据集训练出的候选模型（test 方向准确率 48.2%/49.x%），
**既不构成盈利证据，也不构成亏损证据**——它测的不是实盘跑的那套系统。

### 最致命的 5 条（已逐条复核代码）

1. **实盘 4/10 特征是常数 0**（`app/features/features.py:65` + `app/models/train.py:70` 对 `app/strategy/live_models.py:504/525/335/603`）——训练/服务偏移（train/serve skew），且 OOD 门禁结构上抓不到它。
2. **强平是玩具模型**：固定 0.5% 维持保证金、无风险档位、`liquidation_fee_rate` 定义了从不使用、允许负权益（`app/trading/margin.py:6-16`）。
3. **资金费桶先推进后结算**：`self.funding_bucket=bucket` 在结算循环之前（`app/trading/simulation.py:66`），一次快照失败即永久漏结；8 小时写死，4h/1h 品种漏结。
4. **模型晋级门禁在代码上永不可通过**：`app/models/model_registry.py:27` 要求 `features-v1`，而 `app/features/feature_spec.py:19` 是 `features-v3`；`portfolio_oos` 无任何生产者。系统实际一直跑未过门禁的候选权重。
5. **审计保留策略从未执行过一次**：`app/strategy/decision_loop.py:360` 调用 `runtime.database_worker`，而 `Runtime`（`app/runtime_context.py:65-89`）**没有这个字段**，`main.py:201` 建的是局部变量 → 每次 `AttributeError` 被 `except` 吞掉。实测 `events` 表 158k 行 vs 目标 50k，`retention_pruned` 事件零条。

---

## 二、多视角审查：缺点与缺陷

### 视角 A｜纸面撮合与执行的现实性

- **[S1] 强平链路不完整**（`app/trading/margin.py:14-16`）：`maintenance = Σ notional × 0.005`，与名义规模无关；真实交易所是分档 MMR（0.4% 起，大仓位可达数个百分点）。后果：大仓位强平价被推远，**系统性低估强平概率**；强平时只收普通 taker 费，少记约 0.5% 名义的强平费；`self.cash += gross_pnl - exit_fee` 无下限（`app/trading/simulation.py:96`），穿仓后权益为负且无保险基金/ADL 接管。
- **[S1] 结算价用最新价而非标记价**：`should_liquidate` 传入的 `marks` 由 mark_price 事件驱动，但强平判定按全账户权益而非逐仓，且 `initial_margin = notional / leverage` 用**会话输入杠杆**（`app/trading/margin.py:8`），与 risk 引擎的三个杠杆参数互不相干（见视角 C）。
- **[S1] 离场触发与成交价过于乐观**（`app/trading/simulation.py:76-77`）：`hit = price<=stop or price>=target` 后 `close(symbol, price, ...)` —— 用**穿越价**成交。真实场景是止损按更差价成交（滑点不利方向），这里取的是恰好价格。叠加 `close` 里又加一次 `slippage_bps`，方向对但幅度远小于插针。
- **[S2] 深度不足直接拒单、限价单无排队**（`app/trading/broker.py:88-92` `return None, insufficient_depth`）：真实市场会部分成交并留下剩余敞口；这里"要么全成要么不做"，系统性回避了真实中会发生的风险暴露。`PARTIALLY_FILLED` / `open_fill` 加仓路径在生产链路上基本不可达（死代码）。
- **[S2] 成本模型过于单一**：`app/trading/broker.py:125` 固定 `slip = price × 2bps`，与单量/波动/盘口深度无关；单一 `fee_rate=0.0004`（相当于永远享受 BNB 折扣）；`liquidity="taker"` 硬编码，无 maker/taker 分档。
- **[S2] 权益曲线被截断 + 年化系数写死**（`app/trading/simulation.py:83-84` + `app/core/metrics.py:4` `periods_per_year=365*24*60`）：每次 `mark()` 追加一点（实盘每 symbol 每个 mark 事件一次），超过 20000 点从头删；用 1 分钟系数缩放"每事件"序列 → 年化波动低估、**夏普同倍高估**。
- **[S3] 死配置**：`ContractSpec.leverage/maintenance_margin_rate`（`app/trading/broker.py:9`）、`OrderIntent.stop_price`（`app/core/domain.py:23`）、`MarginModel.liquidation_fee_rate`（`app/trading/margin.py:7`）全部从未被读 → **系统从不向交易所下止损单**，全部离场依赖进程内轮询，进程一停保护即消失。
- **[S3] 时钟混用**：`app/trading/simulation.py:87` 未传 timestamp 的平仓落墙上时间，开仓时间在交易所时间与本地时间之间混用，`holding_ms` 不可信。

**缺失能力**：交易所侧止损/止盈单与 `reduce_only` 撮合期校验；逐仓/双向持仓模式；盘口冲击与队列位置模型；成交回报延迟与拒单限流建模（-1021/-2019）；完整强平链路（档位 + 强平费 + 保险基金 + ADL）；**挂单保证金冻结**（`available_margin` 只算已成交仓位，不含挂单占用）。

### 视角 B｜特征、标签、数据泄漏与样本质量

- **[S1] 4/10 特征在实盘恒为 0**（已复核）：`funding_features()` 全仓库只被 `app/models/train.py:70` 调用；服务侧 4 处 `snapshot(...)` 均不传 funding。数据集里这 4 个特征有真实分布（`funding_rate` 611 个不同取值），实盘一个字节都没喂进去。**这是整套系统最严重的问题**。
- **[S1] OOD 门禁结构性空转**（已复核）：`data/research_v3/training_dataset.jsonl.profile.json` 的 bounds 与 `app/features/feature_spec.py` 的 `FEATURE_CLIP` **逐字相同**（`return_10 [-0.5,0.5]`、`funding_z [-5,5]`、`mark_basis [-0.02,0.02]`、`rsi [0,100]`），而服务侧写特征前已 clip → 这些维度**永远不可能越界**。可触发维度仅 4/10。另外逐特征 min/max 是单变量盒，对联合分布无效，且被坏印价污染（`feature_spec.py` 注释自认 DOTUSDT 出现过单根 135% 的 bar）。
- **[S2] 回测入场价是已收盘 K 线的 close**（`app/models/train.py:77` + `app/strategy/live_models.py` 的 `entry = features[price]`）：离线假设零延迟成交在收盘价，实盘必然更晚。5m ATR 常在 20-170bp，一笔延迟漂移与 12bp 成本同量级。
- **[S2] `mark_basis` 名不副实**（`app/market/funding.py:142-143`）：用的是**上一次资金结算时**的 markPrice（最长 8 小时前）除以当前 close，本质是滞后动量而非基差。±2% 的 clip 导致 **9.71% 的样本被钉死在边界**（66,011 行 = +0.02，56,196 行 = −0.02）。
- **[S2] 重叠标签无去重、无样本权重**：`DEFAULT_HORIZON=12` 且每根 K 线产出一行 → 相邻 12 行标签窗口完全重叠，125.8 万行的**有效独立样本约 1/12**；`grep sample_weight` 无结果。`min_active_samples=500` 用**行数**冒充样本数，而 `app/models/tabular_model.py:44` 自己已标注 `evaluation: independent_overlapping_samples_not_portfolio`——标注了缺陷却仍把它当成绩单。
- **[S2] walk-forward 有泄漏**：`app/models/walkforward.py:118-119` 的 `fit_idx, valid_idx = train_idx[:-guard], train_idx[-guard:]`，早停用的 valid 与 fit 之间**没有 purge**，`train_idx[-guard-1]` 的标签落在 valid 区间内；bootstrap 又对"已剔除空隙"的数组按时间块取样，块内不连续。
- **[S2] 幸存者偏差**：训练 universe 来自**今天的**行情快照（`app/models/training_job.py:211-214` → `app/features/universe.py`），已下架合约不在其中；`candles` 表无 delist 标记。实测数据集恰好是 12 个"活到今天的赢家"主流币。
- **[S2] 单根 `close` 同时参与特征与标签**：`app/models/train.py:72-77` 用 `bars[index]` 的 close 既算特征又算标签起点，本身无未来函数（边界正确、funding 用 `bisect_right` 防前视），但叠加回测按 close 成交即构成乐观偏差。
- **[S2] 质量门禁漏项 + 生产路径关掉了门禁**：`app/features/quality.py:51` 只查 `volume < 0`，**不查 `volume == 0`**；`validate_dataset` 不查缺失 K 线、时间间隔、价格跳变；`app/models/training_job.py:234-237` 用 `strict=False` 跑生产训练，坏 bar 照样进数据集；`app/storage/storage.py:62-68` 静默丢弃畸形行且无计数器。
- **[S3] `app/models/drift.py` 是死代码**（全仓无调用者）——系统声称有漂移监控，实际没有任何在线检查能发现"funding 从有值变恒 0"。

**缺失能力**：订单流/微观结构特征**完全缺失**——K 线里现成的 `taker_buy_base_asset_volume` / `number_of_trades` 在入库那一刻就被丢弃（`app/storage/storage.py:19-25, 63-66` 只取 5 个字段）；ΔOI 缺失；跨品种/BTC beta/市场宽度/时段季节性缺失；真正的 purged K-fold + embargo；**训练/服务特征一致性回归测试**（`tests/test_feature_scale.py` 只验证尺度无关性）。

### 视角 C｜风控、组合构建与仓位管理

- **[S1] 挂单成交路径完全绕过组合风险预算与杠杆上限**（已复核逻辑链）：`app/strategy/decision_loop.py:126-130` 用**当前**持仓算 open_risk 后调 `risk.approve` / `limits.approve`；若 `execution.py` 因 `execution_ready` 为假或深度缺失而 return（不撤单），订单寄存，之后由任意 book 事件再次 `settle_order` 成交（`app/strategy/feeds_handlers.py:42-46`），成交走 `app/trading/simulation.py:27-50` 的 `open_fill`——**只校验 `available_margin`，不再看 risk、不再看 limits**。N 个品种各按空仓账面定尺寸挂单并陆续成交，总风险可达 N×单笔风险，`max_portfolio_risk` / `max_gross_leverage` 形同虚设；已被日内熔断的账户，之前挂出的单仍会成交。
- **[S1] 启动时 risk 与 limits 不同源**（已复核）：`app/main.py:167-168` 只调 `risk.apply_profile(profile)`；`app/runtime_context.py:99` 用 `portfolio_limits(settings)`，而 `app/trading/guards.py:34-40` 取的是 `.env` 的 `max_symbol_leverage/max_gross_leverage/max_positions`。只有运行期切档 API（`app/strategy/session_control.py:33-34`）才同时更新两者。**重启后若持久化档位是 maximum，面板回报总杠杆 20×，实际执行上限仍是 5×/3×/5 个持仓。**
- **[S1] 保证金杠杆与 risk 的三个杠杆参数互不相干**：`MarginModel.leverage` 来自会话输入（`app/web/web.py:71` 默认 3），`set_profile` 从不更新它 → 切到 aggressive/maximum 时 profile 的敞口**永远无法达成**（保证金先付不起），订单以 `account_rejected_fill` 结束。**实际并发持仓数由 UI 上的杠杆输入决定，而不是 max_positions。**
- **[S1] 时间止损单位错误**（已复核）：`app/strategy/decision_loop.py:246` `getattr(runtime.settings, "interval_ms", 300000)`，而 `app/core/config.py` **没有 `interval_ms` 字段** → 永远取 300000。1m 周期下时间止损慢 5 倍。当前 `.env` 恰为 `INTERVAL=5m`，错误被掩盖。
- **[S1] 逐品种门槛统计效力为零**（已复核 `app/strategy/symbol_edge.py:138-151`）：判据只有 `count >= 30` 且 `mean > 0`，**无标准误、无置信区间、无 FDR 校正**。在几百个品种上做滚动单侧检验，纯噪声必然挑出一批"赢家"；`app/main.py:85-87` 的 `horizon_analysis(..., min_samples=8)` 更激进——**用 8 个样本**算出的 argmax 直接写进实盘 `time_stop_bars`（`app/main.py:97`），且 7 个 horizon 取最大、样本重叠、无校正。这是拿噪声当参数。
- **[S2] 探索槽会把负期望品种塞进组合**（`app/backtest/simulation_session.py:159-165`）：当 unknown 为空、rejected 非空时，被 quota 挤出的槽位交给 **rejected**，而注释明写「proven-bad 的品种已经被回答过，花槽位什么也买不到」——注释与实现相反。
- **[S2] 组合层没有任何相关性/因子敞口约束**：加密永续 beta≈1，5 笔山寨多单在 BTC 一根阴线里同时止损；"全部仓位同时止损 = max_portfolio_risk" 只在止损恰好在止损价成交时成立。选币按涨幅排名（`source="gainers"`）→ 追高 + 波动率聚集。敞口用 entry 名义（`decision_loop.py:119`）而面板用 mark 名义（`snapshot.py:18,52`），两者在趋势中发散。
- **[S2] meta-labeling 用「信号结算」而非「成交结算」**：`app/strategy/symbol_edge.py:89-116` 固定 N 根收益扣 12bp 成本；实际持仓有 1.5 ATR 止损、1.8RR 目标、保本/跟踪、时间止损——两者不是同一个分布。门槛是唯一决定"这个品种能不能下手"的机制，口径错即用错误基准分配资金。
- **[S2] 没有回撤熔断/连亏减速/波动率目标/单日次数上限**：唯一的账户级停手机制是日内熔断，且跨 UTC 零点按**亏损后净值**重设基准（`app/trading/risk.py:134-135`）→ "每天重新亏 20%" 是被允许的路径。
- **[S3] `below_min_notional` 永不可达**（`app/strategy/decision_loop.py:141-144` 的 `or` 链被恒真的 `portfolio_ok` 短路）；`risk_cash` 不含手续费与滑点；`open_risk` 用**已收紧的**止损计算 → 保本止损把组合预算静默释放。

### 视角 D｜ML 方法论与模型治理

- **[S1] 晋级门禁双重死锁**（已复核 `app/models/model_registry.py:27` vs `app/features/feature_spec.py:19`）：`feature_version != "features-v1"` 直接拒绝，而现网特征版本是 `features-v3`；`portfolio_oos` 全仓库只有读方没有写方；候选 manifest 的 test 准确率 46.97%~49.70%，也低于 `min_directional_accuracy_pct=50.0`。`data/models/` 为空 → **生产模型永远不可能存在**，`app/strategy/live_models.py:211-213` 静默降到未验证候选权重。
- **[S1] 校准链路不存在**：`fit_isotonic` 只被测试调用，无 `calibration.json` 生产者 → `app/models/model_runtime.py:10-12` 恒返回 `calibrator_missing`。所谓"概率"来自 `calibrate(0.5 + expected_return*10, ...)` 这个魔数映射（`app/strategy/live_models.py:89`）。**README 的"含 isotonic 校准"不成立，且无任何可靠性图/Brier/ECE。**
- **[S1] 集成不是集成**：两个 GBDT 同数据、同标签、同目标；子代理实测前 4 万行 `corr = 0.808`、符号一致率 **99.77%** → `MODEL_MIN_AGREEMENT=0.6` 是空操作（只拒掉约 0.23% 的样本）。
- **[S1] 成本覆盖检查可被几何满足**（`app/strategy/live_models.py:661-663`）：`expected_move = max(target_distance, |expected_return|*price)`，而 `target_distance` 来自纯 ATR 几何 → **模型的期望收益只能把目标推远，永远不能因边际不足而拒绝交易**。`edge_bps = abs(expected)*10000` 是**毛**边际，默认门槛 0.5bp 而往返成本 12bp。
- **[S1] 门槛证据每次重启归零**：`app/main.py:61` 启动时恒调 `seed_from_history`，而它跳过 side 不在 (LONG, SHORT) 的记录（`symbol_edge.py:223`），落盘的却是**门槛之后**的 side → 被拒品种重启后证据为 0。README 的"被拒绝的品种仍在积累证据、可以自己挣回交易资格"**只在同进程内成立**。`decision_loop.py:196` 写的 `symbol_edge` 快照无任何读取方（含 pending，重启即丢）。
- **[S2] 回测不接门槛与 Chronos**（`app/backtest/backtest.py:174` 未传 `symbol_edge`、`chronos_enabled=False`）→ README 由"共用 RiskEngine"外推的"回测结果可以用于预测纸面账户的行为"不成立，回测收益系统性更高。
- **[S2] 评估缺组合口径 + 年化单位错误**：放行判据只有 `active_samples / prob_positive / net_edge_bps` 三项，无 Sharpe/Sortino/MDD/换手；`net_edge_bps` 是"每笔活跃样本均值"，不等于账户收益（忽略相关性、保证金占用、并发限制）。
- **[S2] Chronos 成员被永久锁死在前 8 个品种**（`app/strategy/live_models.py:375-376` 的 `chronos_seen` 从不清理）→ 会话换币后新品种永远拿不到该投票；15 根中位收益与表格模型的 12 根标签**等权相加**，且 `expected_net_return=None, cost_bps=0.0` 意味着它从不承担成本。
- **[S2] 漂移检测与自动重训是死代码**：`app/models/drift.py`、`app/models/retrain.py`（`RetrainScheduler`）、`app/training_policy.validate_split_sizes` 在 `app/` 内无任何调用者；`ModelRegistry` 无 demote/rollback。
- **[S3] shadow 不是影子评估**（`app/strategy/shadow.py:69-70` 只存最新一帧，无结算、无对比）；candidate 选择只看 `st_mtime`（`app/strategy/live_models.py:262`）。

**缺失能力**：保形预测/置信集；模型不确定性（成员方差、分位数宽度——Chronos 的 0.1/0.9 分位算出来后直接丢弃）；regime 检测；在线学习/滚动重训；特征重要性与漂移监控；模型回滚；按波动率/时段/持有期的分层评估。

### 视角 E｜数据管道、存储与运行时可靠性

- **[S1] 保留策略从未成功执行**（已复核）：`app/strategy/decision_loop.py:360` 调 `runtime.database_worker`，`Runtime`（`app/runtime_context.py:65-89`）无此字段，`main.py:201` 建的是局部变量并只传给闭包与 cleanup → 每次 `AttributeError` 被 `decision_loop.py:374-375` 吞掉。`EVENT_RETENTION_ROWS=50000` 与 `EVENT_RETENTION_INTERVAL_SECONDS=900` 是彻底的 dead config。
- **[S1] `events_archive` 只写不读、永不裁剪**：全仓无任何读取查询、无 `event_time` 索引、无二级裁剪；实测 102 万行，`events` 158k 行 vs 目标 50k，库 1.2 GB。**README 的"不丢数据"在工程上等价于"数据不可读"。**
- **[S1] OI 从未被采集**：`app/derivatives.py` 的 `DerivativesStream` 零实例化，`app/market/market.py:117 open_interest()` 零调用，`parse_funding` 不产出 `open_interest` 键 → 实测 `SELECT sum(open_interest) FROM derivatives` = **0.0**（13,140 行）。README 的"持仓量历史 SQLite 表"制造了已采集的假象。
- **[S2] WS 断线后无回补、无缺口检测**：`app/market/websocket_feed.py:53-84` 只有退避重连；`app/market/market.py:93 backfill_klines` **零调用点**；唯一回补是 `decision_loop.py:425` 的 250 根窗口且只覆盖会话选中品种。断线超过 250 根（5m ≈ 21 小时）出现**永久空洞且运行期完全不可见**。
- **[S2] REST 失败会把已收盘 K 线缓存清空**（`app/strategy/decision_loop.py:423-431`）：`return_exceptions=True` 把网络失败变成 `raw=[]` → `closed_rows[symbol]` 被写成空列表，一次超时就让该品种本地历史归零。
- **[S2] REST 客户端零限频、零退避**：`app/market/market.py:63-68` 无 418/429 分支（对比 `ingest.py`/`funding.py` 都正确处理了），无 `X-MBX-USED-WEIGHT-1M` 读取；`decision_loop.py:379` 的 `market_snapshot()` **无 try** → 一次 429 中断整轮（不落 K 线、不决策、不跑保留），被 `main.py:292` 吞掉。
- **[S2] universe 缺字段被静默判为"上市 0 天"**（`app/features/universe.py:40-41`）：`onboardDate` 缺失 → `age_days=0` → 被 `min_age_days` 一票否决，且无人能区分是数据缺失还是真新品。
- **[S3] `/metrics` 只有一行指标且调用完整 `state_fn`**（`app/web/web.py:172`），`app/ops/prometheus.py` 未接线；`/health` 恒返回 ok，不检查 WS 连接、最后事件时间、DB 可写性、决策循环存活。**WS 静默死掉、决策循环每分钟抛异常、保留策略从未运行——这三件事 `/health` 全部报 ok。**
- **[S3] WAL checkpoint 只存在于 README**（全仓无 `wal_checkpoint` 调用）；`domain.Event` 对时间戳无单位校验；`storage.record_equity` 在时钟回拨时**伪造**递增时间戳。

### 视角 F｜决策循环、会话生命周期与可复现性

- **[S1] 重启后 live account 混入全历史成交**（已复核 `app/main.py:158-162`）：`session.account.trades = store.all_trades()`，而 `end()`（`simulation_session.py:87-110`）的胜率/盈亏比/持仓时长全部基于该列表，`return_pct` 却用新会话本金 → **会话绩效口径被污染**；且 `PaperAccount.snapshot()` 含 `trades`，每次持久化整表重写。
- **[S1] 会话启动失败不回滚**（`app/web/web.py:70-82` → `app/strategy/session_control.py:44-51`）：`set_profile` 抛 `ValueError` 时返回 400，但 `session.status` 已是 `running`、account 已更换，而 `risk.reset_for_session` 未执行 → **新会话开局即继承上一会话的熔断状态**。
- **[S1] 决策循环跨会话下单竞态**：`decision_loop.py:336-344` 在 `await runtime.decisions.signal(...)` 之后直接 `submit_entry`，中间无 session_id 校验；该 await 期间 HTTP `POST /api/simulation/start` 可换掉 account，事件里记的却是新 session_id（**审计上看不出异常**）。
- **[S2] 入场被拒时不更新去重状态** → `decision_loop.py:301-311` 只有成功才写回 `state.per_symbol`；对新选入的品种以 `strategy_min_interval_ms=100ms` 节奏重试并每次写 `risk_decision + order_rejected` 两条事件 → **拒绝事件风暴**，50000 行保留额度在 10 分钟内被反复冲掉（而保留策略本来就是坏的）。
- **[S2] "tick 决策"不使用 tick 价格**：`app/strategy/live_models.py:497` 过滤掉未闭合 K 线，缓存键只随闭合 K 线变化 → tick 与 bar 两条路径共享同一信号对象。真正的问题在别处：`manage_open_position`（唯一的移动止损/时间止损入口）**只从 `evaluate_tick` 调用**，而它只由 WS 回调驱动 → **WS 一断，跟踪止损与时间止损全部停止**，`price_pump` 只更新价格（`app/market/pumps.py:43-60`），尽管其文档写着 "decisions keep flowing when the websocket stalls"。
- **[S2] 主循环每轮清空 `last_error`**（`app/main.py:287-294`）→ mark 泵/价格泵/回补的连续失败被无声抹除，面板显示正常。
- **[S2] running 但未选币时仍按 `SYMBOLS` 下单**（`decision_loop.py:421` + `simulation_session.py:69` 清空 `selected_symbols`）→ 会话按"跌幅榜 3 个品种"启动后，头一分钟实际交易的是 BTC/ETH，且不受 `symbol_count` 约束。
- **[S3] README 与代码的多处矛盾**（详见第六节）：端口 8097 vs 8101、`--mode paper` 被静默忽略（`main.py:329` 无 argparse）、"features-v1"、"BTC/ETH 训练"、"active_samples=0"（实际 269/480）、"156 倍冗余已修复"（只修了 `strategy_decision`）。
- **[S3] 可复现性缺口**：一次决策无法用 (bar_id, 模型权重 SHA, exit_policy 快照, risk_profile 快照, 特征向量) 完整复现；`_measured_horizon` 每次重启重算（`min_samples=8`）→ **同一根历史 K 线在不同时刻被不同的时间止损管理**；`processed_bars` 只有写入点没有读取点（面板永远是 `{}`）。

**测试空白**（`tests/` 共 60 个文件，实测无以下覆盖）：`evaluate_tick` / `evaluate_bar` / `run_decision_loop` **零测试**（两条决策路径的防重、tick/bar 一致性、跨会话竞态完全未覆盖）；`on_session_start` 失败回滚无测试；`Runtime.build + restore` 装配路径无集成测试；`session.end()` 绩效口径无测试；**无一条断言"live 路径与 train 路径对同一根 K 线产出相同特征行"**——这正是 S1 级 train/serve skew 能长期存活的原因。

---

## 三、横向对标：开源项目能教我们什么

完整调研报告见 `docs/quant_oss_landscape_report.md`（含各项目 star 量级与架构细节，**star 数为量级估计，未经在线核验**）。以下为提炼后的可落地结论。

### 3.1 撮合真实性分级（由真到粗）

| 层级 | 代表项目 | 关键能力 | 本系统位置 |
| --- | --- | --- | --- |
| L3 | [hftbacktest](https://github.com/nkaz001/hftbacktest) | L2/L3 队列位置、**行情延迟与下单延迟分离**、tick/lot 约束 | — |
| L2 | [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) / [LEAN](https://github.com/QuantConnect/Lean) | 撮合模型分解、OCO/条件单、保证金账户与资金费 | — |
| L1 | [vnpy](https://github.com/vnpy/vnpy) / WonderTrader | tick 级撮合 + 滑点/费率/pricetick | — |
| L0 | [freqtrade](https://github.com/freqtrade/freqtrade) / Jesse | 蜡烛级：次根开盘成交 + `custom_slippage` + 期货资金费 | ← **本系统在这里** |

freqtrade 明确文档化其回测的局限（`--eps` 位置叠加"结果无法在 dry/live 复现"、fee 未设时用交易所默认值等），这种**显式承认边界**的做法本身值得抄。本系统的 README 方向相反：把未验证的能力写成已完成。

### 3.2 最该借鉴的 10 条

| # | 来源 | 做法 | 本系统落地位置 |
| --- | --- | --- | --- |
| 1 | LEAN | 把 Fill/Fee/Slippage/Margin/Settlement 拆成可插拔模型类 | `app/trading/broker.py` + `app/trading/margin.py` 重构为 `execution/models/*` |
| 2 | Nautilus | 策略只发**订单意图**，`ExecutionClient` 可插拔（回测/实盘同构） | 抽 `ExecutionGateway`，`PaperGateway` 与未来的 `BinanceGateway` 同签名 |
| 3 | hftbacktest | 行情延迟与下单延迟**分离建模** + 队列位置 | 采集盘口前 N 档；限价单按队列模型判成交 |
| 4 | Hummingbot | `InFlightOrder` 订单状态机 + 幂等 `clientOrderId` | 建 `OrderState` FSM（NEW/PARTIAL/FILLED/CANCELED/REJECTED/EXPIRED），重启可续 |
| 5 | vnpy / ccxt.pro | 定期与交易所对账（订单/持仓/余额） | 现在 paper 无对账，`app/storage/reconcile.py` 只做 K 线缺口 |
| 6 | freqtrade（期货模式） | 按**真实**资金费与结算周期扣费 | 修 `app/trading/simulation.py:63-70`（先结算后置桶 + 按品种取 fundedIntervalHours） |
| 7 | FreqAI / [Qlib](https://github.com/microsoft/qlib) `RollingGen` | `train_period_days` / `backtest_period_days` 走步重训 | `app/models/retrain.py` 已存在但**未接线** |
| 8 | López de Prado 系 | PurgedKFold + embargo、Deflated Sharpe、PBO | `app/models/dataset_split.py` 只有单次时间切分；门槛需 BH 校正 |
| 9 | freqtrade protections + vnpy RiskManager | `StoplossGuard` / `MaxDrawdown` / `LowProfitPairs` / `Cooldown` + 下单规则插件 | 补 HWM 回撤熔断、连亏减速、品种熔断 |
| 10 | Qlib Alpha158 + LEAN 组合层 + [quantstats](https://github.com/ranaroussi/quantstats) | 因子库 + 组合构建层 + 标准化绩效报告 | 建 `alpha_crypto` 因子库；回测自动出 Sharpe/MDD/换手报告入库 |

另外两个值得注意的"反面教材"：[Gekko](https://github.com/askmike/gekko) 与 Zenbot 均已停维护，共同特征正是**蜡烛收盘价成交、无订单簿、无永续/资金费、无状态对账**——与本系统当前的撮合层级高度相似。

---

## 四、该拉什么数据去训练

> 端点名与路径来自调研，**未经逐条在线校验**，落地前请以官方文档复核。

### 4.1 免费且应当立刻补上的（按 ROI 排序）

| 数据 | 来源 | 历史深度 | 用途 |
| --- | --- | --- | --- |
| K 线（已有） | [data.binance.vision](https://data.binance.vision/) `futures/um/monthly/klines/` | UM 自 2019-09 | 特征+标签主源；比 REST 分页快一个量级 |
| **主动买卖量 + 成交笔数** | 同一份 K 线 CSV 里就有：`taker_buy_base_asset_volume`、`number_of_trades` | 同上 | **零成本**补齐订单流族——当前在 `app/storage/storage.py:63-66` 被丢弃 |
| 逐笔成交 aggTrades | `futures/um/daily/aggTrades/` | 同上 | CVD、大单占比、平均单笔额、成交笔数 z-score |
| 最优挂单 bookTicker | `futures/um/daily/bookTicker/` | 同上 | 价差、microprice 偏离 |
| **5 分钟衍生指标** | `futures/um/daily/metrics/`：`sum_open_interest`、`count_toptrader_long_short_ratio`、`sum_toptrader_long_short_ratio`、`count_long_short_ratio`、`sum_taker_long_short_vol_ratio` | 同上 | **回补 `/futures/data/*` 只有 30 天保留的限制，这是最关键的一条** |
| 资金费率历史（已有） | `/fapi/v1/fundingRate` | 自合约上线 | 已有；但需按品种取真实结算周期 |
| 深度快照 | `futures/um/daily/bookDepth/` | 有限档 | 粗粒度深度不平衡 |
| 期权 IV / DVOL | [Deribit API](https://docs.deribit.com/) `public/get_volatility_index_data` | 多年 | **免费无需 key**；IV-RV 价差、25Δ skew 作 regime 变量 |
| 恐贪指数 | [alternative.me](https://alternative.me/crypto/fear-and-greed-index/) `api.alternative.me/fng/?limit=0` | 2018 起 | 日频 regime，性价比极高 |
| 宏观（DXY/NDX/10Y/VIX） | FRED、Yahoo、[Stooq](https://stooq.com/) CSV | 多年 | 日频 risk-on/risk-off 状态变量 |

### 4.2 需要自建采集器、且**过期不可回补**的

**这是最紧急的一件事**：Binance `/futures/data/*` 全系列只保留 **最近 30 天**，今天不采集，历史永远补不回来。

| 端点 | 字段 | 决策价值 |
| --- | --- | --- |
| `/futures/data/openInterestHist` | sumOpenInterest / sumOpenInterestValue | ΔOI、OI-价格象限、OI/成交额 |
| `/futures/data/topLongShortPositionRatio` | longShortRatio | 大户**持仓**多空比（比账户数更接近真实仓位） |
| `/futures/data/topLongShortAccountRatio` | longShortRatio / longAccount / shortAccount | 大户账户情绪 |
| `/futures/data/globalLongShortAccountRatio` | longShortRatio | 散户拥挤度；与大户的差值 = Smart-Retail Gap |
| `/futures/data/takerlongshortRatio` | buySellRatio / buyVol / sellVol | 官方口径的主动买卖比（订单流代理） |

建议：5 分钟粒度定时落库，与 `derivatives` 表同源。**当前 `derivatives.open_interest` 实测全为 0**。

### 4.3 付费（现阶段不建议买）

[Tardis.dev](https://tardis.dev/)（L2/L3 全量 tick 回放，是订单簿策略回测的事实标准，但只在验证出 L2 alpha 之后才值得）、Kaiko / Amberdata / CoinAPI（跨所归一）、[Glassnode](https://glassnode.com/) / CryptoQuant（链上，日频 regime）、[Coinglass](https://www.coinglass.com/)（清算热力图）。**建议路径：先用免费数据把订单流+资金费+OI/LSR 这条线做出来，验证有 alpha 再买单月 Tardis 做严格回放。**

### 4.4 数据质量陷阱（逐条必须处理）

1. `/futures/data/*` 只保留 30 天；`/fapi/v1/openInterest` **无历史**。
2. K 线 `closeTime = openTime + interval − 1ms`；特征必须用已收盘 bar。
3. **资金费周期不统一**（多数 8h，部分 4h/1h）→ `annualized = rate × (24/间隔h) × 365`，不能一律 ×3×365。当前代码写死 8h（`app/market/funding.py:23 RECORDS_PER_DAY = 3`）。
4. 成交量口径：ticker 的 24h volume 是滚动值，K 线 volume 是区间值，`quoteVolume` 才是 USDT 额——不要混用。
5. 限频：`/fapi` 2400 weight/min/IP，429 指数退避、418 封 IP。
6. 合约生命周期：下架/改名、`onboardDate` 之前无数据、退市末段流动性失真 → 需要 **point-in-time universe**，否则幸存者偏差不可修正。
7. 强平流 `!forceOrder@arr` 不是全量（部分走 ADL/内部撮合）→ 强平量是**下界**。
8. 统一 UTC，禁止本地时区/DST 进桶。

---

## 五、该拉什么数据做决策（实时）

### 5.1 WebSocket 订阅清单（`wss://fstream.binance.com`）

| Stream | 频率 | 内容 | 优先级 | 现状 |
| --- | --- | --- | --- | --- |
| `<sym>@kline_1m` | 250ms | 未收盘 K 线 | 必须 | 已订阅（当前 5m） |
| `<sym>@markPrice@1s` | 1s | mark、index、**funding、nextFundingTime** | **必须** | 已订阅 |
| `<sym>@bookTicker` | 实时 | 最优买卖价+量 | **必须** | 已订阅（另一条连接） |
| `!markPrice@arr@1s` | 1s | 全市场 mark 数组 | 强烈推荐（替代逐订阅） | 未订阅 |
| `<sym>@aggTrade` | 逐笔 | 价格/量/`m`（买方是否 maker） | 推荐 | 已订阅但**未落库、未入特征** |
| `!forceOrder@arr` | 实时 | **全市场强平** | 推荐 | **未订阅** |
| `!miniTicker@arr` | 1s | 全市场 24h 统计 | 推荐（横截面因子） | 未订阅（现用 REST 每 5s） |
| `<sym>@depth@100ms` | 100ms | 增量深度 | 视策略 | **未订阅** |

**关键缺口**：已经订阅了 `aggTrade`（`websocket_feed.py:42`），但 `storage.py` 的 candles 表不存主动买卖量与笔数，也没有任何订单流特征消费它——**订阅了却没有用**。

### 5.2 单机规模经验值（需自测校准）

- 300–500 个合约同时订阅 `kline + markPrice + bookTicker + miniTicker` 是可行的；
- `aggTrade` 全量约 50–150 个合约；`depth@100ms` 全量约 20–60 个合约；
- 单连接上限 1024 streams，24 小时强制断开（必须有自动重连与回补）；
- JSON 解析是瓶颈，建议 orjson。

### 5.3 决策输入的性价比排序

1. **必须**：`markPrice@1s`（含 funding 与 nextFundingTime → 自算 basis 与「距下次结算分钟数」）、`kline`、`bookTicker`、5 分钟 OI/takerRatio（REST 轮询）。
2. **强烈推荐**：`!forceOrder@arr`（强平瀑布）、`!miniTicker@arr`（横截面动量）、`depth` top5（OFI / microprice）。
3. **加分**：aggTrade 全量（CVD/大单）、LSR 三件套（5min 拥挤度）、Deribit DVOL/skew（日频风险状态）。
4. **性价比低**：全市场全深度 L2、逐秒社交情绪、tick 级链上。

### 5.4 特征候选（精选，完整 55 条见数据调研）

按族给出**当前完全没有**的高价值项：

- **订单流/微观结构**：`trade_imbalance_1m`（按 `m` 字段算主动买卖失衡）、`cvd_z`、`large_trade_ratio`、`avg_trade_size_z`、`trade_count_z`、`spread_bps`、`microprice_dev`、`depth_imb_l20`、`ofi_top5`、`amihud_illiq`、`vwap_dev`。
- **资金费与基差**：真正的 `basis = mark/index − 1`（当前 `mark_basis` 是滞后动量，需重做）、`basis_z_7d`、`basis_mom_1h`、`annualized_funding`、`funding_term_struct`（当期−上期）、**`mins_to_funding`**。
- **持仓量与多空比**：`oi_chg_5m/1h/24h`、`oi_price_quadrant`（sign(ΔOI)·sign(Δprice)）、`oi_notional_z`、`oi_vol_ratio`、`top_pos_lsr`、`global_lsr`、`smart_retail_gap`、`taker_ratio`。
- **清算**：`liq_long_usd_5m`、`liq_imbalance`、`liq_z_24h`、`cascade_risk = 强平额/OI`、`mins_since_big_liq`。
- **波动率状态**：`rv_5m/1h/24h`、`rv_ratio`（波动扩张领先指标）、`yang_zhang`、`vol_of_vol`、`dvol_chg`、`iv_rv_spread`、`skew_25d`。
- **跨品种/因子**：`beta_btc_1h`、`resid_mom`（剔除 BTC 的残差动量）、`xs_mom_rank`（全市场排名分位）、`btc_dominance_chg`。
- **时间/季节性**：`hour_sin/cos`、`session_flag`（亚/欧/美时段）。
- **日级 regime（做交互项，不要指望日内 alpha）**：`fng_value`、`etf_net_flow`、`exch_netflow_z`、`risk_on_composite`。

**建议第一批做前 35 个**（订单流 + 资金费基差 + OI/LSR + 清算 + 波动率），全部可由 Binance 免费数据在 1–5 分钟粒度算出。

---

## 六、README 与代码不符清单（对照核验）

| README 声称 | 代码/产物实际 | 证据 |
| --- | --- | --- |
| "features-v1 特征" | `FEATURE_VERSION = "features-v3"` | `app/features/feature_spec.py:19` |
| "ema20/ema50/atr 是绝对价格量纲，山寨币属于越界输入" | 这三个键**不在** `FEATURES` 里，模型特征全是尺度无关量 | `app/features/feature_spec.py:13-18` |
| "候选 manifest 在 12bp 成本下 active_samples=0" | 在线加载的两个 manifest 是 269（lightgbm）/480（catboost）；`=0` 只出现在 `candidates_retired_v1/` | `data/research_v3/candidates/*/manifest.json` |
| "当前权重只由 BTC/ETH 训练" | 数据集 profile 含 **12 个品种** | `data/research_v3/training_dataset.jsonl.profile.json` |
| "含 isotonic 校准" | `calibration.json` 无生产者，`model_runtime` 恒返回 `calibrator_missing` | `app/models/model_runtime.py:10-12` |
| "持仓量历史 SQLite 表" | 无任何写入路径，实测 sum = 0.0 | `app/derivatives.py` 零实例化 |
| "156 倍冗余已修复" | 只对 `strategy_decision` 做了签名去重；`risk_decision`/`order_rejected` 无去重 | `app/strategy/decision_loop.py:301-311` |
| "按行数保留，更早的先进 archive 再删除，不丢数据" | 保留策略因 `AttributeError` 从未执行 | `app/strategy/decision_loop.py:360` + `app/runtime_context.py:65-89` |
| "回测复用实盘同类，可用于预测纸面账户" | 回测不接 meta-labeling 门槛、不接 Chronos、离场判定机制不同 | `app/backtest/backtest.py:174` |
| "被拒绝的品种仍在积累证据，可以自己挣回资格" | 重启后 `seed_from_history` 丢掉被拒品种的证据 | `app/main.py:61` + `symbol_edge.py:223` |
| `python -m app.main --mode paper` | `main.py` 无 argparse，`--mode` 被静默忽略 | `app/main.py:329` |
| 端口 8097（文档正文又写 8101） | 实际 8101 | `start_paper.bat:5` |

---

## 七、落地路线图

### P0 — 让结论重新可信（不改策略，只修测量）

1. **修 train/serve skew**：把 funding 取值下沉进 `snapshot()`（传 store+symbol+bar_time），live 与 train 共用同一函数；加一条断言——manifest 中训练方差 > 0 而当前输入为常数 → 直接落 `feature_degenerate` 并拒绝交易。
2. **加一条回归测试**：同一根 K 线，train 路径与 live 路径产出的特征行必须逐字段相等。这一条测试的价值高于本仓库现有 60 个测试文件的总和。
3. **修时间止损单位**：`interval_ms = ingest.interval_ms(settings.interval)`；`app/main.py:85` 的 `min_samples=8` 提到 `symbol_edge_min_samples`；horizon 选择加 bootstrap 下界 + 多重比较校正，不达标就不覆盖用户配置。
4. **修资金费结算**：先结算再置桶；rates 缺失时不推进并记 `funding_missed`；桶长按品种从 `/fapi/v1/fundingInfo` 取。
5. **修保留策略**：`Runtime` 加 `database_worker` 字段（或注入），`except` 改为写 `retention_failed` 事件。
6. **修强平**：接 `/fapi/v1/leverageBracket` 的真实 MMR 档位；强平计 `liquidation_fee_rate`；`cash < 0` 记 bankrupt 并冻结账户。
7. **统一风险参数来源**：启动时同时 `risk.apply_profile` 与 `limits.apply_profile`；启动断言 `target_exposure ≤ max_symbol_leverage ≤ max_gross_leverage ≤ session leverage`，越界拒绝启动。
8. **挂单风险预留**：`accept_open_fill` 内重新执行 risk/limits 校验（或 submit 时预留、撤销时释放）。
9. **修离场口径**：`mark()` 复用 `evaluate_bar_exit` 的 high/low 保守判定；`reason` 拆成 `stop_loss`/`take_profit`；回测与实盘共用同一条离场函数。
10. **对齐 README**：把与代码不符的条目全部改掉；端口、`--mode`、feature_version、数据集品种、active_samples 逐条修正。

### P1 — 补数据与特征（这一步才可能产生真实 alpha）

11. **立刻上 5 分钟衍生数据采集器**（OI / LSR×3 / takerRatio）——过期不可回补，优先级最高。
12. **存 K 线里现有的 `taker_buy_volume` 与 `number_of_trades`**（改 `storage.py` 表结构与 upsert，零外部依赖）。
13. **采集并落库 aggTrade**，生成订单流失衡、CVD、大单占比、成交笔数族特征。
14. 订阅 `!forceOrder@arr` 与 `!miniTicker@arr`；接 Deribit DVOL。
15. 把特征集从 10 个扩展到 35–50 个；引入波动率 regime 与 BTC beta/残差动量。
16. **建立 point-in-time universe**（symbol 生命周期表），重做数据集以消除幸存者偏差。
17. 标签改为从 `bar[index+1].open` 起算（或加显式延迟惩罚），并做敏感性分析。

### P2 — 策略与治理

18. 门槛改为 `mean > max(min_net_bps, z × SE)`，z≈2，并对多品种做 Benjamini-Hochberg 校正；同时维护"成交口径"（按 R 倍数）统计，两套同号才放行。
19. 补 HWM 回撤熔断（需人工复位）、连亏减速、波动率目标、单日交易次数上限、同向 beta 加权净敞口上限。
20. 接 `app/models/retrain.py` 与 `app/models/drift.py`；加模型回滚；生产模型过期时**拒绝交易**而非回落候选。
21. 修 `model_registry` 的 `feature_version` 判据并把 walk-forward 结果写成 `portfolio_oos`；让回测与实盘共用同一个模型构造工厂。
22. 按 LEAN 的模型分解重构撮合层；按 Hummingbot 的订单 FSM 重构订单生命周期；加定期对账。
23. 评估指标补 Sharpe/Sortino/MDD/换手与按波动率/时段的分层表现；`periods_per_year` 由 interval 推导。

---

## 八、本报告的局限

- **未运行任何代码**，全部结论来自静态阅读与产物检查（JSON manifest、profile、SQLite 统计由子代理执行）。文中已标注「已复核」的条目由主报告二次独立核验。
- **外部调研的链接与端点未经逐条在线校验**（子会话联网工具中断）：Binance `/futures/data/*` 端点名、data.binance.vision 各数据集路径、Tardis/Kaiko 等的当前报价与免费额度，落地前请复核官方文档。
- **`.env` 的 `BINANCE_WS_URL=wss://fstream.binancefuture.com` 未实测**：该主机名不在官方文档列出的生产端点中（官方为 `fstream.binance.com` 与 `stream.binancefuture.com`），子代理推测其可能指向非生产环境，但**未抓包验证**。若成立，则撮合用价（来自该 WS）与特征用价（来自生产 REST）属于两套价格体系混用——建议加一条启动自检：REST 与 WS 各取一次 BTCUSDT 最新价，偏差 > 50bp 拒绝启动。
- 子代理实测的模型相关系数（corr=0.808、符号一致率 99.77%）用的是数据集前 4 万行（对这两个模型属训练期），足以证明"成员同源"，但不是它们的样本外数值。
- 组合层面的相关性敞口、波动率目标等属**建议**而非缺陷判定——系统从未声称具备这些能力。


---

## 九、修复记录（本轮实施）

上述局限中「`.env` 的 `BINANCE_WS_URL` 未实测」一条已实测，并确认为**本报告严重度最高的问题**。

### 9.1 实测：市场数据来自测试网（原局限第 3 条，已由推测变为确证）

对 `BTCUSDT @bookTicker` 采样 20 秒：

| 站点 | 20s 消息数 | 中位价差 | 中位买一量 | 中位卖一量 |
| --- | --- | --- | --- | --- |
| `fstream.binancefuture.com`（`.env` 原配置） | 129 | 0.54 bps | 0.002 BTC | 353.8 BTC |
| `fstream.binance.com`（生产） | 10,645 | 0.01 bps | 2.156 BTC | 10.864 BTC |

同时 `stream.binancefuture.com` 的 update id 与前者同量级（4.26e11），而 `fstream.binance.com` 是 1.15e13，相差约 27 倍——前者是低频的测试网撮合引擎。价格中位偏离生产 −2.68 bps，最大 −5.41 bps。

结论：`BINANCE_BASE_URL`（生产 REST）与 `BINANCE_WS_URL`（测试网 WS）**属于两套价格体系**。全部特征、闭合 K 线与纸面成交来自测试网，而标记价、资金费与合约过滤器来自生产。深度成交模型读取的正是买一/卖一挂单量，而测试网这两侧相差 17 万倍，因此买卖两侧的成交规则完全不同。

修复：`.env` 与 `.env.example` 改为 `wss://fstream.binance.com`；新增 `app/market/venue_check.py` 与启动自检（`VENUE_CHECK_POLICY=warn|block|off`），静态主机分类永远执行、采样部分在 `warn` 下后台执行，采样不到数据时状态为 `unknown` 而非 `ok`。

### 9.2 已完成项对照

| 原报告条目 | 状态 | 落点 |
| --- | --- | --- |
| 实盘 4/10 特征为常数（train/serve skew） | 已修 | `app/features/feature_source.py` 训练与实盘共用；`tests/test_feature_parity.py` |
| 强平玩具模型（固定 MMR、不用强平费、允许负权益） | 已修 | `app/trading/margin.py` 分档 MMR + 逐仓强平价；`app/trading/simulation.py:settle_liquidation` |
| 资金费桶先推进后结算 | 已修 | `app/trading/simulation.py:apply_funding` 逐品种窗口、失败不消耗窗口 |
| 晋级门禁永不可通过 | 已修 | `app/models/model_registry.py` 比对 `FEATURE_VERSION`；加校准器门禁 |
| 审计保留策略从未执行 | 已修 | `Runtime.database_worker` + `main.py` 赋值；`events_archive` 亦加保留与回读 |
| `interval_ms` 兜底 300000 | 已修 | `app/core/config.py:interval_to_ms`；`symbol_edge` 分桶改用配置周期 |
| 重启污染 `session.account.trades` | 已修 | `trades_for_session` |
| 启动时 `risk` 与 `limits` 参数不一致 | 已修 | `PortfolioLimits.from_profile` |
| 离场只用最新价/mark price | 已修 | `mark()` 走 `evaluate_bar_exit`（OHLC），区分 stop/target 原因 |
| 优势门槛统计上无效（`mean > 0`，n=30） | 已修 | 标准误 + t 检验 + Benjamini-Hochberg FDR；`symbol_edge.statistics/q_values/allows` |
| 持有期 argmax 选择偏差 | 已修 | `horizon_analysis` 仅在单侧 95% 下界 > 0 的持有期中选择 |
| OOD 门禁结构性失效（边界 == `FEATURE_CLIP`） | 已修 | `app/models/dataset_io.py` 分位数边界 + `bounds_are_informative`；`/health` 暴露 |
| `below_min_notional` 分支不可达 | 已修 | 拒绝原因改为显式 if/elif |
| 标签从信号 K 线收盘价起算 | 已修 | `LABEL_VERSION=label-v2`，从 `bars[i+1].open` 起算 |
| `/futures/data/*` 无采集（30 天硬截止） | 已修 | `app/market/derivatives_collect.py` + `derivatives_detail` 表 |
| K 线订单流字段被丢弃 | 已修 | `parse_klines`/`_kline_rows` 保留 7/8/9 号位；`app/features/order_flow.py` 特征 |
| 挂单不占组合额度 | 已修 | `reserve_entry`/`fill_is_affordable` |
| 缺少 HWM 熔断/连亏/波动率目标/相关性敞口 | 已修 | `app/trading/risk.py`、`app/trading/concentration.py`、`app/trading/guards.py` |
| `events_archive` 无界且不可读 | 已修 | `Store.prune_archive` / `Store.archived_events` |
| `prometheus` 未接线、`/health` 过浅 | 已修 | `app/ops/health.py` 七项检查 + 计数器；`/ready` 区分就绪与存活；`app/ops/prometheus.py` 全量指标 |
| `DerivativesStream`、`backfill_klines` 死代码 | 已修 | 与 `derivatives_collect.py`/`ingest.py` 重复，已删除 |
| walk-forward purge/embargo、`portfolio_oos` 生产者 | 已修 | `app/models/portfolio_oos.py` 用同一 `PaperAccount`/`RiskEngine` 回放样本外预测；训练作业接线并写入候选 manifest |
| 价格档位/状态机分散在六个模块 | 已修 | `app/core/order_state.py` 单一状态定义；`EXPIRED` 与 `CANCELED` 分离 |
| 账户三套视图（持仓/现金/挂单）从不互校 | 已修 | `app/ops/account_reconcile.py` 定时比对并上报差异（原在 `app/backtest/`，策略层为用它反向依赖回测层，已移入诊断层） |

### 9.3 修复过程中发现的新问题

- `RiskEngine.restore()` 用**位置参数**构造 12 个字段。为加入 HWM 与连亏字段而在中间插入字段时，`target_exposure` 被读成 1.0 而快照里是 0.7，且没有任何报错。已改为全关键字构造，并把新字段追加在末尾。
- `Store.upsert_candles` 与 `Store._upsert_candles` 是两条独立的 INSERT 语句（实盘走批量路径、CLI 走单条路径）。加列时只改一条会静默丢列，已合并为一条。
- K 线订单流字段被丢弃的直接后果：OOD 边界对 `return_10` 等四个特征等于截断范围，因为 `app/features/features.py` 在写入数据集前已截断，边界取到的极值就是截断值本身。
- `main.py` 中 `limits` 在设置并发上限之后被 `PortfolioLimits.from_profile` 重新赋值，配置值被静默覆盖。已调整赋值顺序，并让 `apply_profile` 只在档位真的带有该字段时才覆盖。

## 十、第二轮修复（按优先级逐轮执行）

### 10.1 最重要的新发现：`mark_basis` 在 9.711% 的训练行上被截断

这是本轮唯一一个"用数据直接量出来"的缺陷，也是全部修复里影响面最大的一个。

**测量方法**：对已落盘的 `data/research_v3/training_dataset.jsonl`（578.6 MB，1,258,344 行，12 个品种）逐行统计每个特征落在 `FEATURE_CLIP` 边界上的比例。

| 特征 | 截断范围 | 触界行数 | 占比 |
| --- | --- | ---: | ---: |
| **`mark_basis`** | **(-0.02, 0.02)** | **122,201** | **9.711%** |
| `funding_z` | (-5.0, 5.0) | 576 | 0.046% |
| `rsi` | (0.0, 100.0) | 196 | 0.016% |
| `volume_ratio` | (0.0, 50.0) | 60 | 0.005% |
| `ema50_gap` | (-0.5, 0.5) | 16 | 0.001% |
| `ema20_gap` | (-0.5, 0.5) | 11 | 0.001% |
| `return_10` | (-0.5, 0.5) | 10 | 0.001% |
| `atr_pct` / `funding_rate` / `funding_carry_24h` | — | 0 | 0.000% |

`mark_basis` 的触界率是其余任意特征的 **211 倍**。截断机制本身没问题，问题出在这一个特征的定义上。

**根因链**（每一环都可验证）：

1. `app/market/funding.py:funding_features()` 里 `mark = marks[position]`，而 `marks` 与 `times` 共用同一个下标——`times` 是**资金费**的时间轴。
2. 资金费 8 小时结算一次：`derivatives` 表每个品种恰好 1,095 行（365 天 × 3 次/天），时间跨度 1757520000000–1789027200007。
3. 于是 `basis = mark / price - 1` = **"距上次结算以来价格涨跌了多少"**，而不是基差。行情 8 小时内动 2% 是常态。
4. `derivatives.mark_price` 只有 `funding.py:parse_funding()` 写入（`/fapi/v1/fundingRate`，8 小时）；`derivatives_collect.py` 造行时把 `mark_price` 硬编码为 `0.0`，且只写 `derivatives_detail` 表。

**修复**：

- `mark_series(store, symbol)` 从 `derivatives_detail` 读 5 分钟粒度的标记价：`open_interest_value / open_interest` 就是该桶均价。采集器**一直在取这两个数**，只是从来没人用它们做过这件事。表为空时回退到 8 小时行，老库行为不变。
- `funding_features(..., mark_series=(times, values))` 让资金费与标记价各走各的时间轴与二分查找。标记价必须**与时间轴成对传入**：只传时间轴、值仍用资金费的值，看起来对，实际错——第一版就是这么写的，被 `tests/test_mark_basis.py` 抓出来。
- `MAX_BASIS_AGE_MS = 15 分钟`：远高于采集器的 5 分钟节奏，远低于资金费的 8 小时节奏。超龄或缺失时 `mark_basis` 返回 `None`，**不是 0.0**——0 落在训练边界内部，下游无法把它和"真实的零基差"区分开。
- 无任何可见发布时四个资金费特征全部返回 `None`（原先返回 0.0），由 `FeatureSource` 填默认值并计入 `degraded_features`。

**顺带发现**：`tests/test_feature_parity.py::test_the_same_decision_trades_once_its_inputs_are_real` 声称在"输入健康"时门禁不应触发，但它的 fixture 用 `interval_ms=HOUR` seed 到按 1 毫秒步进的 K 线上，发布时刻落在最后一根 K 线之后 3,600,000 毫秒——**一个可见的发布都没有**，四个资金费特征全是占位零，测试一直在用占位零通过。已按 fixture 自己的时钟重设间隔。

---

## 十一、第 3–18 轮修复记录

> **本节及其后的 §15 是重建的。** 在一次追加写入时，工具返回的行数少于文件实际行数，写入把 §10.1 之后的 573 行覆盖掉了；仓库没有版本控制，也没有备份，这部分原稿无法找回。
>
> 原稿的 §12–§14 三节也在这批丢失里，它们记的是第 13–17 轮；其内容已折进下面的 §11.2–§11.4，没有另立小节，以免把复述写得像原文。
>
> 重建的依据是**留下的证据**，不是记忆：README 的第 14–74 行逐条记录了每一轮的实测数字与结论；每一个修复都在代码注释与测试名里留了它要防的那个偏差；本轮（§16）之前的会话记录还在。凡是原稿里的表格与数字，只保留能在这三处对上的；对不上的不写。措辞与原稿不同。

### 11.1 轮次与落点

| 轮 | 目标 | 主要落点 | 判定方式 |
| ---: | --- | --- | --- |
| 1 | 初次审查，建立可测量的基线 | `docs/paper-trading-ml-review.md` §1–§8 | 逐条复核代码 |
| 2 | P0：让结论重新可信 | `app/market/funding.py`、`app/features/feature_source.py`、`scripts/profile_dataset.py` | 在训练集上直接量 |
| 3–8 | P0 深化：测量、撮合、风控同源 | `app/features/quality.py`、`app/trading/execution.py`、`app/trading/margin.py`、`app/trading/risk.py`、`app/trading/broker.py`、`app/core/order_state.py` | 每个修复配一个命名了偏差的测试 |
| 9–12 | P0 收尾 + `/health` 重写 | `app/ops/health.py`、`app/market/derivatives_collect.py`、`app/market/venue_check.py` | 十项检查，`status` 取最差 |
| 13–16 | P1：校准器、组合样本外、晋升门禁接线 | `app/models/calibration.py`、`app/models/portfolio_oos.py`、`app/models/model_registry.py`、`app/models/training_job.py` | 端到端晋升测试 |
| 17 | P1：`mark_basis` 的定义与边界 | `app/market/funding.py`、`app/storage/storage.py`、`tests/test_mark_basis.py` | 9.711% 的截断行 |
| 18 | P1：把 order-flow 三个族接上 | `app/features/feature_spec.py`、`app/features/feature_source.py`、`app/features/order_flow.py`（见 §15） | 生产调用点数 = 0 |

### 11.2 第 2 轮：P0 与 `mark_basis`

第 2 轮在**训练集上直接量**出了一件事：`mark_basis` 有 **122,201 行（9.711%）压在 ±2% 的截断值上**，而其余任何特征最高只有 0.046%——211 倍的差距。原因是 `funding_features()` 里标记价与资金费共用同一个时间轴与同一个二分下标，而资金费 8 小时才结算一次，于是 `mark/price - 1` 算的是「距上次结算以来的涨跌幅」。

修复：`mark_series()` 从 `derivatives_detail` 读 5 分钟粒度的标记价（`open_interest_value / open_interest` 即该桶均价——采集器一直在取这两个数，从来没人用它们做过这件事）；`MAX_BASIS_AGE_MS = 15 分钟`，超龄或缺失返回 `None` 而不是 `0.0`，因为 0 落在训练边界内部、下游无法把它与「真实的零基差」区分。

同一轮还量出了 OOD 边界本身的缺陷：边界取每特征的 **min/max**，而数据写入前已被 `FEATURE_CLIP` 截断，于是任何触及过截断值的特征得到一条**恰好等于截断范围**的边界，服务侧输入又被同一套截断夹回该范围——这道门禁**永远无法触发**。改成 0.1%/99.9% 分位（直方图流式统计）。重跑 `scripts/profile_dataset.py`（578.6 MB / 1,258,344 行，约 24 秒）后，退化边界从 4 条降到 1 条，`return_10` 的门禁从 ±0.5 收紧到 ±0.039。

### 11.3 第 13–16 轮：门禁要求的每一种证据都要有生产者

晋级门禁要求的四类证据此前**都没有生产者**，所以没有任何模型可能被晋级，而 `MODEL_REQUIRE_PROMOTED=0` 让这件事静默发生：

| 门禁要求 | 此前的生产者 | 现在的生产者 |
| --- | --- | --- |
| 校准器已拟合 + artifact 文件存在 | 无 | `app/models/calibration.py` |
| 组合样本外证据（`costs_included is True`、`trades >= 30`、`net_return` 有限为正、`0 <= max_drawdown <= 0.2`） | 无 | `app/models/portfolio_oos.py` + `TrainingJob._portfolio_oos` |
| `feature_version` 与训练时一致 | `fit_model.py` 写死字面量 `features-v1` | 写 `FEATURE_VERSION` |
| `split.ready` + `split.label_intervals_verified` | 有 | 有 |

组合证据还漏了资金费：实盘每个 tick 对每个持仓调用 `account.apply_funding`，回放只算手续费与滑点，而 `costs_included` 是硬编码的 `True`——门禁读的第一个条件正是这个标志位。现在 `costs_included = bool(funding)`，没有费率数据时为 `False` 并附 `costs_missing`，`total_funding` 字段被**移除**而不是留成 `0.0`（0.0 无法区分「净额为零」与「没有建模」）。实测同一组 fixture 只改费率符号，`net_return` 从 1.486209 变到 1.488171。

### 11.4 第 18 轮的一句话

`app/features/order_flow.py` 能算出 20 个特征，而它在 `app/` 与 `scripts/` 下的**生产调用点数为 0**。详见 §15。

---

## 十五、第 18 轮：把「一直在采集、从来没有建模」的 20 个特征接上

> 本节同样是重建的。§15.9 是原稿，其余各节的措辞与原稿不同，数字与结论一致。

### 15.1 断在哪一环

| 环节 | 状态 |
| --- | --- |
| 交易所 `/futures/data/*` 端点 | 有 |
| 采集器 `app/market/derivatives_collect.py` | 有，定时写 `derivatives_detail` |
| 每分钟 flow 桶（aggTrade + forceOrder） | 有，写 `flow` 表 |
| K 线 7/8/9 号位（taker 买量、成交笔数、quote volume） | 有，`parse_klines` 保留 |
| 特征函数 | **有，20 个，带单测** |
| `FEATURES` 契约 | **没有这 20 个名字** |

### 15.2 接线时暴露的四个独立缺陷

**1. `oi_price_quadrant` 的价格腿永远是 0.0。** `positioning_features` 从 `derivatives_detail` 行的 `price_change` 列读价格方向，而写入方只写八列，schema 里也没有这一列——持仓量端点本来就不返回价格。实测把 20 个特征逐个检查「是否移动过」：19 个移动，它恒为 `{0.0}`。

**2. `since` 不是窗口。** `Store.flow` / `Store.derivatives_detail` 以 `ORDER BY event_time DESC LIMIT n` 结尾，只给下界时返回的是**最新的 n 行**。实测（1500 根 K 线，取第 49 根，`since = bar - 1h`、`limit = 240`）：只给下界返回 240 行、最早一行在第 1211 根之后（约 4.2 天）、可见 0 行；两端有界返回 49 行、可见 13 行。200 根 K 线的 fixture 表只有 200 行，`LIMIT 240` 取回全部，所以测试一直通过。

**3. 同一个 bar 的值取决于缓存块对齐。** 块缓存是必需的（150,000 行 × 3 次查询不可接受），但第一版只按「不晚于 bar_time」过滤：同一根 bar 在块开头读到 47 个 flow 桶、下一根读到 13 个，`flow_delta_z` 因此是 `-1.370238` 与 `-1.337793`。修复后窗口两端都有界，值只由 `(跨度, bar_time)` 决定。

**4. 块向前延伸在实盘是错的。** 实测同一段 80 根 bar：实盘语义 `reuse=False` 下与全历史读取不一致的 bar = 0；向前延伸 `reuse=True` 下 = **52**（第 61 根起）。实盘 12:00 取回的块被缓存到 15:00，里面没有 13:00 的行，`_visible` 会从陈旧的少数桶里取值——**得到一个错的数，而不是缺失的值**，后者会被上报，前者不会。

### 15.3 契约怎么改才不是一次停机

把 20 个名字直接加进 `FEATURES` 会让每个已加载 artifact 立即失效（它们声明 10 列）。这个判断混了两件事：「和当前常量一致吗」与「这段代码能产出它需要的列吗」。现在是 `FEATURE_SETS`（`features-v3` → 10 列、`features-v4` → 30 列）+ 服务侧按 manifest 声明的列取数（`ModelDecision.required_features()` 取各成员声明的并集）。实测两个线上候选仍正常加载，各 10 列。

诊断计数（`order_flow_bars`/`positioning_rows`/`flow_buckets`/`trades`/`measured`）**刻意不是特征**：它们描述「有多少数据可用」，在采集正常的训练集里是常数，正好是 OOD 门禁看不见的那种退化边界。

### 15.4 性能：313 到 5,136 行每秒

`build_dataset` 传 `ttl_ms=0`，而判据是 `now - cached_at < ttl_ms` → **恒为假** → 每行重读资金费与标记价序列。新增 `reuse=True`（离线专用）后：3000 根从 9.4 秒降到 0.6 秒；12000 根复测 2.34 秒，线性；退化列从 15 个降到 0 个。

### 15.5 报告方式的改变

构建器按**列名**而不是一个总数报告退化（`degraded_features`），因为「970 行是退化的」不说明该重建资金费历史还是等采集器。

### 15.6 本轮改动落点

`app/features/feature_spec.py`、`app/features/feature_source.py`、`app/features/order_flow.py`、`app/storage/storage.py`、`app/models/train.py`、`app/models/tabular_model.py`、`app/strategy/live_models.py`、`app/ops/health.py`、`app/models/fit_model.py`；测试在 `tests/test_feature_parity.py`、`tests/test_feature_scale.py`、`tests/test_health_endpoint.py`。

### 15.7 测试（本轮 +15，共 634，起点 619）

```
634 passed（本轮开始时 619：+8 个特征源与契约测试，+6 个 /health features 测试，+3 个 OOD 无边界列测试，另有两处旧测试被改写为它们本该断言的契约）
```

### 15.8 验证

两个线上候选加载后声明的特征列数仍为 10；`/health` 的 `features` 项在真实进程上给出 `all_inputs_measured` 或具体缺失列；`ood_gate` 在无边界列不属于任何已加载模型的声明时正确地不降级。

### 15.9 顺带：OOD 门禁对它没看过的列报 ok

改完契约后核对线上状态，发现 `ood_gate` 给出的是 `informative, checked: 10`。但 `data/research_v3/training_dataset.jsonl` 只有 10 列，而契约现在有 30 列——**另外 20 列根本没有边界**，`out_of_range` 无从测试它们，而 `checked: 10` 读起来像一个完整的答案。

这和第 2 轮「边界等于截断范围」是同一个形状：一个门禁对自己没检查的东西保持沉默，而报告里的数字看起来是完备的。现在 `unbounded` 集合会与**已加载权重实际声明的列**取交集，只有「有模型在读这一列，而没有任何东西能界定它」时才降级。实测当前状态：

```
profile: checked=10  unbounded=20  degenerate=['mark_basis']  verified=True
ood_check（当前加载 v3 权重）: degraded / unreachable_for:mark_basis
  -> unbounded 的 20 列不在任何已加载模型的声明里，因此正确地不触发 bounds_missing_for
```

两个待办因此显式化了：`mark_basis` 的边界仍然等于截断范围（第 2 轮重跑 profile 后从 4 条降到 1 条），以及 profile 覆盖的列比契约少 20 列。两者都要靠**重建数据集再重跑 `scripts/profile_dataset.py`** 解决，而重建数据集是破坏性动作（578.6 MB），仍然等确认。

新增测试 +3（`tests/test_health_endpoint.py`）：已加载模型没读无边界列时不触发；读了则降级并列出列名；没有 decisions 层时回退读各成员的声明。写第三个测试时发现回退路径读错了对象——`runtime.models` 是模型运行时而不是成员字典，`.values()` 抛 `AttributeError`，而**健康检查抛异常会把整份报告一起带下去**，所以那个测试最初表现为 `ood_gate` 这一项整个消失。

---
## 十六、P1 第 2 件：即时点（point-in-time）universe —— 算出来、存下来、打印出来，然后丢掉

### 16.1 断在哪一环

`app/features/universe_history.py` 是一个写得很认真的模块：文件头明确写着它要解决的两个偏差（幸存者偏差、前视偏差），有 13 个测试，有完整的三态报告。它被 `app/models/training_job.py` 的 `_point_in_time` 调用，结果存进 `job.point_in_time`，在日志里打印，写进 `training_runs`。

然后**没有任何一行代码用它做决定**。紧随其后的 `dataset` 阶段是这样开头的：

```python
job.stage = "dataset"
destination = dataset_path(self.settings.data_dir, job.tier)
built = await loop.run_in_executor(None, lambda: build_dataset(
    self.settings.data_dir, tuple(symbols), options["interval"], ...))
```

`symbols` 仍然是 `universe` 阶段从**今天的快照**里选出来的那一份。重建的结论只影响日志。这和第 15 轮 order_flow 的形状完全一样——模块存在、测试存在、调用点为零——区别是这一次调用点存在，只是返回值没人读。

### 16.2 去修的时候，发现它读到的数据本身就是错的

在把结论接上去之前先量了一下当前的真实输出。第一次测就出现了一个不可能的数：重建说数据在 **2026-01-20** 结束，而同一批 symbol 的最新 K 线是 **2026-09-10**，差了 234 天。

`app/storage/storage.py` 的 `candles_range`：

```python
def candles_range(self, symbols, start_ms, end_ms, interval=None, limit=500000):
    ...
    'SELECT * FROM candles WHERE %s ORDER BY open_time ASC LIMIT ?'
```

**默认 500,000 行，升序，所以超出时丢掉的是最新的部分。** 实测 14 个 symbol、366 天、5 分钟：范围内共 **1,391,212 行**（读取 7.7 秒，峰值分配 **1.64 GB**），其中 **891,212 行（64%）被静默丢弃**。重建拿到的 36% 恰好是**最旧**的那一段，于是它把「24 小时滚动窗口」量在了数据末端之前 234 天的地方，然后报告说**整个 universe 都停止交易了**。

这是第 15 轮 `Store.flow` 那个「`since` 不是窗口」的同一个形状：一个带边界的查询，边界不生效，也没有任何东西说它不生效。函数的 docstring 甚至写着「用无界 limit 逐个 symbol 问 N 次才是重建不可行的原因」——limit 是为了可行性加的，加完以后答案从「这个窗口」变成了「这个窗口最旧的 50 万行」。

### 16.3 三个修复

**一、范围读取分页到底，超预算就报错。** `candles_range` 改为 keyset 分页（`open_time > cursor` 而不是 OFFSET，因为多个 symbol 会共享 `open_time`，用 OFFSET 会漏行），`limit` 默认改为 `None`；传了 `limit` 而超出时**抛异常** `candle_range_over_limit:N>M`，而不是返回一个看起来完整的短答案。另一个调用方 `app/models/advanced_model.py` 的 portfolio 回放窗口本来就在 50 万行以内，因此行为不变，只是不再依赖那个巧合。

**二、重建不再需要整个范围。** 新增 `Store.candle_windows(symbols, moments, window_ms)`：每个 symbol 的**首个**和**末个** bar，加上每个时刻往前 24 小时的 bar。重建真正要问的就是这两件事，不是中间那 139 万行。配套 `extent_from_windows` 和 `series_from_windows`（后者按 close_time 去重——两个重叠窗口会把同一根 bar 的成交量算两遍，而量能门槛读的就是它）。实测：

```
改前：读 1,391,212 行（其中 64% 被丢），7.7 秒，1.64 GB，horizon 量到 2026-01-20
改后：两次有界读取，1.31 秒，horizon 量到 2026-09-11 09:24（与最新 K 线一致），feed 落后 0.40 天
```

**三、末端状态按「数据到哪」评估，不按墙上时钟。** 训练档的 K 线只在训练运行时刷新，所以两次运行之间它们会落后于实时行情。在 `now` 处重建，每个 symbol 都读成「已停止交易」，而这是一个关于**我们采集**的事实被报成了关于**市场**的事实。现在末端状态评估在 `extent["horizon"]`。

### 16.4 顺手拆开的三个「同一个缺失」

改的时候发现模块把三种完全不同的情况折叠成了同一种「不在 universe 里」：

| 情况 | 原来 | 现在 |
|---|---|---|
| 符号根本没有数据 | 不在 `graded` | `untraded_reasons[sym] = "no_data"` |
| 符号当时还没上市 | 不在 `graded` | `"not_listed_yet"` |
| 符号上市过，但滚动窗口内没有 bar | 不在 `graded` | `"no_bars_in_window"` |
| 重建产生了分类，但全被年龄/量能门槛排除 | `counts={"excluded":N}`，读起来像「市场是空的」 | `measurable=True`、`tradeable=False` |

`measurable` 与 `tradeable` 是两个问题：**重建跑出结论了吗**，和**有东西可交易吗**。空 counts 是前者失败，全 excluded 是后者失败，而调用方对这两者的反应完全相反。

### 16.5 结论接上去

新增 `TrainingJob._restrict(job, rows)`，在数据集加载之后、walk-forward 之前执行：把行裁到「当时在 universe 里」的 symbol，并且 `timestamp >= 窗口起点`。两种丢弃原因**分开计数**（符号未上市 vs 窗口比请求更长），一个总数会掩盖到底是哪一种在起作用。

四种**拒绝**各有各的名字，因为它们要四种不同的反应——去建数据集、去修取数窗口、去修采集、或者接受请求比数据更久远：

```
no_rows / no_start_universe / start_universe_unmeasured / nothing_tradeable_at_start
```

写测试时在这里抓到了**我自己刚写的一个 bug**：`graded` 里包含被排除的 symbol，所以把它当成白名单会让「全体被排除」的情况**一个符号都不裁、status 报 ok**——正好是这个方法存在的意义所在。改成只取 `tier != TIER_EXCLUDED` 之后，测试通过。

### 16.6 真实数据上的结论

在线上库（36.6 万行 K 线、15 个候选 symbol、`days=365`）实测：

```
days=365  1.31s  diagnosis=request_reaches_before_the_store   tradeable=False  at_start={'excluded': 12}
        restriction=skipped nothing_tradeable_at_start
days=330  1.30s  diagnosis=nothing_tradeable_at_start          tradeable=False  at_start={'excluded': 12}
```

`request_reaches_before_the_store` 的含义是：库里最早的一根 bar 就在取数边界上，因此窗口起点处每个 symbol 量出来都是一天新——**年龄门槛量的是我们的数据边界，不是上市时间**。判据用的是「数据是否**始于**边界」，不是一个天数阈值；先用一天做过阈值，结果因为库里最早的 bar 比移动的取数边界早了几分钟，诊断在 1.0007 天上翻面。

所以现在这台机器上，365 天的训练请求**不会**被裁剪，而它会明说为什么。这是正确的结果：重建还没能建立那个起点，而假装裁过了比不裁更糟。

### 16.7 本轮改动落点

`app/storage/storage.py`（`candles_range` 分页 + 超限报错、`candle_windows`）、`app/features/universe_history.py`（`exclusion_reason` / `data_horizon` / `extent_from_windows` / `series_from_windows`、`untraded_reasons` / `measurable` / `tradeable` / `feed_lag_ms`）、`app/models/training_job.py`（末端评估点、`_restrict`、`diagnosis`、`removed_with_current_feed` / `removed_with_stale_feed`）、`tests/test_universe_history.py`（+9）、另修掉 `app/models/training_job.py` 里 14 处 `\\`` 转义警告。

### 16.8 测试（本轮 +9，共 643，起点 634）

```
643 passed（本轮开始时 634：+9 个即时点 universe 测试）
```

新增的 9 个测试各自盯住一个偏差：feed 落后不等于退市、三种排除各有各的名字、全体被排除不等于空 universe、窗口读取只取需要的部分、重叠窗口不重复计量、列表时间在窗口够不到时仍然保留、重建的结论真的裁掉了行、失败的测量不许把数据集清空、以及「已解析」这一侧（防止 `diagnosis` 变成常量）。另有 1 个旧测试的空断言 `== [] or True` 被换成真断言——它此前无论实现对错都会通过。

---

## 十七、P1 收尾：universe 建不起来，不是市场的问题，是一列空的

### 17.1 上一轮留下的尾巴

第 16 轮把即时点重建接到了数据集上，线上实测却是 `days=365 → diagnosis=request_reaches_before_the_store`、`days=330` 也仍然 `@{"excluded": 12}`@。当时的结论是「库里的数据不够久」。

这一轮把那个数字追到底，结论不一样。

### 17.2 为什么每个 symbol 都被排除

把 `days=270`（此时年龄已够 96 天，远超 90 天门槛）的判定逐条打出来：

`@
  1000PEPEUSDT   excluded  age= 96.3d vol=   358088673 trades=        0  ['too few trades to fill']
  HYPEUSDT       excluded  age= 96.3d vol=   347127364 trades=        0  ['too few trades to fill']
  ARBUSDT        excluded  age= 96.3d vol=   101749127 trades=        0  ['too few trades to fill']
  RAYSOLUSDT     excluded  age= 96.3d vol=     5898210 trades=        0  ['24h volume below 10M', 'too few trades to fill']
`@

**`trades=0`，全部。** 量能和年龄都没问题，排除全部来自第三条判据。

### 17.3 那一列是空的

`@
trades             rows=2781239  zero_or_null=2781239 (100.0%)  total=None
quote_volume       rows=2781239  zero_or_null=2781239 (100.0%)  total=None
taker_buy_volume   rows=2781239  zero_or_null=2781239 (100.0%)  total=None
`@

**`"total=None"` 才是关键**：不是 0，是 NULL。而写入路径本身是好的——在同一台机器上新建一个库写一行，三个值都正确落库。也就是说，这 278 万行的 INSERT 语句里**根本没有这三列**。

于是 `symbol_facts` 里 `_quantity(row, "trades")` 对每一行都返回 0.0，求和是 0，门槛判定 `0 < 20000` → 排除。**一个从没被记录过的数，被当成了一次测量。**

这和第 2 轮 `mark_basis`、第 15 轮 `since` 不是窗口、第 16 轮 K 线只取最旧 50 万行是同一类缺陷的第四次出现：**边界/缺失没有表示法，于是在下游变成了一个具体的值**。

### 17.4 三个后果

**后果一：即时点 universe 永远建不起来。** 无论窗口取多长，每个品种都倒在同一条判据上，而报告的措辞是 `"too few trades to fill"`——**一个关于市场的陈述**。真实的陈述是「这一列没被记录过」。

**后果二：K 线订单流那 5 个特征全是占位值。** `order_flow_features` 的 `usable` 判据是 `"trades > 0"`，NULL 让 `usable` 为空，函数返回零面。线上实测 v4 契约：

`@
v4 features requested: 30  reported: 35
degraded (21): avg_trade_size_z, flow_delta_ratio, flow_delta_z, flow_largest_trade_z,
               flow_trade_intensity_z, global_long_short_ratio, liquidation_pressure,
               liquidation_share, liquidation_z, long_short_ratio_z, mark_basis,
               oi_change_1h, oi_change_5m, oi_price_quadrant, oi_z, quote_volume_ratio,
               smart_retail_gap, taker_buy_sell_ratio_z, taker_imbalance, taker_imbalance_z,
               trade_count_z
measured (14): atr, atr_pct, ema20, ema20_gap, ema50, ema50_gap, funding_carry_24h,
               funding_rate, funding_z, price, return_10, rsi, volume, volume_ratio
`@

**30 个特征里 21 个是缺的**，包括整整 20 个新接上的列和 `mark_basis`。这一环是对的（`FeatureSource` 早就把「族为空」转成 `None` 并逐列上报，`/health` 的 `features` 检查读的就是它），但在此之前没有任何东西把「这 21 列缺的是同一件事」说出来。

**后果三：OOD 边界与训练集都在描述一个不存在的分布。** `data/research_v3/training_dataset.jsonl` 的 10 列里，`"taker_imbalance"` 这类名字之所以不在其中，正是因为写它的时候这列还是空的。

### 17.5 修复：让「没测量」可以表示

这条链上有五个环节，每一环原来都把缺失压成 0：

| 环节 | 原来 | 现在 |
| --- | --- | --- |
| `ingest._number` / `_count`（短数组） | `0.0` / `0` | `None` |
| `storage._optional_float` / `_optional_int` | 写 0.0 | 写 NULL |
| `symbol_facts` | `trades` 求和为 0 | `trades` 为 `None`，附 `trades_measured` / `trades_rows_measured` |
| `universe.describe` / `classify_one` | `"too few trades to fill"` | `"unmeasured: trade count"` |
| `point_in_time` / `TrainingJob` | 排除计数 | `unmeasured_criteria` + `diagnosis=criteria_unmeasured` |

原来的注释写着「零成交笔数把它标记为未知，而不是一根真正安静的 K 线」——这句话本身就不成立：一根真正安静的 K 线**也是**零笔。两种情况的差别在**有没有记录**，不在值。NULL 是唯一能表达它的写法。

`quote_volume` 保留回退到 `volume × close`（对 USDT 本位合约是精确的），但新增 `quote_volume_source` 字段说明它是读来的还是算出来的。

### 17.6 順带：拒绝要说清楚能支持多长

一个只说「不行」的拒绝不是可执行的。窗口能被建立的上限是个数：数据能回溯多久，tier 在窗口起点需要多少天年龄，差值就是最长能建立的窗口。

`@
tier=speculative days=365  diagnosis=request_reaches_before_the_store  max_window_days=274
   -> the store reaches back 366 days and the speculative tier needs 90 days of age at the
      window start, so the longest window it can establish is 274 days; ingest older candles
      or shorten the window
tier=speculative days=270  diagnosis=criteria_unmeasured  unmeasured={'trade count': 12}
   -> tier criteria decided by an empty column: trade count x12; these symbols were not
      excluded on what they are, but on what was never recorded
tier=mainstream  days=365  diagnosis=request_reaches_before_the_store  max_window_days=-365
`@

`mainstream` 的 `-365` 是对的：该档要 730 天年龄，库里只有 366 天，**这一档在当前库上不可能被建立**，负数就是这个意思。

`_restrict` 的拒绝理由也从 `nothing_tradeable_at_start` 改成沿用重建的诊断：被一列空的排除和被年龄门槛排除，结果一样、下一步相反（重新灌数据 vs 缩短窗口）。

### 17.7 本轮改动落点

`app/market/ingest.py`（`_number` / `_count` 返回 None）、`app/storage/storage.py`（`_optional_float` / `_optional_int` 写 NULL）、`app/features/universe_history.py`（`trades` 可空 + `trades_measured` + `quote_volume_source` + `unmeasured_criteria`）、`app/features/universe.py`（`describe` 保留 None、`classify_one` 命名未测量判据、`_below_count` 辅助）、`app/models/training_job.py`（`diagnosis=criteria_unmeasured`、`max_window_days`、`_restrict` 理由沿用）。

### 17.8 测试（本轮 +5，共 648，起点 643）

`@
648 passed（本轮开始时 643：+1 个 NULL 贯穿 store 的测试，+3 个「未测量 ≠ 稀疏」与 quote_volume 来源的测试，+1 个拒绝理由沿用诊断的测试；另改写 1 个把旧语义钉死的测试）
`@

被改写的是 `tests/test_data_collection.py::test_kline_parsing_tolerates_a_short_array`：它此前断言短数组解析出的 `quote_volume == 0.0 and trades == 0`——**把「没测量」钉死成了 0**。现在断言两者都是 `None`，并另加一条断言真正的 0 仍然是 0。

---

### 17.9 仍然要做的（数据操作，非代码）

1. **重新灌 K 线**：278 万行里的三列是 NULL，只有重新拉取才会填上。在此之前 `trades` 相关的门槛与特征都只能是「未测量」。
2. **重启服务**：`derivatives_detail` / `flow` 表在运行中的进程里不存在。
3. **重建数据集**（578.6 MB，破坏性）：契约是 30 列而数据集是 10 列，profile 也只有 10 列的边界。

## 十八、P2 第一件：三个「已经装好、从来没有接线」的开关

P2 的清单里，「治理」这一类的共同形状不是缺功能，是**功能已经在跑、判断已经在做、结论没有任何人读**。这一轮修了三个。

### 18.1 `DRIFT_POLICY=block` 是一个不接线的开关

`app/models/drift_monitor.py` 的 `blocks_entries()` 写在 `DriftMonitor` 上，注释是「the decision layer can refuse on」。**调用它的只有 `app/ops/health.py` 和三个测试**：

`@
$ grep -rn blocks_entries app/ tests/
app/models/drift_monitor.py:139:    def blocks_entries(self):
app/ops/health.py:165:        blocked = bool(monitor.blocks_entries())
tests/test_drift_and_retrain.py:86:   assert monitor.blocks_entries() is False
`@

即：运营商把 `DRIFT_POLICY` 设成 `block`，`/health` 会报 `drift_blocking_entries`、面板上写着「正在拒绝入场」，**而入场一次都没被拒绝过**。一个只写一行事件的策略比没有策略更糟——因为设置了它的人相信拒绝正在发生。

修复：`app/strategy/decision_loop.py` 新增 `_drift_entry_check(runtime)`，接进共用入场门（和 `guard.approve_new_entry` 同一位置），拒绝理由码是 `feature_drift:<列名>`，进入 `order_rejected` 事件与 `rejections_total` 指标。三条边界写清楚：

- 没有 monitor、或 monitor 还没有结论 → **放行**（不能因为一次没跑过的检查停掉会话）；
- `policy=warn`（默认）→ 只记录，不拒绝；
- monitor 抛异常 → **放行并记 `errors.note`**，不允许一个坏掉的监控悄悄停掉交易。

### 18.2 drift 与 retrain 是两条平行线

`app/training_policy.retrain_required()` 的输入全是**关于产物**的事实：多旧、哪个数据集、哪个特征契约。没有一条是关于市场的。模型可以是一小时前训的、在最新的数据集上、契约也对，却仍在回答一个已经不存在的分布的问题——**而这正是 drift monitor 每小时测出来、并且没有任何人消费的那个东西**。

修复：`retrain_required(..., drifted=None)`，drift 判定排在三条产物判据**之前**（它是四条里唯一关于现在的），理由码 `feature_drift:<列名>`；`RetrainScheduler` 增加注入式 `drift_fn`，在 `evaluate()` 时**现取**而不是构造时快照：

`@
scheduler                  drifted_features()  evaluate()
drift_fn 未配置             None                走产物判据（"没检查" ≠ "干净"）
monitor 尚未出结论          None                走产物判据
monitor status=ok/drift     [...]               feature_drift:x
`@

`/health` 的 `drift` 检查同时增加 `retrain_on_drift` 字段，并在 drift 已发生、retrainer 存在、却没接 drift 时报新状态 `feature_drift_unactionable`——**「发现了但没人会行动」以前读起来和「系统已响应」一模一样**。

### 18.3 每日交易次数上限：所有风控都在管「多大」，没有一条管「多少次」

`risk.py` 里每一条限制回答的都是「这一笔最多亏多少」。没有一条回答「这个东西一天能开几次火」。而 `decision_loop` 的注释自己记着：单个品种**十秒内 37 次往返**、净亏。日内亏损熔断在换日时故意宽容，所以「许多笔小亏」这条路径对当时引擎里的每一条上限都是隐形的。

新增 `RiskEngine.daily_trade_limit` / `trades_today`：

- 检查在 `approve()` 内，**计数在 `submit_entry()` 的提交成功处**。理由：`approve()` 对随后被组合上限、执行就绪、lot step 拒绝的计划同样会跑，在那里计数会让「系统说 no 越多、预算消耗越快」——一个会自我收紧的上限；
- 计数随 UTC 换日重置（和日内亏损预算同类），高水位熔断**不**随之重置；
- 进 `snapshot()` / `restore()`：忘记当天已用多少，就等于每次部署重启一次预算，而跑飞最可能就发生在部署时；
- 默认 `MAX_DAILY_TRADES=60`（宽裕的兜底，不是策略约束），0 关闭。

### 18.4 顺带查出的真缺陷：换日逻辑有两份，已经不一致

加每日计数时，测试直接失败：换日**从不发生**。原因是 `approve()` 里有两条并行实现——给了 `event_time_ms` 的一条（**生产永远走这条**）和第二份 `_roll_day()`：

`@
旧 event_time 分支: day_key=key; day_start_equity=equity
                    if halt_scope != "high_water_mark": halted=False        # 只清 halted
旧 _roll_day():     day_key=key; day_start_equity=equity
                    if halt_scope != "high_water_mark":
                        halted=False; halt_reason=""; halt_equity=0; halt_threshold=0   # 清四个
`@

两份已经漂移：event-time 分支只清 `halted`，把 `halt_reason`/`halt_equity`/`halt_threshold` 留在那里描述**昨天**。于是日内熔断触发之后，第二天每一次拒绝都会带上昨天的理由码，面板上写着账户处于 halted 而它并不是。

修复：两份合并为 `_roll_day(equity, key=None)`，两条路径都走它，返回是否真的换日。**如果只把交易计数加进其中一份，上限会在第一天之后永久锁死**——这正是测试逼出来的。

新增的 `test_a_stale_halt_reason_does_not_outlive_the_halt` 钉住这条。

### 18.5 本轮改动落点

`app/strategy/decision_loop.py`（`_drift_entry_check` + 入场门 + `risk.record_entry()`）、`app/models/training_policy.py`（`drifted` 判据）、`app/models/retrain.py`（`drift_fn` 注入）、`app/main.py`（`drift_report` 接线、`risk.daily_trade_limit`）、`app/trading/risk.py`（每日计数 + 两条换日路径合并）、`app/core/config.py`（`MAX_DAILY_TRADES`）、`app/ops/prometheus.py`（三个新指标）、`app/ops/health.py`（`retrain_on_drift` / `feature_drift_unactionable`）。

### 18.6 测试（本轮 +6，共 658，起点 652）

`@
658 passed（+4 个 drift 治理测试，+2 个换日一致性测试；新增 tests/test_daily_trade_budget.py 6 个）
`@

其中 `test_a_blocked_drift_verdict_actually_refuses_entries` 覆盖了 policy=block / warn / 无 monitor / 无结论四种组合，`test_the_scheduler_reads_the_monitor_rather_than_a_snapshot` 覆盖了 drift_fn 的三种状态，`test_zero_disables_the_cap_and_the_budget_rolls_with_the_date` 同时断言了「计数随换日重置」和「高水位熔断不随换日重置」——两者相反，写在同一个测试里才不会被各自单独改坏。

---

## 十九、P2 第二件：从「已晋级的过期模型」静默回落到「被门禁拒绝的权重」

### 19.1 链路是完整的，直到最后一跳把它丢掉

注册表这一侧其实做对了：`current()` 区分 `active` 与 `expired`，`load_production()` 对过期抛 `ValueError('production_model_expired')`。然后：

`@
model_registry.load_production()      -> ValueError("production_model_expired")
  model_runtime.load_calibrated_model -> {"status": "fallback", "reason": "...expired"}
    live_models._load_production()    -> self.promotion = {...}; return   # 没有 models
      live_models.load()              -> promoted = False
                                       -> self._load_candidates()   ← 这里
`@

`load()` 里那一跳的判据是 `if not promoted`，**只有一个条件**。于是「注册表里根本没有晋级模型」和「有晋级模型，但它过期了/校准器丢了/文件读不出来」走的是同一条路：静默加载**晋级门禁拒绝过的**候选权重。

从外面看，这两种状态的 `mode` 都是 `candidate`、面板都在报「真实候选权重（未通过晋级门禁）」，服务照常交易。**从「已批准但过期」降级到「已被拒绝」，是一次provenance的降级，穿的是「可用性」的外衣。**

而 `provenance_problem()` 也接不住：它以 `if not self.runtime.models: return None` 开头——候选加载完之后 `@models` 非空，于是它返回 None；唯一能兜住的 `require_promoted` 默认是 `0`（这是对的，本仓库从没有模型晋级过）。

### 19.2 修复：把「没有」和「有但不可用」分开

`model_runtime.py` 新增 `classify_reason()`，把加载失败分成两类：

`@
absent    no_production_model                     ← 回落到候选是对的
unusable  production_model_expired                ← 回落到候选是错的
          production_model_missing
          calibrator_missing / calibrator_invalid
          (以及任何不认识的失败)
`@

**不认识的失败归到 unusable**：「生产模型加载失败，原因我不认识」的安全读法不是「悄悄换上从未被批准的权重」。

`live_models.load()` 改为：production 存在但不可用时 **不加载任何东西**，`mode='unavailable'`，`load_errors['production']='production_unusable:<reason>'`。`_fallback()` 在 `rule_fallback=False`（默认）下产出 FLAT，即**拒绝交易**——正是路线图第 20 条要的。

运营方想在这种情况下用未审核权重，可以显式 `MODEL_ALLOW_CANDIDATE_FALLBACK=1`。**默认关闭**：没人应该靠意外拿到它们。

`unavailable_reason()` 把具体原因带进被拒信号的 `reason_codes`：以前无论空注册表、模型过期还是文件损坏，审计里都是同一句 `model_unavailable`，而三种情况该做的事完全不同（训一个 / 重训那个 / 修文件）。

### 19.3 顺带查出的第二个真缺陷：/health 从错误的对象上读晋级状态

`model_check()` 里 `getattr(runtime, "promotion", None)` —— **`Runtime` 没有 `promotion` 这个字段**（它在 `runtime.models.promotion` 上）。实测：

`@
外层 Runtime（生产实际形状）:  {'detail': 'trading_unpromoted_weights', 'mode': None, 'version': None}
直接传 ModelRuntime:          {'detail': 'tradable', 'mode': 'active', 'version': 'v1'}
`@

**一个正确晋级的模型，/health 报的是「在交易未晋级权重」，`mode=None`。** 同一类缺陷：状态从错误的对象上读、拿不到就退化成一个看起来合理的值。

修复 `_model_runtime(runtime)`：认两种形状（服务把 model runtime 挂在 `Runtime.models` 上，测试直接传 model runtime），并从**真正持有它的对象**上读 `promotion` 与 `@models`。修完三种状态各自如实上报：

`@
已晋级    -> ok        tradable
已过期    -> degraded  production_unusable:production_model_expired   mode=unavailable
无 runtime-> skipped   no_model_runtime
`@

「没有模型运行时」与「有运行时但什么都没加载」原本都会被读成 SKIPPED（「未配置」），而后者是一个被要求交易却做不到的服务。

顺带把 `promotion` 里的 `created_at` 补上：`provenance_problem()` 有一段 `manifest_created` 的年龄判据读的就是它，而这个键从来没有被写过，那段代码一直不生效。

### 19.4 本轮改动落点

`app/models/model_runtime.py`（`ABSENT_REASONS`/`UNUSABLE_REASONS`/`classify_reason`、返回 `kind`）、`app/strategy/live_models.py`（`load()` 的拒绝分支、`allow_candidate_fallback`、`unavailable_reason()`、`_unavailable_reason()`、`promotion` 带 `kind` 与 `created_at`）、`app/core/config.py`（`MODEL_ALLOW_CANDIDATE_FALLBACK`）、`app/main.py`、`app/ops/health.py`（`_model_runtime` + 三态上报）。

### 19.5 测试（本轮 +7，共 665，起点 658）

新增 `tests/test_model_provenance.py` 7 个，全部用**真实注册表**而不是桩：`register` 一个通过门禁的模型、`promote` 之后把 `expires_at` 改到过去，再断言 `current()` 报 expired、`load_calibrated_model` 报 `kind=unusable`、`RealModelRuntime.load()` 的 `mode` 是 unavailable 且 `@models` 为空、`/health` 报 `production_unusable:production_model_expired`。

另外两条守住边界不被过度收紧：`test_the_refusal_can_be_waived_explicitly`（显式开关打开后不再是拒绝拦住的）与 `test_an_empty_registry_still_falls_back_to_candidates`（空注册表仍走老路——回落不是被删掉，只是被收窄到它本来要处理的那一种情况）。

---

## 二十、P2 第三件：订单状态机 —— 一张从来没被应用过、而且用不了的转移表

### 20.1 表是对的，只是没人用；而且用不了

`app/core/order_state.py` 的开场白写着：

`@
This module is the single definition. The transition table is data rather than control
flow so it can be asserted against, and so an illegal transition is a returned reason
instead of a state the rest of the system has to cope with.
`@

实测：

`@
$ grep -rn "order_state.transition" app/ tests/
(无输出)
`@

`transition()` **在整个仓库里没有被调用过一次**。真正在改状态的是三处直接构造：

`@
app/trading/broker.py:102   cancel()           -> PaperOrder(..., status or CANCELED, ...)
app/trading/broker.py:146   process()          -> PaperOrder(..., status_for_fill(...), ...)
app/trading/execution.py:332 _fill_from_depth() -> type(order)(..., status, ...) + track_order()
`@

于是转移表是**文档**，不是规则：任何状态都能被写进去，写进去之后账本和审计里看起来和合法转移一模一样。

### 20.2 更深一层：这张表对 `PaperOrder` 根本无法使用

把转移表接进去的时候，测试立刻报错：

`@
dataclasses.FrozenInstanceError: cannot assign to field 'status'
`@

**`PaperOrder` 是 frozen dataclass**，而 `transition()` 的实现是 `order.status = status`。

也就是说 `transition()` 对**这个模块唯一为之存在的类型**根本跑不通。它之所以一直没被发现，是因为它一次都没被调用过——一个连一个订单都套不上去的生命周期表。原有测试用的是一个可变的桩类 `Order`（`self.status = status`），所以测试是绿的。

**「测试通过」在这里恰好是问题的一部分**：测试用一个真实系统里不存在的形状验证了一个真实系统里不生效的函数。

### 20.3 表本身也是错的：它禁止了系统里最常见的那个转移

`PENDING: (OPEN, REJECTED, CANCELED)`，而：

- `PENDING` 在 `OPEN_STATUSES` 里，broker 因此把它当作**可操作**订单，`process()` 会去成交它；
- `status_for_fill()` 返回 `FILLED` 或 `PARTIALLY_FILLED`，**两个都不在表里**；
- 可成交的订单是立即成交的，所以「提交后直接成交」是系统里最常发生的那一个转移。

实测：`can_transition(PENDING, FILLED) -> False`，`transition(PENDING 订单, FILLED) -> illegal_transition:PENDING->FILLED`。

**先把表接进去、不先修表，会把每一笔 pending 订单的第一次成交变成拒绝。** 这一条是顺序问题，不是程度问题。

顺带：`PENDING` 这个状态**没有任何代码路径能产生**（`submit()` 用的是 `OPEN`），却出现在 `OPEN_STATUSES`、存储查询和重启恢复里。

### 20.4 修复

**表**（`app/core/order_state.py`）：

`@
PENDING:          OPEN, PARTIALLY_FILLED, FILLED, CANCELED, EXPIRED, REJECTED
OPEN:             PARTIALLY_FILLED, FILLED, CANCELED, EXPIRED, REJECTED
PARTIALLY_FILLED: PARTIALLY_FILLED, FILLED, CANCELED, EXPIRED
终态:              ()
`@

`PENDING -> FILLED/PARTIALLY_FILLED` 是必需的：交易所不承诺先确认再成交。`PARTIALLY_FILLED -> PARTIALLY_FILLED` 是「又一笔部分成交」。

**`transition()`** 改成对 frozen dataclass 用 `dataclasses.replace` 重建、对可变对象原地赋值，并接受 `changes` 携带同一次移动的其它字段（成交数量、时间戳）。同一个状态**带**变化是合法移动（进一步部分成交），**不带**变化才是迟到 book 事件产生的空操作。

**唯一应用点**：`PaperBroker.advance(order, status, timestamp, filled_quantity)`。合法性判定与重建**都在 `order_state` 里**，broker 只负责调用与计数——把表抄进 broker 正是让两条成交路径对「订单允许变成什么」产生分歧的原因。三处直接构造全部改走它，包括 `execution._fill_from_depth()`（按深度成交的那条路径此前完全绕过了表）。

**被拒绝的转移要计数**：`broker.transition_refusals`，进对账报告的 `order_transition_refusals`，有值时加入 `findings`（`order_transition_refused`，`consistent=False`），并导出为 `paper_order_transition_refused{reason=...}`。

成交路径上的拒绝是**账本与交易所的分歧**，不是统计量：返回这笔成交会给账本记下一笔订单上看不到的交易，静默丢弃则让账户持有无人开仓的仓位。所以 `process()` 与 `_fill_from_depth()` 在此时返回空并给出原因。

### 20.5 本轮改动落点

`app/core/order_state.py`（表修正 + `transition(changes)` + `_rebuilt()`）、`app/trading/broker.py`（`advance()` 唯一应用点 + `transition_refusals`）、`app/trading/execution.py`（深度成交路径改走 `broker.advance`）、`app/ops/account_reconcile.py`（拒绝进 `findings`）、`app/ops/prometheus.py`（`order_transition_refused`）。

### 20.6 测试（本轮 +11，共 676，起点 665）

新增 `tests/test_order_lifecycle.py` 11 个。几条值得单独说：

- `test_a_pending_order_may_fill`：**先钉住表允许 `PENDING -> FILLED`**，因为这是接线之后第一个会炸的地方；
- `test_the_depth_path_and_the_quote_path_agree`：用 `inspect.getsource` 断言 `_fill_from_depth` 里出现 `broker.advance(` 且不再出现 `PaperOrder(`/`type(order)(`，并断言整个 `PaperBroker` 里只剩 **1** 处 `PaperOrder(`（`submit` 创建订单；每一处移动都走 `order_state.transition`）。这是一条防回归的**结构性**断言——第二个构造点就是第二条绕过表的路；
- `test_a_repeated_partial_fill_is_still_written`：同状态、数量变大必须写；同状态、数量不变才是空操作。首版把两者都当空操作，会把成交丢掉；
- `test_the_broker_refuses_an_illegal_transition_instead_of_writing_it` 与 `test_an_unknown_status_is_refused_rather_than_replaced`：非法转移返回 `(None, reason)`、**账本不动**、拒绝被计数。未知状态那条用 `dataclasses.replace` 造出一个损坏订单（因为 `PaperOrder` 是 frozen 的）——这也正是 20.2 那个缺陷的来源；
- `test_reconciliation_reports_a_refused_transition`：一条被拒的转移会让对账 `consistent=False`，并出现在 `/metrics` 上。

写测试过程中我自己错了三次（`cancel()` 对终态订单返回订单本身而非 None、`PaperOrder` 不可变、broker 里的构造点现在是 1 处不是 2 处），三次都是**测试先于代码说出了真相**。

---

## 二十一、P2 第四件：撮合层分解 —— 一个不看数量、也不看流动性来源的成交模型

### 21.1 一个函数同时做了三件事，于是两个假设错了都看不出来

`broker.execute()` 里连着三行：算价格、算滑点、算手续费。LEAN 把这三件事拆成 FillModel / FeeModel / SlippageModel / MarginModel / SettlementModel，正是为了让每个假设**能被单独陈述**。这里是一个函数，于是下面两个错误共用了同一段代码，谁也没暴露谁。

### 21.2 缺陷一：成交量完全不进成交价

`slip = price * slippage_bps / 10000` —— **常量**。实测旧代码：

`@
qty            notional   fill price   slippage
0.001          50         50011.00     10.0000
1.0            50011      50011.00     10.0000
100.0          5001100    50011.00     10.0000
10000.0        500110000  50011.00     10.0000
`@

买单 50 USDT 和买单 **5 亿 USDT**，成交价一模一样（相对 50001 的卖一各滑 10.00）。**一个假设自己能吃掉任何数量的纸面账户，没法报出容量上限——因为它没有上限。**

### 21.3 缺陷二：每一笔成交都是 taker

`Fill` 上早就有 `liquidity` 字段：`execute_depth()` 把它**硬编码**成 `taker`，`execute()` 走默认值。于是一笔**挂单成交**被记成「吃掉了价差」，并按 taker 费率收费。交易所标准档 maker 0.0200% / taker 0.0400%：**这笔手续费收了一倍**，而且因为字段写着 taker，下游没有任何东西能发现。实测（修之前）：`LIMIT fill -> fee 0.199996 liquidity taker`，而 49999 × 0.01 × 0.0002 = 0.099998。

### 21.4 缺陷三：实盘的成交成本模型是构造函数的默认值，.env 里改不了

`app/main.py` 的 `PaperBroker(order_sink=..., order_ttl_ms=...)` 不传 `fee_rate` / `slippage_bps`，而 `config.py` 里**根本没有对应的配置项**（`grep FEE_RATE app/core/config.py` 无输出）。实盘每一笔成交都用硬编码的 0.0400% taker + 2bp 定价，且**任何一段历史 session 都无法用它假定的成本来解释**。

### 21.5 修复：app/trading/fill_models.py

`@
FeeModel(maker_rate, taker_rate)             fee(notional, liquidity) / rate(liquidity)
FillModel(spread_bps, impact_coefficient_bps, reference_notional)
                                             slip_bps(liquidity, participation, notional)
                                             price(quote, side, liquidity, participation, notional)
participation(quantity, available)           -> share | None
`@

**两个加性项，因为成因和量纲不同**：

- `spread_bps`：穿越价差的成本。任何主动单都要付，**与数量无关**；挂单被吃掉时为 0。
- `impact_coefficient_bps`：市场冲击，按**参与率**表达——本单吃掉了盘口可用流动性的多大比例。吃掉整个盘口移动系数那么多；吃掉四分之一移动一半（**平方根**，标准经验形式）。参与率未知时退回用名义价值比 `reference_notional`；两者都未知时**按系数全额收取**，而不是假装这单很小——**「不知道多大」不是「很小」**。

**挂单成交既不过价差、也不计冲击**：`slip_bps(liquidity=maker) == 0`。挂在盘口被吃掉的单子按自己的价格成交，它付出的代价是逆向选择（adverse selection），这个模型看不见，也不假装看得见。

**谁的流动性**：LIMIT 单走到 `process()` 就是「挂过」的，成交时是 maker；MARKET 单是 taker；唯一能把 LIMIT 变成 taker 的是「到达即成交」，而这只有在调用方给出与建单同一时刻的 timestamp 时才可知。这个判断写在代码里而不是埋在别处。

**标签跟着费用走**：`liquidity` 与 `participation` 现在写进 `Fill`。此前标签算出来了、用它选了费率、然后在构造 `Fill` 时**丢掉了**——审计里每一笔都写着 taker，而其中一半收的是 maker 费率，**记录和它自己旁边的数字互相矛盾**。

### 21.6 实测（修之后）

`@
=== 成交价随数量变化，impact_coefficient_bps=10 ===
qty        notional       fill price   slippage   bps
0.001      50             50011.20     10.2000    2.04
1.0        50016          50016.00     15.0000    3.00
10.0       500268         50026.80     25.8000    5.16
100.0      5006100        50061.00     60.0000    12.00

=== 挂单成交（maker 费率 0.0002）===
liquidity maker  fee 0.099998      # 此前 0.199996，且标着 taker
=== 到达即成交的 LIMIT ===
liquidity taker  fee 0.200036
=== 按深度成交 ===
filled 3.0 @ 101.3000 fee 0.121560 liquidity taker participation 0.5000   # 6 个挂单里吃掉 3 个
`@

`impact_coefficient_bps` 默认 **0**，即默认行为与之前完全一致（`test_zero_impact_keeps_the_previous_behaviour` 钉住这一点）；配置项 `TAKER_FEE_RATE` / `MAKER_FEE_RATE` / `SPREAD_BPS` / `IMPACT_BPS` 让成本假设第一次可以被陈述和记录，`snapshot` 新增 `cost_model`。

### 21.7 本轮改动落点

`app/trading/fill_models.py`（新增）、`app/trading/broker.py`（`__post_init__` 建模型、`execute()` / `execute_depth()` 走模型、`process()` 判定流动性、`cost_model()`）、`app/core/domain.py`（`Fill.participation`）、`app/core/config.py`（四个成本配置项）、`app/main.py`（实盘 broker 显式传模型）、`app/backtest/snapshot.py`（`cost_model`）。

### 21.8 测试（本轮 +12，共 688，起点 676）

新增 `tests/test_fill_models.py` 12 个。其中：

- `test_the_fill_price_responds_to_the_order_size` 直接钉住 21.2 那个数字（参与率 1.0 → 12bp，0.25 → 7bp）；
- `test_an_unknown_size_is_not_treated_as_a_small_one` 钉住「未知 ≠ 很小」；
- `test_a_maker_fill_pays_neither_spread_nor_impact` 与 `test_a_resting_limit_fill_is_a_maker_fill` 钉住 maker 语义与手续费减半；
- `test_zero_impact_keeps_the_previous_behaviour` **钉住旧行为可复现**（`50_011.0` 与旧公式逐位相同）——一个不能复现旧行为的成本模型没法用来对比；
- `test_participation_is_unmeasurable_rather_than_zero`：`participation(1.0, 0) is None`。返回 0 会被读成「这单没吃掉盘口任何东西」，那是**乐观**答案；None 才是事实。

修的过程中测试抓到我三次：流动性判定用「timestamp 晚于建单时刻」在**没传 timestamp 时**（最常见的情况）会退化成 taker，把挂单成交按穿越价差定价；`Fill` 构造时漏传 `liquidity`，导致费用按 maker 收、标签写 taker；以及测试自己残留的一段脚手架。第一和第二个都是**真实的、会进入审计记录的**错误。

---

## 二十二、P2 第五件：晋级门禁只比较一个字符串，而它比较错了方向

### 22.1 门禁检查的是「版本名」，不是「输入契约」

`model_registry.promote()` 的相关判据只有一行：

`@
if manifest.get('feature_version') != FEATURE_VERSION:
    raise ValueError('incompatible_feature_version')
`@

这一条在两个方向上都是盲的。实测（注册一个通过其它全部门禁、`feature_version` 填当前版本、但 `features` 里写了本代码根本算不出来的列名）：

`@
promote(bad features) -> ACCEPTED as bad
declared features: ['price', 'not_a_real_feature']
`@

**一个声明了自己读不到的列的模型，因为版本字符串对得上，直接晋级。**

反过来同样成立：只要版本字符串不等于「当前代码产出的那个」，一个**完全可服务**的旧版本产物也会被拒绝——这等于宣布「特征契约永远不许再扩展」，而契约分版本的全部意义就是允许扩展。

### 22.2 更糟的一半：这个洞会把「退化检查」本身关掉

服务侧 `ModelDecision.required_features()` 做的是**交集**：`tuple(name for name in FEATURES if name in declared)`。实测：

`@
declared not_a_real_feature           -> required 0 []
`@

**`required_features()` 返回空元组，而空的需求列表会把退化检查整个关掉**——那个「拒绝模型去读谁也填不上的输入」的门禁，被它本该拦下的那个输入**关掉了**。而且 `/health` 的 `features` 检查看的是 `feature_source`，它照样报 `all_inputs_measured`。

一句话：**一个读不到任何自己训练时用过的列的模型，会让系统报告「一切正常」。**

### 22.3 修复（两半）

**门禁侧**（`app/models/model_registry.py`）：

`@
declared = manifest['features']                              # register() 现在写进 manifest
unknown  = unproducible(declared)                           # 本代码算不出来的列名
  -> unproducible_declared_features:<names>
known    = features_for(manifest['feature_version'])        # 这个版本本代码还认不认
  -> incompatible_feature_version                            # 不认
  declared ⊆ known ?                                         # 声明的列必须在该版本契约内
  -> features_outside_declared_version:<names>               # 越界
`@

`register()` 把 `features` 写进 manifest：门禁要判断「声明的契约」，manifest 里就得有它。此前 manifest 只有 `feature_version` 一个字符串，**门禁想比也没得比**。

四个方向实测：

`@
v4 + 真实 v4 列                     -> PROMOTED
v4 + 算不出来的列                   -> refused: unproducible_declared_features:not_a_real_feature
v4 + 只声明 v3 列（子集）           -> PROMOTED
v3 如实声明 v3                      -> PROMOTED
v3 声明一个 v4 才有的列             -> refused: features_outside_declared_version:taker_imbalance
未知版本 features-v9 / 无版本       -> refused: incompatible_feature_version
不声明任何列                        -> PROMOTED（运行时按「契约全集」处理）
`@

**「空声明」与「声明了算不出来的列」是两回事**：前者是「输入清单未知」（运行时保守地要全集），后者是「读了我给不出的东西」。门禁只拒绝后者。

**服务侧**（`app/strategy/live_models.py` + `app/ops/health.py`）：声明非空、但与契约的交集为空时，记录 `unusable_features` 与 `load_errors['features']`，`/health` 立刻报 `degraded / declared_features_unproducible` 并列出列名。

### 22.4 顺带：我自己误报了一次

第一次实测时我把 `price` 也当成「算不出来的列」。查证后发现 **`price` 根本不是被建模的特征**（它是 30 列之外的价格字段，`snapshot()` 里用于成交与显示）。`unproducible(["price"]) == ["price"]` 是**正确**行为。记在这里因为「查证前的怀疑」和「查证后的结论」都值得留下。

### 22.5 本轮改动落点

`app/models/model_registry.py`（`features` 进 manifest、两个新判据、`features_for`/`unproducible` 导入）、`app/strategy/live_models.py`（`unusable_features` + 空交集上报）、`app/ops/health.py`（`declared_features_unproducible`）。

### 22.6 测试（本轮 +10，共 698，起点 688）

新增 `tests/test_feature_contract.py` 10 个，覆盖上表每一种情况（含「不声明任何列仍然可晋级」「旧版本如实声明仍然可服务」「越界列被拒」），以及服务侧的三条：空声明 → 全集、部分声明 → 取交集、**交集为空 → 上报而不是关闭门禁**（`test_a_declaration_that_intersects_nothing_is_reported_not_ignored`，这条直接断言 `features_check` 返回 `degraded` 并带 `unproducible` 列名）。

写测试时又错了一次：`_decision()` 夹具没有把 decision 挂到 runtime 上，而健康检查是从 runtime 上取 decision 的——**测试夹具与生产装配不一致**，这正是本报告里反复出现的同一类缺陷。

