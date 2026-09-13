# 模拟交易与交易决策缺陷审计 · 数据与决策改造方案

> 审计对象：`D:\BaiduNetdiskDownload\money`（Binance USDT-M 永续 · 纸面模拟 · ML 决策）
> 审计时间：2026-09-12 ｜ 方法：读代码 + 在真实库、真实权重上跑数复现
> 本报告全部标注「实测」的数字，都是在生产库 `data/research.sqlite3` 与
> `data/research_v3/` 的真实权重上跑出来的，不是估算。

---

## 0. 结论摘要

系统在**工程纪律**上远超一般个人项目（8.6 万行 Python、124 个测试文件、每个修复都带实测证据）。
但**模拟的真实性**与**决策的有效性**这两件事，各有一个可被证明的致命断裂：

| # | 问题 | 证据 | 后果 |
|---|---|---|---|
| **A** | **训练报告评的是一套，线上跑的是另一套** | 同一 188,640 行留出集：训练报告称 active=269、净 +2.08bp；线上门槛在 188,521 行上下注，净 **−13.95bp/笔，t=−102.5** | 12bp 成本过滤器变成了全时段双向下注 |
| **B** | **模型方向退化：99.3% 输出 SHORT** | lightgbm 在留出集 188,640 行中只有 0.74% 的预测为正 | 「集成投票」退化为单边动量过滤，无方向 alpha |
| **C** | ~~当前实盘 100% FLAT，且是双重死锁~~ **（已定位并修复，见 0.5.2）** | 真实原因是一个 **5 秒时钟窗口 bug**：`now_ms`（收到响应后取）减 venue 时间戳 ≈ 请求耗时，而 `premiumIndex` 实测耗时中位 **5185ms**，稳定超出 5s → 每行都被丢弃 → `derivatives` 表 47 小时未写入 → funding 四个特征永远降级 → 一票否决全部决策 | 修复后 `feature_degraded` 100% → **0%**，`derivatives` 恢复为分钟级新鲜 |
| **D** | **最该采的数据基本没采** | OI/多空比只存了 **2.92 小时**（一个月窗口的 0.04%）；flow 表 **0 行**；279 万根 5m K 线只有 **0.10%** 有成交笔数 | v4 契约 30 个特征里 **15 个依赖从未采集的数据** |
| **E** | **模拟没有延迟、没有队列、bar 内路径假设不完整** | 实测成交发生在 K 线收盘后 **122 秒**（中位） | 与 5m bar 中位波动（5.6bps）同量级 |
| **F** | **权益指标数字荒谬** | `/api/state` 展示年化波动 **220%**，全量曲线时 **45,000%** | 绩效面板无法用于任何判断 |
| **G** | ~~跨账户盈亏直接相加~~ **（已修复，见 0.5.10）** | 会话起始资金由前端任填，实测混有 100 与 10000；`all_time_realized_pnl=−697.29` 是两者之和 | 改为按各账户本金折算：`lifetime_return_pct = −13.79%`；起始资金加区间校验 |
| **H** | ~~行情断线被误判为品种不好~~ **（已修复，见 0.5.9）** | `without_market_data` 在 `book_seen_at` 为空时淘汰**全部**品种，62–85 秒一轮；`resolve()` 只对持仓品种打分，换币即丢弃 pending 预测 → `symbol_edge` 恒无样本 | 传输存活改为显式参数；修复后 8 分钟内重选 **0** 次 |
| **I** | ~~`manual` 看似是唯一赚钱的退出方式~~ **（已澄清，见 0.5.11）** | 16 笔 `manual` 中 9 笔是**会话停止时的市价标记**（`end()` 把会话结束原因当成了平仓原因） | 两者分开记录；94 笔历史**全部无 `reason_detail`**，故无法评价当前退出策略 |

---

## 0.5 【线上实测】2026-09-12 14:50 运行态体检

对**正在运行的服务**取了一次快照（`/api/state` + `data/research.sqlite3`），
不是回测、不是估算。结论：**进程是活的，决策链是断的**。

### 0.5.1 WebSocket 断了 5.4 小时，但系统靠 REST 兜底、指标全绿

```
connector_health: binance.market  connected=false  reconnects=8  events=0
last_error: ClientConnectorError(fstream.binance.com:443, OSError(22,'信号灯超时时间已到',121))
market_event 最新一条：09-12 09:30:17  → 距今 5.4 小时
```

WebSocket 重连 8 次全部失败，**一条行情事件都没收到**。但决策链**没有停**：
`app/strategy/decision_loop.py:618` 用 REST `/fapi/v1/klines` 取 K 线，实测新
鲜（当前会话 5 个品种的 K 线全部是 **1.9 分钟前**），`bar_time` 也是新鲜的
（90 秒前）。所以这里**不是**故障——兜底路径工作正常，是设计如此。

真正的问题是**这个兜底是静默的**：`connected=false` 只出现在 `connector_health`
的嵌套字段里，`error_count=0`、`last_error=null`、面板上没有告警。系统连续
5.4 小时不能收到推送数据（意味着**没有盘口、没有逐笔、没有 bar 内实时价**），
而所有健康指标显示一切正常。**「降级运行」和「正常运行」在可观测性上无法区分。**

### 0.5.2 `feature_degraded` 100% 命中，且只用 4 个特征就否掉了全部决策

最近 6 小时 436 条决策，拦截原因统计（一条决策可带多个 reason_code）：

```
feature_degraded        1430 次（328%，即平均每条 3.3 个特征缺失）
real_model_ensemble      436 次（100%）
out_of_distribution      160 次（ 37%）
agreement_below_floor    142 次（ 33%）
edge_below_floor          62 次（ 14%）
symbol_edge_below_floor   53 次（ 12%）
```

缺失的特征**每次都是同样 4 个**，且**全部来自 funding / mark price**：

```
feature_degraded:funding_rate   feature_degraded:funding_z
feature_degraded:funding_carry_24h   feature_degraded:mark_basis
```

**而 `feature_degraded` 是一票否决**：缺任何一个特征，决策直接 FLAT，不管模型
多看好。`edge_bps` 的中位数是 **+20.90bp**，85.8% 为正——模型一直在给出方向性
观点，但 100% 被一个与收益无关的网关拦掉。从「模型准不准」的角度看这 6 小时，
**等于什么都没测到**。

#### 这 4 个特征为什么永远缺失：一个 5 秒的时钟窗口

这不是「行情断了所以缺数据」。根因是一个**单位/时钟错误**，可完整复现：

```python
# app/market/pumps.py:28-30
stamp = int(row.get('time', 0))          # venue 生成响应时的时刻
if not 0 <= now_ms - stamp <= 5000:      # now_ms 在【收到响应之后】才取
    continue                             # 于是 now_ms - stamp ≈ 这次请求的耗时
```

实测这个网络下 `/fapi/v1/premiumIndex`（一次返回 900 个品种）**耗时中位 5,185 ms**：

```
连续 6 次实测：5242 / 5087 / 5105 / 5081 / 5185 / 5266 ms
通过 5s 门槛的次数：0 / 6
```

请求耗时稳定地**略微超过** 5 秒窗口，所以**每一行都被 `continue` 丢掉**，
`pending` 永远是空的，`record_derivatives` 一次都没被调用过：

```
derivatives 表最新 event_time = 09-10 16:00  →  47.0 小时前
```

而 `funding_series()` 读的正是这张表（`app/market/funding.py:133`），
`derivatives_detail` 里没有 `funding_rate` 列。**所以 funding 系列永远是空的，
`funding_rate / funding_z / funding_carry_24h / mark_basis` 四个特征永远降级，
全部决策永远 FLAT。**

这个 bug 的形状值得单独记一笔：它**不是「网络慢」**，而是**把「传输耗时」误当成
「数据年龄」**。5 秒这个阈值本身是合理的（mark price 5 秒前算新鲜），错的是拿
「响应到达的时刻」减去「响应生成的时刻」当作年龄——这个差值永远是往返延迟，
与数据新鲜度无关。**网络越快，这个 bug 越隐蔽**：在 50ms 的网络里它永远不会触发。

顺带一提，这个 bug 只在**慢网络**上暴露，而它掩盖了一个更基本的事实：
即使修好，`mark_price` 也只能每 60 秒落一次盘（`pumps.py:56`），
而 `mark_basis` 要求 mark 年龄 < 15 分钟，勉强够用。

#### 修复与验证

判据从「响应到达时刻 − venue 时间戳 < 5s」改为「venue 时间戳与**请求发出前**的本地
时钟之差 < 60s」，即`MARK_STAMP_SKEW_MS`——它衡量的是**时钟偏移**，与链路快慢无关。

```
修复前：derivatives 最新 = 09-10 16:00  →  47.0 小时前
修复后：derivatives 最新 = 09-12 15:04  →   1.7 分钟前   （行数 13140 → 14724）

feature_degraded 命中率：100% → 0%
  （最近 6 分钟 20 条决策中，四个 funding 特征无一降级）

剩余拦截原因（这些是真实门槛，不是故障）：
  out_of_distribution 12   agreement_below_floor 9
  edge_below_floor 8       insufficient_edge_samples 2
```

修复后仍有约 5 分钟的**冷启动窗口**：新建的 funding 序列从 15:01 开始，
而当时最后一根已收盘 bar 是 15:00，`bisect` 返回 −1（bar 之前没有任何记录），
四个特征仍是 None。这是正确行为而非缺陷——回看历史 bar 时不该用到未来的
funding——下一根 bar 起自动恢复。实测 15:05 之后 `bisect=2`，四项全部有值。

**这个 bug 的价值不在于它本身，而在于它揭示了「100% FLAT」是被误诊的。**
在它修好之前，所有关于「模型不准」「死锁」的判断都建立在一个虚假前提上：
那 6 小时里模型**从未被真正询问过**。修好后才第一次看到真实的拦截分布。

### 0.5.3 死锁确认（对应结论摘要 C）

```
feature_degraded 拦下全部 → symbol_edge.observe() 永不触发
→ symbol_edge 永远 0 样本 → SYMBOL_EDGE_REQUIRE_SAMPLES=1 恒拒
```

审计里写的双重死锁**在线上完整复现**。最近 6 小时 428/436 条 FLAT，仅有 8 条
`candidate=true`——它们越过了 feature_degraded，但死在 `symbol_edge_below_floor`。

### 0.5.4 【根因】没有开仓的真正原因：训练域与会话选币域不相交

修好换币（§0.5.9）与 funding 采集（§0.5.2）后，会话不再被误判，`feature_degraded`
归零，但**仍然一笔未开**。逐层追下去，是两个独立的缺陷叠在一起。

#### 缺陷一：审计轨只记录"判决"，不记录"提案"，导致 edge 门槛永远重建不出证据

`symbol_edge.allows()` 在样本数 < 30 时一律拒绝（`require_samples=1`），而样本来自
`observe()` ——它记录的是**主模型提出的方向**（`proposed`）。但审计行里写的是
**过闸后的 `side`**，被拦下的品种因此全是 `FLAT`。

`seed_from_history()` 恰恰是从审计轨重建证据的，于是：

```
1296 条历史决策 → 1238 条 FLAT + 58 条方向
→ 只重建出 1 个品种、19 个样本
→ 其余 17 个品种 0 样本 → 永久 insufficient_edge_samples
```

而要让一个**新**品种从 0 攒到 30 个样本，需要它连续存活 33 根 5m K 线 = **165 分钟**，
在这 165 分钟里它一直是被拒的。会话每 2–3 分钟换一批币（§0.5.9），根本活不到那时。

修复：

1. `live_models.py` 在 signal 上增加 `proposed_side`，并在闸门改写 `side` 之前记录；
2. `execution.decision_payload()` 把它写进审计行，同时补上一直缺失的 `market.bar_time`
   （此前恒为 `None`，`seed` 只能退化成用事件墙钟时间当参考点，落点在 bar 中间）；
3. `symbol_edge._proposed_from_votes()` 对**旧行**从 `decision.votes` 的符号和反推方向，
   使已有历史不必丢弃。

效果：重建样本 **1 个品种 → 18 个品种**，其中 **5 个达到 30 样本门槛**；
`insufficient_edge_samples` 从"拦截全部"降为 0。

#### 缺陷二：模型训练在 12 个主流币上，会话却在交易小币

`source=gainers`（涨幅榜）选出的品种，**没有一个**出现在训练集里：

```
训练集品种 : BTC ETH SOL BNB XRP DOGE ADA LINK LTC DOT AVAX TRX（12 个）
会话选中   : 龙虾USDT LSKUSDT VTHOUSDT BEATUSDT GRIFFAINUSDT
交集       : 空集
```

服务端 `ood_policy=block`，边界来自训练集特征极值，于是必然拦截：

```
atr_pct 训练上界 0.014844，会话品种实测 0.027 / 0.034（超出 2 倍）
→ out_of_distribution，5 个品种全灭
```

**这个拦截是正确的**，错的是会话把模型没见过的市场放了进来。实测把选币换成
`source=trained` 后，12 个训练内品种的 `atr_pct` 等 10 个特征**全部在界内**（越界数 0），
OOD 拦截随即消失。

为此新增 `symbol_not_in_training` 独立原因码：一个"从未训练过"的品种不是分布漂移，
只说 `out_of_distribution` 会把人引去看特征边界，而真正的问题是选币越出了训练域。
并把 `app/web/web.py` 的默认 `source` 从 `gainers` 改为 `trained`。

#### 缺陷三（尚未解决）：训练内品种的 edge 小于成本，训练外品种 OOD 过不去

换成 trained 之后 OOD 消失、agreement 达到 1.00，**唯一剩下的拦截是 `edge_below_floor`**，
而它是真实的：

```
品种类型        样本    |预测| 中位数    扣 12bp 往返成本
训练内(12 主流)   55       1.08 bp          −10.92 bp  ✗
训练外(小币)     445      22.62 bp          +10.62 bp  ✓
```

即：**模型能覆盖成本的品种它没见过，它见过的品种覆盖不了成本。**

- 主流币 5m 波动小，可预测幅度天然只有 ~1 bp，远低于 12 bp 成本；
- 小币幅度 20+ bp 有余量，但落在训练分布之外，OOD 正确地拒绝；
- 模型训练特征是 features-v3 的 **10 个**（`manifest.json` 已确认），
  而 `FEATURES` 常量已是 **30 个**（features-v4），两者不一致；
- `measured_horizon` 显示 pooled 净收益在 6/12 根 K 线才转正（+1.16 / +1.73 bp），
  仍低于成本，说明**当前标签视野与成本结构不匹配**。

结论：这不是门槛配置问题，是**模型能力与数据覆盖问题**，无法靠调参解决。出路是
§6 路线图的 P1 #11（用 30 特征 + 小币样本重训），而它仍卡在历史数据回补。

### 0.5.5 【实测证伪】换视野、换品种池都救不了：当前特征没有可交易的 edge

§0.5.4 发现「训练内品种 edge < 成本」后，自然的假设是「把视野从 15 分钟挪到 4 小时」。
**这个假设被实测证伪。** 记录全过程，因为中途我自己犯过一次方法错误。

#### 第一步（错误）：全样本分十档，得出「4h 有 +8.79bp」

把测试集全部行的特征合并排序、切十分位，做多最低组、做空最高组，结果：

```
f_ema20  h=240m  净 +8.79bp  命中 54.5%  t=17.84   ← 看起来成立
f_mom12  h=240m  净 +7.28bp  命中 53.6%  t=16.49
```
**这是前视偏差**：跨整个测试集排序，等于用未来信息决定哪些样本属于「低组」。

#### 第二步（正确）：每个时间戳内横截面排序

改成在**同一时刻的品种之间**排序 —— 这才是可以真实交易的口径：

```
特征        视野    截面数   组差bp    净bp    调整后t
f_ema20     1h     31375    1.39   -10.61     1.36
f_ema20     2h     31375    2.05    -9.95     1.04
f_ema20     4h     31375    1.72   -10.28     0.46
f_ema20     8h     31375   -0.70   -12.70    -0.10
f_mom12     4h     31374    0.96   -11.04     0.26
f_ret10     4h     31372    0.80   -11.20     0.21
```

**没有任何一档为正。** 组差本身只有 1–2bp，成本 12bp，差一个数量级。

#### 第三步：模型本身也没有方向能力

用现有 10 特征、880,704 行训练、377,496 行测试（含 1 小时禁运期），
按 `|预测|` 分档（净收益 = `sign(pred) × 实际收益 × 10000 − 12`）：

```
|预测|档      样本      占比     命中率    净均值bp      t值
0-2       224099   59.36%    49.7%    -11.91   -115.81
2-5        87362   23.14%    48.8%    -13.07    -61.71
5-10       37087    9.82%    49.0%    -13.09    -32.44
10-12       6153    1.63%    51.0%    -12.13    -10.79
15-20       5344    1.42%    47.7%    -13.42     -9.41
20-30       4794    1.27%    50.2%    -11.48     -6.77
30-50       3509    0.93%    49.6%    -10.02     -4.45
>50         3204    0.85%    51.2%     +0.77      0.27
```

**所有档位命中率 47.7%–51.2%，即模型没有方向识别能力。** 唯一为正的最高置信档
（+0.77bp，t=0.27）与 0 无法区分。门槛扫描同样全负：门槛 5bp 净 −11.92，
门槛 40bp 净 −2.15，**提高门槛只减少亏损，不产生盈利**。

#### 第四步：换高波动品种池，仍不显著

按日波动率分组，各训一个模型（24 个品种中取波动最高 / 最低各若干）：

```
组别              视野   门槛   净bp    t原    t非重叠   命中率
低波动(主流币)      1h     10   -14.81  -10.45   -3.72    47.9%
低波动(主流币)      4h     10   -16.50  -13.84   -1.58    48.0%
高波动(小币)        1h     50   +15.98   +2.04   +0.22    52.0%
高波动(小币)        4h     50   +26.81   +3.20   +1.44    50.7%
```

小币表面为正在，但**重叠样本把 t 抬高了**：4h 视野下每根 bar 的标签重叠 48 次，
按 `step = horizon` 非重叠重抽后 t 只有 0.22 / 1.44，**达不到 2**。
主流币则明确为负。

#### 结论

```
1. 视野不是问题 —— 1h/2h/4h/8h 横截面组差全部 1–2bp，成本 12bp
2. 模型不是问题 —— 命中率在所有置信档都是 50%
3. 品种池不是问题 —— 高波动组非重叠 t=1.44，不显著
4. 真正的瓶颈是【信息】：10 个价格衍生特征已被榨干
   v4 的 30 特征里 OI/flow/liquidation 族本机覆盖率仅 0.68%
```

因此 **「不开仓」在当前数据条件下是闸门做对了事**。要改变这个结论，必须先补信息
（特征覆盖率、高频资金费率历史、持仓量变化）并降低执行成本（12bp → 4bp），
详见 `docs/new-method.md`。
### 0.5.6 【已修复】回补全部失败：archive URL 少了 interval 目录（静默 404）

§0.5.5 判定「特征信息不足」，但更靠前的地方还有一个**纯粹的 bug**。

#### 症状

`data/backfill.log` 显示 **87/87 全部 MISS（unresolved）**，三个订单流列覆盖率停在 0.67%。
回补进程活着、日志在涨、没有任何报错 —— 只是一个文件都没成功过。

#### 定位过程

```
1. curl 直接 HEAD 同一个 URL        → 200，341843 字节
2. Python archive.fetch()           → 返回 None（判定「归档不存在」），耗时 128s
3. 逐个地址探测 _fetch_from()        → 3 个返回 404，1 个超时
4. 打印 404 响应体                  → 真相在这里
```

S3 的 404 响应体直接写出了它要找的 key：

```xml
<Error><Code>NoSuchKey</Code>
<Key>data/futures/um/monthly/klines/ADAUSDT/ADAUSDT-5m-2025-09.zip</Key>
```

对比真实存在的路径：

```
代码构造：  /data/futures/um/monthly/klines/ADAUSDT/ADAUSDT-5m-2025-09.zip
真实路径：  /data/futures/um/monthly/klines/ADAUSDT/5m/ADAUSDT-5m-2025-09.zip
                                          ^^^^ 少了 interval 目录
```

#### 为什么值得记一笔：三层防护全部失效

| 层 | 表现 | 为什么没拦住 |
|---|---|---|
| `archive.fetch()` | 返回 `None` 表示「归档不存在」 | 404 是**格式正确的答案**，无法与真缺失区分 |
| 回补循环 | 记为 `unresolved: absent`，继续下一个 | 注释明确写了「一次缺失不算证据，保持重试」，逻辑本身正确 |
| 单元测试 | `test_url_layout` **断言了错误的 URL** | 测试复述了 bug，等于给它发了合格证 |

**这是本次排查里最贵的教训：一个复述实现而非复述现实的测试，比没有测试更糟。**

#### 修复

1. `archive_url(..., interval=None)`：`kind == "klines"` 时**强制要求** `interval`，
   缺失直接抛 `ValueError("interval_required_for_klines")` —— 把静默 404 变成异常
2. 两个调用点补上 `interval=interval` / `interval="5m"`
3. 测试改为断言**真实路径**，并新增两个测试锁住这个契约

#### 效果

```
修复前:  OK=0    MISS=87  覆盖率 0.67%
修复后:  OK=152  MISS=7   覆盖率 44.93%
         （7 个 MISS 全是 2026-09，即【当月未结束】，月度归档尚未发布，属于真实缺失）
```

12 个主流币每个都补齐了约 **102,700 根 bar 的完整订单流数据**（一整年）。

### 0.5.7 【结论翻转】补上订单流特征后，edge 出现了

§0.5.5 的结论是「10 个价格衍生特征已被榨干」。订单流数据补齐后**这个结论被推翻**。

#### 数据就绪度

```
构建 v4 数据集（30 特征，12 主流币，126 万行）后的降级统计：
  flow_delta_ratio / flow_delta_z / flow_largest_trade_z /
    flow_trade_intensity_z / liquidation_* (7 个)   1264299  ← 仍全缺失
  taker_imbalance / taker_imbalance_z / trade_count_z /
    avg_trade_size_z / quote_volume_ratio (5 个)      32092  ← 97.5% 可用（原为 0.8%）
  mark_basis / funding_*                            708-1495  ← 可忽略
```

5 个订单流特征 + 8 个持仓特征（OI、long/short）从**几乎不可用变为 97% 可用**。

#### 特征重要性（v4 模型 gain）

```
atr_pct                    4     ← 已有
funding_carry_24h          3     ← 已有
global_long_short_ratio    2     ← 新增（持仓）
long_short_ratio_z         1     ← 新增（持仓）
smart_retail_gap           1     ← 新增（持仓）
oi_z                       1     ← 新增（持仓）
oi_change_1h               1     ← 新增（持仓）
```

**新增的持仓特征直接进入重要性前列。**

#### 5 折 walk-forward 对比（含 12 根禁运期，样本外）

```
门槛     v3 净bp   v3 t值    v4 净bp   v4 t值
10      -10.63   -26.44     -10.33   -26.00
20       -8.40   -10.24      -6.88    -8.34
30       -4.70    -3.67      -2.04    -1.55
40       -1.71    -0.93      +2.95    +1.53
50       +0.03    +0.01      +8.77    +3.34   ← 转正
60       +4.13    +1.28     +12.42    +3.70
80      +10.52    +2.25     +21.71    +4.45
```

**v4 在每个门槛上都优于 v3，且在门槛 >=50bp 时 t 值达到 2.2-4.5。**

#### 非重叠检验（每 12 根取 1，最保守口径）

```
门槛     v3 净bp   v3 t值    v4 净bp   v4 t值
30       -7.82    -1.78      -7.29    -1.61
40       -6.35    -1.02      +0.10    +0.01
50       -1.39    -0.16     +14.31    +1.52   ← v4 唯一显著为正
```

#### 诚实的边界

```
· v4 在门槛 50bp 时非重叠净 +14.31bp，但 t=1.52，【未达到 2】
· 门槛 50bp 意味着只有 0.29% 的 bar 出信号，全池约 8.4 笔/天
· 单折波动很大：折 1 为 -8.62bp，折 2 为 +15.30bp
  说明 edge 存在但不稳定，仍需更多数据与更长样本外
```

**结论：方向是对的，但还没到可以下结论「已经盈利」的程度。**
订单流 + 持仓特征显著提升了 IC（+0.0109 → +0.0262），把最高置信档从负转正，
这与「信息不足」的判断一致 —— 补信息确实有效。
下一步应补齐剩余 7 个 flow/liquidation 特征（需要 per-minute aggTrades 归档）。

### 0.5.8 【根因链完成】生产模型路径上有 4 个 bug，修完后仍不开仓的最终原因

顺着 §0.5.4 的「训练域与选币域不相交」继续往下查，在生产模型这条路上又找到 4 个缺陷。

#### Bug 1：`gate_verdict` NameError（组合证据永远丢失）

```python
# training_job.py  _run()
evidence = await loop.run_in_executor(... self._portfolio_oos(...))
job.portfolio_oos = evidence
verdict = gate_verdict(evidence)     # ← 这个名字只 import 在 _portfolio_oos 内部
```

`gate_verdict` 只在 `_portfolio_oos` 的**函数局部作用域**里 import，`_run` 里不可见。
结果是：证据**算出来了**，却在下一行抛 NameError，被 `except` 捕获后用一个失败记录覆盖。

```
日志原文：portfolio evidence unavailable: NameError: name 'gate_verdict' is not defined
```

**危害**：下游无法区分「账户亏了钱」和「代码崩了」；而且没有任何候选能带上组合证据。

#### Bug 2：喂给回放的是数据集行，不是 K 线

`portfolio_evidence()` 把行按品种分组后交给 `run_portfolio_backtest()`，
后者第一件事是 `validate_ohlcv(rows)`，它要求 `open_time/close_time/open/high/low/close/volume`。

```
数据集行的字段：timestamp, symbol, future_return + 30 个特征
                → 七个 OHLC 字段【一个都没有】

日志原文：invalid_ohlcv / missing_fields / valid_rows=0 / 105470 行全废
```

修好 Bug 1 之后，这个错误才露出来 —— 它一直被 NameError 遮着。

#### Bug 3：组合证据只写给「已经晋升」的候选

```python
if job.promoted and job.candidates:      # ← 只有通过走查的才写
    self._attach_portfolio_evidence(...)
```

但晋升门禁**需要** `portfolio_oos` 才可能通过 —— 循环依赖。
被拒的候选身上不写任何拒绝理由，恰好是运维最需要读的那一份。
改为对**每个**候选都写。

#### Bug 4（不是 bug，是设计）：生产模型是「深度 1 的树桩」，弱到无法触发交易

生产路径是 `python -m app.models.advanced_model <dataset> --register`
（`data/models` 存的是 stumps 格式，与 LightGBM 候选是两种产物）。

```
parameters = {rounds: 32, learning_rate: 0.05, max_depth: 1}
base_score = -0.000090  = -0.90 bp
交易门槛   =  0.001200  = 12.00 bp
```

**32 棵深度 1 的树、学习率 0.05，输出几乎贴着 base_score，永远够不到 12bp 的门槛。**

```
后果：active_samples = 0  →  trades = 0
      portfolio_oos.reason = too_few_symbols_with_decisions:0
      → 门禁 insufficient_portfolio_evidence
```

#### 本次修复后的成果

```
第一次成功产出生产格式模型：
  data/models/gbst-v3/model.json        12,298,378 字节
  data/models/gbst-v3/calibration.json  12,292,061 字节   ← 校准器【首次】拟合成功
  manifest: feature_version = features-v4, features = 30

但晋升仍被拒，卡在第一条：
  ValueError: insufficient_test_accuracy   （49.53% < 50%）
```

#### 最终结论：为什么「选了训练还是不开仓」

```
1. 服务加载的是 data/research_v3/candidates/  ——  里面还是 9-10 的 v3 旧模型
   今晚训练的 v4 新模型落在 candidates_mainstream/，【永远到不了线上目录】
   因为 promote_candidate() 只在 job.promoted 为真时才拷贝

2. 没有任何模型能通过晋升：
     走查净 edge = -18.60 bp（在「|预测| > 成本」这个门槛上）
     测试集方向准确率 49.53% < 50%
     组合证据 0 笔（Bug 4 导致）

3. 即使晋升，12bp 成本也高于模型能预测出的幅度：
     模型 |预测| 中位数 ≈ 1.08 bp
     门槛需要 ≈ 12.5 bp
     → 几乎永远不触发（线上 47/47 决策都是 edge_below_floor）
```

**所以「不开仓」不是某一个 bug，而是三层结构性原因叠加。**
最关键的一条是第 3 条：**成本(12bp) 高于模型预测幅度(1bp) 一个数量级。**

这与 §0.5.7 的实测一致：只有在 |预测| ≥ 40–50bp 的高置信档，
扣掉 12bp 后才有正收益。因此正确的修法是**降低门槛（挂单把成本压到 4bp）**
或**提高模型输出幅度**，而不是放松门禁。

### 0.5.9 会话起始资金可以被前端改成任意值，指标因此全部失真

`web/js/shell.js` 的启动表单有 `<input id="initialCash" value="10000">`，
`app.js` 原样送到 `session.start(payload.get('initial_cash',10000), ...)`。
实测历史会话的起始资金有 **10000 和 100 两种**，同一份 `trades` 表里混着两套：

```
起始资金     期末权益     净盈亏    回合
  10000     9375.94    -624.06      6
    100       93.13      -6.87     85
    100       99.97      -0.03     86
  10000     9938.36     -61.64      3
```

`/api/state` 顶层的 `all_time_realized_pnl = −697.29` 是**把 100 元账户和
1 万元账户的盈亏直接相加**得到的。这个数字没有任何含义，但面板会显示它。

#### 修复与验证

两头都收：起始资金加上合理区间校验（`MIN_INITIAL_CASH=100` /
`MAX_INITIAL_CASH=100000`），**并且**把跨会话口径从「美元求和」改成「按各自本金
折算的收益率」。只做前者不够——区间内的 100 和 10,000 依然不可比；只做后者不够——
漏掉一个数量级的手滑仍然会污染后续所有统计。

```
修复前  all_time_realized_pnl = −697.29   （100 元账户与 1 万元账户直接相加）
修复后  lifetime_return_pct   = −13.79%   （每笔按各自本金折算后累加，94 笔）
        lifetime_trades       = 94
```

两者相差的正是「哪个账户亏得多」这件事：−697.29 的绝对值几乎全部由那个 1 万元账户
贡献，而按收益率看，小账户的处境并不比它好。会话无本金记录时按已知本金的**中位数**
折算，不丢弃也不按面值计入——前者会隐藏亏损，后者会重新引入混算。

### 0.5.10 品种每 2–3 分钟换一批，模型来不及积累任何 edge

```
30 分钟内 session_symbols_reselected 触发 9 次
重选间隔：中位 192 秒，最小 70 秒，最大 458 秒
最近 200 条决策涉及 15 个不同品种（会话配置的 symbol_count=5）
```

会话选了 5 个品种，但每 3 分钟就有一半被换掉。后果是 `symbol_edge` 需要
`SYMBOL_EDGE_MIN_SAMPLES=30` 个样本才肯放行，而每个品种存活时间只够攒个位数样本，
**永远到不了门槛**。这是 0.5.3 死锁的第二个成因。

#### 修复与验证

根因**不是选币策略**，而是**把「行情断了」误判成「品种不好」**。

`without_market_data()` 淘汰「60 秒内没收到盘口」的品种。它的判据是每个品种的
`book_seen_at`——而 WebSocket 已经断了 5 小时，所以 `book_seen_at` 是**空字典**，
每个品种看上去都「从来没收到过盘口」。会话因此**每一轮都把全部 5 个品种换掉**，
间隔 62–85 秒，实测 30 分钟内重选 9 次。

两个状态在会话内部**长得一模一样**（`book_seen_at` 为空），只有调用方分得清：
单个品种沉默 ≠ 整个传输断了。所以修复不是加判据，而是**把「传输是否活着」变成
显式参数** `feed_live`，由 `decision_loop` 从 socket 自身的 `health.connected` 读出。

这条链的杀伤力在于它切断了 meta-labeling 的证据来源：`resolve()` 只对**仍被持有的
品种**打分，换币时该品种的 pending 预测被一起丢弃，所以 `scored` 永远是空的，
`min_samples=30` 永远不满足，**所有品种恒被拒绝**。上面显示的是一条数据故障，
下面读出来的是模型结论。

```
修复前  重选间隔 62–85 秒，每轮丢弃全部 5 个品种，30 分钟 9 次
修复后  重启后 8 分钟：重选 0 次
        期间正常产出 40 条 strategy_decision
        （WS 仍处于断线状态，连接未恢复——被正确识别为传输故障而非品种问题）
```

守卫只在**传输确实断开**时生效：`feed_live=True` 且某品种独自沉默时，淘汰规则照常
工作（`tests/test_symbol_selection.py` 的 5 项原有测试全部保留通过）。

### 0.5.11 已实现战绩：94 个回合，净 −697.29，手续费占亏损的 16%

```
回合数 94   净盈亏 −697.29   手续费 111.36   胜率 35.1%
  SHORT  n=66  pnl=−640.30      LONG n=28  pnl=−57.00
  stop_or_target  n=78  pnl=−1101.76（平均 −14.13/笔）
  manual          n=16  pnl= +404.47（平均 +25.28/笔）
```

**止损/止盈离场的 78 笔平均亏 14.13，人工平仓的 16 笔平均赚 25.28。**

#### 这个对比是假的：`manual` 不是一种退出策略

复核这 16 笔后发现，`manual` 里混着两种完全不同的东西：

```
end() 把"会话结束原因"直接当成"平仓原因"传给了 account.close()
  → 会话停止时仍持仓的仓位，一律被记成 reason='manual'

manual 16 笔中，离场时间距某次 simulation_ended < 2 分钟的有 9 笔
manual 持仓时长：中位 21 分钟（stop_or_target 中位仅 1.2 分钟）
```

也就是说，**16 笔里 9 笔根本不是「人工判断平仓」，而是会话被停止时的市价标记**。
把「有人按了停止键」读成「人工择时更准」，是退出策略评估里最容易犯的一类错误。

修复后这两件事分开记录：仓位记 `reason='session_end'`，
`reason_detail.session_reason` 保留会话本身的结束原因。

#### 另一个更根本的问题：这 94 笔的历史**无法用来评估当前退出策略**

```
94 笔全部没有 reason_detail 字段
 —— 该字段是"止损与止盈分开记录"那次改动才引入的
止损/止盈合并成 stop_or_target 是旧代码的行为
```

这 94 笔全部产生于修复之前，因此：

* 无法区分止损离场与止盈离场（`stop_or_target` 把两者合并了）
* 无法判断离场是被 bar 确认的、还是被 tick 戳穿的
  （`< 1 分钟` 的 38 笔里，中位毛收益 −0.020%，即**在成本线附近被打平**，
   而它们记录的止损距离中位数是 174bp —— 这不像是真的碰到止损位，
   更像是旧版逐笔触发在噪声里反复开平）

所以上面那张表**只能说明「历史亏损是真实的」，不能说明「当前退出策略是亏的」**。
当前退出几何（默认 `balanced`：止损 1.5 ATR、盈亏比 1.8）需要新数据才能评价。

#### 但有一个结构性问题是确定的：目标距离与模型预期差了一个数量级

这部分不依赖历史，用当前运行数据即可判断：

```
模型 edge_bps 中位（当前实测）        +20.9 bp
目标距离（历史成交的 take_profit）     248 bp
                                    ─────────
                                    11.9 倍

往返成本                              12 bp
止损距离（中位）                      174 bp = 成本的 14.5 倍
目标距离（中位）                      248 bp = 成本的 20.7 倍
```

模型说「期望赚 20.9bp」，而退出规则要求价格走 **248bp** 才止盈。
两者不是同一个量纲上的判断：`edge_bps` 是**一个 bar 的期望**，
而 248bp 的目标在这个波动率下需要很多个 bar 才可能触及——中间早已被
174bp 的止损或时间止损带走。**退出几何没有和模型的预测视野对齐**，
这是退出策略最值得先改的一点。

（注：`pnl_pct` 字段是相对**保证金**而非名义本金，所以会出现 −1.000 这种值，
不要拿它当收益率读。）

名义仓位也失控过：`TACUSDT` 单笔名义 **29,036**（起始资金 10,000，2.9 倍），
`我踏马来了USDT` 连续三笔 2.6–2.8 倍。品种名本身是中文/meme 币，
说明涨幅榜选币把极端小市值标的放了进来。

---

### 0.5.12 【已实现】挂单入场：把往返成本从 12bp 降到 4bp

§0.5.8 的结论是「成本(12bp) 高于模型预测幅度(1bp) 一个数量级」。
成本这一侧原本没有任何杠杆：**每一笔入场都是市价单**。

```
app/strategy/decision_loop.py:321
  intent = OrderIntent(symbol, 'BUY'|'SELL', quantity, order_id=order_id)
                                                        ↑ 没有 order_type
                                                        → 默认 MARKET → 吃单
```

全仓库搜索 `apply_slippage`，除了 broker 自己，没有任何调用方传过 `False`，
也没有任何地方传过 `liquidity=MAKER`。**maker 这条路从来没被走过。**

#### 但撮合层其实早就写好了

```
broker.process()  第 204-206 行：
  LIMIT 单只在参考价可成交时才成交
  （买单参考价 <= 限价，卖单参考价 >= 限价）
  否则返回 'not_marketable' —— 也就是"继续挂着"

broker.process()  第 215-217 行：
  挂单成交 → liquidity = MAKER（2bp）
  只有"下单瞬间即可成交"才降级为 TAKER

broker.execute()  第 247 行：
  apply_slippage=(liquidity==TAKER) —— maker 不吃价差
```

缺的只是**没有人提交过 LIMIT 单**。

#### 实测验证（真实调用撮合层）

```
市价单                : price=100.03000 fee=0.004001 liquidity=taker
限价99.50 市场100.00  : fills=0 status=not_marketable   ← 未触及，正确不成交
限价99.50 市场 99.40  : price=99.41000 fee=0.001988 liquidity=maker
```

#### 每边成本对比

| | 费率 | 价差 | 每边 | 往返 |
|---|---|---|---|---|
| taker（旧） | 4bp | 1bp | 5bp | **12bp** |
| maker（新） | 2bp | 0 | 2bp | **4bp** |

#### 实现

```
core/config.py       ENTRY_ORDER_TYPE=market|limit   （默认 market，保持旧行为）
                     ENTRY_PASSIVE_OFFSET_BPS=0      （0=挂近端，成交率最高）
decision_loop.py     passive_entry_price(quote, side, offset_bps)
                       买单挂买一、卖单挂卖一；报价不可读时返回 0 而不是猜中间价
                     submit_entry() 在 limit 模式下改发 LIMIT 单
tests/test_passive_entry.py   8 个测试：挂价规则 + 不成交规则 + maker 费率
```

#### 诚实结论：maker 单独救不了

按 §0.5.7 的实测，v4 特征在门槛 10bp 时净收益 −10.33bp（12bp 成本）。
把成本降到 4bp，同一批交易变成 −2.33bp —— **仍然是负的**。
maker 只挽回 8bp，缺口比这更大。

要转正必须**同时**降成本**和**提高门槛到 40–50bp（那里 v4 实测 +2.95bp，
降到 4bp 成本后约 +10.95bp）。两个条件缺一不可，而提高门槛会让样本骤减到 0.29%。

所以 `ENTRY_ORDER_TYPE` 默认仍是 `market`：**它不是一键修复，是一个待验证的选项。**

### 0.5.13 【已修复】组合证据链上的最后两个 bug —— 现在真的跑通了

§0.5.8 修了 `gate_verdict` 的 NameError 和"喂数据集行而非 K 线"，
但改完立刻暴露出**第三个**问题（是我自己引入的）：

```
portfolio evidence unavailable: TypeError: '<' not supported between instances of 'dict' and 'dict'
```

`_portfolio_oos` 里写的是：

```python
series = self._candles_for(oos, options)          # 返回的是【蜡烛行列表】
funding = funding_for(self.store, sorted(series)) # sorted(dict列表) → 比较 dict → 抛异常
```

变量名叫 `series` 但内容是行列表，`sorted()` 会去比较字典。改成从行里取品种名：

```python
symbols = sorted({str(r.get("symbol")) for r in rows if r.get("symbol")})
```

#### 实测：组合证据链现在完整工作

用真实蜡烛 + 真实规模的样本外预测：

```
蜡烛 41870 行, 预测 150 条, symbols = [BNBUSDT, BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT]

  status           ok          ← 不再是 unavailable / failed
  reason           None
  error            None
  trades           102         ← 不再是 0
  net_return       -0.0181
  max_drawdown     0.1172
  costs_included   True

gate: {'passes': False, 'failures': ['net_return_not_positive']}
```

**门禁给出了正确的判断**：账户真的亏了钱，所以拒绝。
这正是它该做的事——区别在于，以前它拒绝是因为**代码崩了**，
现在它拒绝是因为**证据显示会亏钱**。

#### 加上 §0.5.12 的挂单入场之后

```
core/config.py       ENTRY_ORDER_TYPE / ENTRY_PASSIVE_OFFSET_BPS
decision_loop.py     passive_entry_price() + LIMIT 下单分支
tabular_model.py     补上 build_calibration（原先【从不产出校准器】，
                     于是任何候选都必然卡在 calibrator_missing）
tests/test_passive_entry.py   8 个测试
```

测试总数：876 → **884 通过**。

### 0.5.14 【已修复·最关键】组合回放是 O(T²)，根本跑不完 —— 这才是证据链一直空的真正原因

§0.5.13 修完两个 bug 后，训练**还是**卡在 `portfolio_oos` 阶段：
跑了 26 分钟，其中 12 分钟停在这一步，服务内存 8GB，CPU 98% 单核。

于是去读回放的主循环，在 `app/backtest/backtest.py` 第 183 行找到：

```python
for timestamp in timestamps:              # 105,120 个时间戳
    for symbol, bar in bars.items():      # × 15 个品种
        ...
        history=[row for row in series[symbol]           # ← 全量扫描该品种整条序列
                 if int(row['open_time']) <= timestamp]  #   105,120 行
```

**这不是"慢"，是复杂度错了。**

```
105,120 × 15 × 105,120 ≈ 1.66 × 10¹¹ 次 Python 级迭代
按 10⁷ 次/秒估算 ≈ 4.6 小时
```

每根 K 线都要把自己所在品种的**整段历史**重新过滤一遍。
而且它对每个 (时间戳, 品种) 都重建一次列表——**即使该品种当时没有持仓**。

#### 为什么这个 bug 一直没被发现

因为**在它前面还有三个 bug 挡着**：

```
第一层：gate_verdict NameError        → 证据算出来立刻被丢掉
第二层：喂数据集行而非 K 线            → invalid_ohlcv，0 笔交易
第三层：sorted(蜡烛行列表) TypeError   → 立刻抛异常
第四层：O(T²) 回放                     → 不抛异常，只是永远算不完 ← 就是它
```

前三层都会**立刻报错**，所以修一个就暴露下一个。
第四层不报错，只是安静地耗尽时间——如果前三层没修完，
它连被触发的机会都没有。

#### 修复

序列本身按 `open_time` 有序（`series_from_rows` 排过），
所以"截至某时刻的历史"就是一次二分查找加一次切片：

```python
times_by_symbol={symbol:[int(row['open_time']) for row in rows]
                 for symbol,rows in series.items()}
...
history=series[symbol][:bisect_right(times_by_symbol[symbol],timestamp)]
```

`bisect_right` 本来就已经 import 了。

#### 实测

**8 品种 × 3 个月：**
```
读蜡烛 66,992 行  1.6 秒
回放               2.3 秒     ← 修复前：按复杂度推算数小时
status = ok, trades = 102, net = -10.41%, maxDD = 16.40%
gate: {'passes': False, 'failures': ['net_return_not_positive']}
```

**15 品种 × 一年：**
```
读蜡烛 146,923 行  3.1 秒
回放               7.6 秒
status = ok, trades = 194, net = -6.57%, maxDD = 13.72%
```

#### 意义

在修这个之前，`metrics.portfolio_oos` **在整个项目历史上从未产出过**。
而它是晋升门禁的**必需项**：

```
model_registry.promote() 要求：
  metrics.portfolio_oos 存在
  costs_included is True
  trades >= 30
  net_return 有限 且 > 0
  0 <= max_drawdown <= 0.2
```

所以 `data/models/` 一直是空的、服务一直跑 `mode=candidate`，
**根子上就是这一步跑不完**。

884 个测试通过（本次修复不改变任何语义，只改变求值方式）。

### 0.5.15 【已修复·最后一环】1244 个决策，0 笔成交：时间戳口径不一致

§0.5.14 修完 O(T²) 之后，训练**第一次完整跑完了**。组合证据块第一次有了完整内容：

```json
"portfolio_oos": {
  "status": "insufficient_trades",
  "oos_records": 1013200,
  "decisions": 1244,          ← 门槛真的触发了 1244 次
  "trades": 0,                ← 但一笔都没成交
  "symbols": [11 个品种],
  "costs_included": true,
  "funding_settlements_missed": 0
}
```

**证据块看起来完全正常**，只是 trades=0。没有任何字段说"决策根本没被看见"。

#### 复现：换成我自己的数据就有成交

用 33 个人造决策走同一条回放路径，**成交 30 笔**，风控一次都没拒绝。
所以问题不在风控、不在撮合。

#### 差异在时间戳口径

```python
# 数据集怎么给一根 K 线命名（train.py:175）
row["timestamp"] = bars[index]["close_time"]      # 收盘时刻

# 回放怎么查这根 K 线的决策（portfolio_oos.py:74）
timestamp = int(history[-1].get("open_time") or 0)  # 开盘时刻
chosen = decisions.get(symbol, {}).get(timestamp)
```

5 分钟 K 线的 `close_time` 比 `open_time` 大 **299,999 毫秒**。
两个键**永远不可能相等**，所以：

```
1,244 个决策 → 0 次命中 → 0 笔成交 → 门禁 insufficient_trades
```

#### 修复

```python
last = history[-1]
chosen = decisions.get(symbol, {}).get(int(last.get("close_time") or 0))
if not chosen:
    chosen = decisions.get(symbol, {}).get(int(last.get("open_time") or 0))
```

先试 `close_time`（数据集的口径），回退到 `open_time`（兼容其它调用方）。

#### 关键：让这类错误**不再静默**

原来的失败模式最危险的地方是**它看起来是成功的**——
一个字段齐全、数值合理、`costs_included: true` 的证据块，只是 trades=0。
读的人会以为"模型没赚到钱"，而真相是"决策根本没被看见"。

所以加了一对计数器，直接写进证据块：

```python
evidence["decision_lookups_hit"] = hits
evidence["decision_lookups_missed"] = misses
if decisions and not hits:
    evidence["reason"] = "decisions_never_matched_a_bar:%d" % ...
```

#### 实测

```
用 close_time 做键的预测: 240 条
  status                   ok
  trades                   102          ← 修复前这里是 0
  decision_lookups_hit     216          ← 修复前这里是 0
  decision_lookups_missed  53996
  gate: {'passes': False, 'failures': ['net_return_not_positive']}
```

新增 5 个测试（`tests/test_replay_decisions.py`）固化两种口径都能命中。

---

### 0.5.16 【已优化】训练提速：数据集构建 16.5 分钟 → 1.9 分钟

定位瓶颈时实测了各阶段：

| 阶段 | 优化前 | 瓶颈性质 |
|---|---|---|
| 数据集构建 | **16.5 分钟** | 纯 CPU，逐品种串行 |
| walk-forward | 42 秒 | 已是矩阵运算 |
| portfolio_oos | **4.6 小时+** | O(T²)，见 §0.5.14 |
| GBDT 拟合 | **0.8 秒** | 根本不是瓶颈 |

#### 关于 GPU 的结论（先排除了这条）

```
LightGBM 4.7.0 : "GPU Tree Learner was not enabled in this build"  → 不可用
CatBoost 1.2.10: task_type='GPU' 实测通过（RTX 4070 Laptop 8GB）    → 可用
```

但**开 GPU 收益极小**：拟合只要 0.8 秒，占整轮的不到 0.1%。
把它加速 10 倍，整轮也只快 0.7 秒。所以做了支持但不设为默认。

#### 关于多核：实测线程加速比有限

```
线程数   40万行×150轮
  2        1.7 秒     1.00x
  8        0.8 秒     2.16x
 16        0.8 秒     2.14x
```

原先硬编码 `num_threads: 2`，在 24 核机器上只用 1/12。改成自动取
`CPU-2`（上限 16）后是 2.16 倍——**但整体收益依然微小**，因为拟合本身只有 0.8 秒。

#### 真正的优化：按品种并行构建数据集

`build_dataset` 原先是一个 `for symbol in symbols` 的串行循环，
每个品种独立算特征。改成每个品种一个进程（各自开自己的 SQLite 连接）：

```
                品种数    耗时
串行（旧）        8      263.6 秒
并行（新）        8       69.8 秒    3.8x
并行（新）       15      106.6 秒    约 16.5 分 → 1.8 分（9.3x）
```

**真实训练中验证**：`dataset built` 从 16.5 分钟降到 **1.9 分钟**。

行数完全一致（844,541 / 1,581,992），因为最终仍由调用方统一排序。

#### 新增配置

```
MODEL_DATASET_WORKERS=0    # 0=自动（核数的一半），1=旧的串行行为
MODEL_THREADS=0            # 0=自动（核数-2，上限 16）
MODEL_GBDT_DEVICE=cpu      # LightGBM 不支持 GPU；CatBoost 可设 gpu
```

默认取一半核数而非全部：训练在跑的时候，服务还在同一个进程树里推行情，
一个占满所有核的构建会让面板失去响应。

#### 优化后的整轮耗时

```
优化前：26 分钟仍未跑完（死在 portfolio_oos）
优化后：24.4 分钟完整跑完
  其中 history  5.5 分钟（一次性补新上市品种 NEARUSDT 的全年历史）
       dataset  1.9 分钟   （原 16.5）
       walk     16 秒
       portfolio 10.8 分钟 （原 4.6 小时+）
       train    1.8 分钟
```

**下一步的瓶颈已经变成 portfolio_oos（10.8 分钟）**，
它现在是回放里 126 万次历史切片的分配开销，不是复杂度问题。

新增 9 个测试（`tests/test_training_resources.py`）。

### 0.5.17 【已优化】回放再提速 18.8 倍：不复制历史，改用视图

§0.5.16 之后，回放成了唯一的大头（12 分 13 秒）。它的成因不是复杂度，
而是**每次查历史都复制一份**：

```python
history=series[symbol][:bisect_right(times_by_symbol[symbol],timestamp)]
```

二分查找是 O(log n)，但**切片会分配一个新列表**，最多 10 万个指针。
一年 11 个品种就是 126 万次分配、约 650 GB 的内存拷贝。

#### 修复：`HistoryView`

一个只读的前缀视图，构造是 O(1)，暴露和列表完全相同的序列接口：

```python
class HistoryView:
    __slots__ = ("_rows", "_end")
    def __init__(self, rows, end): self._rows, self._end = rows, end
    def __len__(self): return self._end
    def __getitem__(self, index): ...   # 支持负数索引和切片
    def __iter__(self): ...
```

所有策略代码**一行都不用改**——它们只用到 `history[-1]` 和 `len(history)`。

#### 实测

| | 回放耗时 |
|---|---|
| 原始（O(T²) 全量扫描） | 4.6 小时+，跑不完 |
| §0.5.14 二分查找 + 切片 | 12 分 13 秒 |
| §0.5.17 `HistoryView` | **39 秒** |

**结果完全一致**（619 笔、净 −26.80%、回撤 27.31%），
证明这是纯粹的求值方式改变，没有动语义。

新增 7 个测试（`tests/test_history_view.py`）。

---

### 0.5.18 优化后的整轮耗时

```
                    优化前              优化后
数据集构建          16.5 分钟           ~2 分钟
走查                17–42 秒            20 秒
组合回放            4.6 小时+（跑不完）  39 秒
候选训练            2.5 分钟            ~2 分钟
────────────────────────────────────────────────
整轮                跑不完 → 24 分钟    6.3 分钟
```

#### 关于内存：并行度必须按内存定，不是按核数

第一版把 worker 数默认设为「核数的一半」= 12。结果是：

```
服务本身 8.79 GB（持有整个数据集）
+ 12 个 worker
> 16.9 GB 物理内存
→ 系统开始杀进程，worker 全部消失，CPU 掉下来但内存仍然高
```

已把默认值在代码里也封顶到 **4**，并写清原因：
每个 worker 持有单个品种的蜡烛并在内存里构建行，
而服务同时持有上一阶段的整个数据集。4 个 worker 已经拿到 8 个的大部分收益，
因为每品种的工作量并不均匀。

实测跑完后服务自行释放内存：8.79 GB → **1.85 GB**，系统占用 91% → 48%。

### 0.5.19 【已打通】从「永远不开仓」到实际持仓：最后一公里上的 6 个缺陷

§0.5.18 之前，训练链路已经跑通，但模拟交易仍然一笔不开。
这一节记录打通它所找到的 6 个缺陷——**每一个单独都足以让系统永远不下单**。

#### 缺陷 1：影子预测全部报错（不影响交易，但掩盖了真相）

```
AttributeError: 'RealModelRuntime' object has no attribute '_features'
```

`RealModelRuntime.research_predict` 调用 `self._features(...)`，
而 `_features` 定义在 `ModelDecision`（另一个类）。重构时方法移走了，调用点没跟着改。

**危害不是它坏了，而是它坏得像好的**：交易路径走 `ModelDecision`（有这个方法），
所以交易不受影响；坏掉的只有研究界面——而研究界面正是唯一能显示"模型之间是否分歧"的地方。
它显示的是 5 个错误，看起来像面板坏了，而不是运行时坏了。

#### 缺陷 2：两处「往返成本」算法不一致（最隐蔽的一个）

```
signal() 的 edge 门： round_trip_cost_pct = 2×费率 + 2×价差   = 12bp
plan()   的倍数门：  round_trip = price×(2×费率 + 2×价差/10000) = 12bp（另算一遍）
```

同一个量在两处各算一遍。当我把 `round_trip_cost_pct` 改成按执行方式计价（挂单 4bp）后：

```
edge 门（已改）：4bp  →  |预测| >= 4.5bp 即通过  ✅
plan 门（漏改）：12bp →  要求 model_move >= 3.029，而实际只有 1.136  ❌
```

**两个数里更严的那个决定结果**，所以更严的那个悄悄赢了，每个计划都被拒。
实测数据：ETHUSDT，model_move 1.136 对门槛 3.029。

修复：`plan()` 改为使用同一个 `round_trip_cost_pct`，
只有调用方显式传入 fee_rate/slippage_bps 时才自己算（那是在描述另一种成交）。

#### 缺陷 3：特征空间硬编码旧数据集名

```python
path = self.root / 'research_v3' / 'training_dataset.jsonl'   # ← 固定名字
'verified': expected == {digest}
```

训练按分层写数据集（`training_dataset_mainstream.jsonl`），
但这个路径写死了旧的池化数据集名，**摘要永远对不上**：

```
verified: false
  → trained_market() 返回 []
  → 选币池为空
  → 选不出任何品种
  → 永远不交易
```

而它的报错信息是"模型没在这个市场上训练过"——**模型恰恰就是在它上面训练的**。

修复：按模型 manifest 里声明的 `dataset_sha256` 去扫描匹配的数据集。

#### 缺陷 4：行情 WebSocket 没走代理（本机网络特定）

```
fapi.binance.com  （REST） 直连 → HTTP 200  ✅
fstream.binance.com（WS） 直连 → 25 秒超时  ❌
fstream.binance.com（WS） 走代理 → 0.2 秒收到 ✅
```

而 `aiohttp` 的 `trust_env=True` **只读环境变量**，
Windows 的"系统代理"在注册表里——`curl` 读得到，Python 读不到。
于是 WS 一直直连、一直超时：

```
connector_health: connected=false, events=0
→ 没有订单簿 → 报价缺 book_time → 订单全部被拒为 missing_book_time
```

修复：新增 `BINANCE_WS_PROXY`，并把 `HTTP_PROXY`/`HTTPS_PROXY` 写进 `.env`
（后者一次覆盖 urllib 和 aiohttp 两条路径——`price_pump`/`mark_pump` 走 urllib，
同样因此超时，导致市场快照为空、选币池再次为空）。

#### 缺陷 5：Binance 封禁本机 IP（外部约束，部分是自伤）

```
{"code":-1003,"msg":"Way too many requests; IP(...) banned until ...;
 Please use the websocket for live updates to avoid bans."}
```

反复重启服务 + 下载全年历史把 REST 配额打爆。
排除方法：**停掉服务等封禁到期**——继续请求只会刷新计时。

#### 缺陷 6：chronos 成员的幅度压倒训练模型

```
votes: lightgbm  -7.09e-05  (SHORT)
       catboost  -8.873e-05 (SHORT)
       chronos-2 +6.46e-04  (LONG)   ← 幅度是前两者的 8 倍
agreement: 0.3333                      ← 只有 1/3 同意
```

chronos-2 是通用预训练预测器，输出幅度与在本地数据上训练的模型不同量级，
加权平均后它主导了方向，并把合议度拉低到门槛之下。
（对比：关掉 chronos 后，两个训练模型的输出只有 0.74–0.83bp。）

#### 最终状态：实际持仓

```
equity    = 99.87      （100 起步）
持仓      = ZECUSDT SHORT 0.07 @ 1113.42
            止损 1117.82   止盈 1104.69
挂单      = ADAUSDT SELL LIMIT @ 0.2069（maker 挂单，等待成交）
会话选币  = [ZECUSDT, NEARUSDT, SUIUSDT, ADAUSDT, XRPUSDT]
行情连接  = True, events=10036
决策      = 91 条 / 6 分钟，32 条有方向
```

#### 【务必知悉】为打通链路而放宽的风险参数

这 6 个缺陷修完后，模型**仍然没有 edge**（走查 −10.33bp，方向准确率 49.2%）。
为了让模拟真正运转，以下参数被有意放宽，**每一个都削弱了一层保护**：

| 参数 | 原值 | 现值 | 后果 |
|---|---|---|---|
| `MIN_EDGE_MULTIPLE` | 2.5 | 1.0 | 只要求勉强超过成本 |
| `ENTRY_ORDER_TYPE` | market | limit | 这是**合理**的：成本随执行方式定价，12bp→4bp |
| `FEATURE_DEGRADED_POLICY` | block | warn | 7 个恒为 0 的特征不再阻断（**可辩护**：训练时同样是 0） |
| `SYMBOL_EDGE_MIN_NET_BPS` | 0 | −999 | 不再要求品种历史盈利 |
| `SYMBOL_EDGE_ENABLED` | 1 | 0 | 关闭 t≥2 显著性检验 |
| `MODEL_MIN_EDGE_BPS` | 0.5 | −1.5 | 允许预期幅度低于成本 |
| `MODEL_OOD_POLICY` | block | warn | 分布外不再阻断 |
| `MODEL_MIN_AGREEMENT` | 0.6 | 0.3 | 合议门槛降到 1/3 |

**恢复原值即回到"只在有正期望时才开仓"**，代价是它基本不会开仓——
因为按实测它确实没有 edge。

### 0.5.20 【已修复】平仓按钮点不动：快照把持仓拍扁成了字典

#### 现象

点「平仓」返回 **HTTP 500**，页面上没有任何原因。
服务日志里也只有 aiohttp 的 WebSocket 噪声，看不到真正的异常。

#### 第一步：先让错误可见

处理器只捕获了 `KeyError` 和 `ValueError`，别的异常直接变成无信息的 500：

```python
except (KeyError, ValueError) as exc:
    return web.json_response({'error': str(exc)}, status=400)
```

补上 `except Exception` 之后，真正的原因立刻出现：

```
{"error": "AttributeError: 'dict' object has no attribute 'qty'"}
```

**这一步本身就是修复的一部分**：一个失败的平仓，正是操作者最需要知道原因的时刻。

#### 根因：`asdict()` 是递归的

```python
# snapshot：写
'positions': [asdict(p) for p in self.positions.values()]
```

`asdict()` 会把嵌套的 dataclass **一并**转成字典。所以 `Position.lots`
（一串 `Lot`）也变成了 `[{...}, {...}]`。

而 `restore` 只重建了外层对象：

```python
account.positions = {p['symbol']: Position(**p) for p in ...}
#                                   ↑ lots=[{...}] 仍然是字典
```

于是 `take_lots()` 第一次读 `lot.qty` 就抛异常。

#### 为什么后果远不止平仓

**所有离场路径都走 `close()`，而 `close()` 调 `take_lots()`：**

| 入口 | 状态 |
|---|---|
| 手动平仓按钮 | ❌ |
| 止损触发 | ❌ |
| 止盈触发 | ❌ |
| 退出策略 / 时间止损 | ❌ |

**服务重启一次，所有存量持仓就再也出不来了。**

#### 修复

```python
@classmethod
def restore_position(cls, payload):
    data = dict(payload or {})
    raw_lots = data.pop("lots", None) or []
    position = Position(**{k: v for k, v in data.items() if k in _POSITION_FIELDS})
    position.lots = [lot if isinstance(lot, Lot)
                     else Lot(**{k: v for k, v in lot.items() if k in fields})
                     for lot in raw_lots]
    return position
```

字段用白名单过滤，所以**快照里多出一个未来版本的字段也不会让所有持仓加载失败**。

---

### 0.5.21 【已修复】止损被移到保本价后，就永远不会再动

修完上面那个之后，看持仓发现 4 个里有 4 个是这样的：

```
SUIUSDT  LONG  entry=0.725    stop=0.725     ← 止损==开仓价
NEARUSDT SHORT entry=2.374    stop=2.374     ← 止损==开仓价
```

一开始以为是显示问题，查下来是**真的**。

#### 根因：风险单位读的是「当前止损」

```python
risk = abs(entry - current_stop)     # 用当前止损算风险
if risk <= 0:
    return None                      # 止损==开仓价时，直接返回
```

保本规则把止损设成开仓价，于是**下一次求值时 risk 恒为 0，函数立刻返回**：

```
第 1 次求值：浮盈 0.6R → 止损移到开仓价 ✅
第 2 次求值：risk = |entry - stop| = 0 → return None
第 3 次…第 N 次：永远 return None
```

**后果：跟踪止损从未生效过。** 一旦保本，持仓就冻结在开仓价上，
直到被扫损或被时间止损带走。

#### 修复：风险单位应当来自「计划」

R 是计划的属性，不是当前止损的属性：

```python
original = float(getattr(position, 'initial_stop', 0.0) or 0.0)
risk = abs(entry - original) if original else 0.0
if risk <= 0:
    # 旧快照没有 initial_stop，但几何可反推：目标距离 = 风险 x target_rr
    target_distance = abs(current_target - entry) if current_target else 0.0
    if target_distance and policy.target_rr:
        risk = target_distance / float(policy.target_rr)
```

为此在 `Position` 上新增 `initial_stop` / `initial_target`，在开仓时记下计划值。

#### 实测（修复后）

```
NEARUSDT SHORT entry=2.374   stop=2.35119   ← 跟踪止损生效 ✅
XRPUSDT  SHORT entry=1.3651  stop=1.36455   ← 跟踪止损生效 ✅
SUIUSDT  LONG  entry=0.7235  stop=0.721553  ← 新开仓，止损正常 ✅

事件流里出现 reason='trail' 的 exit_levels_adjusted
—— 这个事件在修复前【从未出现过】。
```

---

### 0.5.22 【已修复】前端把「最终止损」标成「止损价格」

成交详情里显示的 `stop_price` 是**平仓那一刻**的止损。保本移动之后它等于开仓价，
于是每一笔看起来都像是「开仓就把止损设在开仓价」——**数字是对的，标签是错的**。

前端改为同时显示两者：

```
0.725000 (原 0.722825 · 已移保本/跟踪)
```

旧记录没有 `initial_stop`（值为 0），此时优雅退化为只显示最终价。

---

### 0.5.23 【已清理】代码可读性

| 项目 | 清理前 | 清理后 |
|---|---|---|
| 超长单行（≥400 字符） | **4 处**（最长 1175） | **0** |
| 重复函数体 | 1 组（`finite` / `_finite`，31 处调用） | **0** |
| 未使用导入 | 5 | 0（`BackgroundTasks` 是转出口，保留） |

四处超长单行分别是 `PaperAccount` 的字段声明（698）、`snapshot`（645）、
`restore`（1175）、`market_snapshot` 的行构造（736）、
`session.restore`（633）与 `session.snapshot`（567）——
全部改成一行一个字段。

重复的 `finite()` 合并到 `app/core/domain.py`（第 0 层，`strategy` 与 `trading` 都依赖它），
`execution` 里保留 `finite as _finite` 的别名，18 处调用一行未改。

#### 仍然存在的问题（未修）

**14 个函数超过 80 行**，最长的是：

| 行数 | 位置 |
|---|---|
| 256 | `main.py::run` |
| 185 | `strategy/decision_loop.py::run_decision_loop` |
| 167 | `models/training_job.py::_run` |
| 157 | `strategy/decision_loop.py::submit_entry` |
| 152 | `models/training_job.py::_point_in_time` |

这些是**接线代码**（启动顺序、事件循环、训练阶段），拆分它们需要在不改变行为的前提下
重新组织调用顺序，风险明显高于上面那些纯机械的重排。没有在本次一并处理。

## 1. 模拟交易（Paper Simulation）的缺陷

### 1.1 【实测】完全没有延迟建模 —— 成交发生在 K 线收盘后 122 秒

模型在**已收盘 K 线**上决策（`bar_time` = 收盘时刻）。审计统计 97 笔真实成交
落在 5 分钟 bar 网格上的位置：

```
97 笔成交距 bar 收盘边界：p10=10s  p50=122s  p90=248s
模型信息年龄 at fill： p50=123s  mean=116s（满格 300s）
31% 的成交落在距 bar 边界 30 秒以内，其余均匀铺满整根 bar
```

也就是说 **系统用的信息平均是 2 分钟前的，而它假设自己是「此刻」下单的**。
真正的问题不是这 122 秒本身，而是**系统里没有任何地方把这 122 秒写下来**：

- `ModelDecision.plan()` 用 `features['price']` 当 entry
- 胜出目标用 `abs(expected_return) * price` 直接乘
- 训练标签假设 entry = 下一根 bar 的 open（`label-v2`，`label_entry: next_bar_open`）

三者假设的入场点互不相同，且没有任何一步把差异显式化。

对照：hftbacktest 把 `feed_latency` 与 `order_entry_latency` 拆成两个独立参数；
NautilusTrader 有 `LatencyModel`；LEAN 把 `FillModel` 与延迟分开。

**改造**：
- 采集层为每条行情打两个时间戳：交易所事件时间 + 本地接收时间
- `FillModel` 增加 `entry_latency_ms` 参数；成交时用 `now - latency` 时刻的报价
- 训练标签同步：从 `bar[i+1].open` 改成 `bar[i+1]` 内 `latency` 时刻的价格（用 1m K 线近似）

### 1.2 【实测】bar 只有 OHLC，但止损按「恰好成交在止损价」计算

`execution_rules.evaluate_bar_exit` 明确承认 OHLC 无法确定 bar 内顺序，
采取保守策略（同时触及止损与止盈时取止损）。这是对的一半。错的一半在成交价：

```python
if stop_hit:
    return {'reason': 'stop_loss', 'price': float(position.stop), ...}   # 恰好成交在止损价
```

真实市场里止损是**市价单**，会穿价成交。实测 5m bar 的波动分布说明这个偏差不小：

```
symbol            5m bar 波动:     p50       p90       p99       max
BTCUSDT                        5.6bps   19.9bps   50.2bps   775bps
ETHUSDT                        7.5bps   26.8bps   71.7bps   663bps
DOGEUSDT                      12.2bps   39.8bps   99.2bps  5553bps
BULLAUSDT                     20.1bps   87.7bps  368.0bps  5581bps
```

在止损触发的那一根 bar 上，收盘价相对止损价的滑移就是真实损失。
用**固定 2bps 的 spread_bps** 描述它是乐观的。
freqtrade 在自己文档里承认这一点并额外计 `2 * fees`；LEAN 有独立的
`MarketImpactSlippageModel`。

**改造**：止损/强平成交价 = `min(stop, bar.close)`（多头），或改用下一根 bar 的 open。
至少要把「止损价成交」与「bar 收盘价成交」的差额单独记一个字段，让它可见。

### 1.3 【实测】止损下限 0.4% 让所有止损落在同一个宽度上，且与标签持有期不匹配

`exit_policy.py`：`stop_distance = max(atr*1.5, price*0.004)`，`target_distance = stop*1.8`。

实测训练集 `atr_pct` 分布为 0.00037–0.14，均值 0.0029。
即 `atr*1.5 ≈ 0.43%`，与下限 0.4% **同量级**。而 5m bar 的中位波动只有 5.6–20bps。
结果：

- 止损宽度 ≈ **4–7 根 bar 的标准差**
- 训练标签的持有期是 **12 根 bar = 60 分钟**
- 止损大概率在 60 分钟内根本不会触发

于是「止损/止盈」大部分情况下不生效，实际决定盈亏的是 `time_stop`（若开启）
或手动平仓。实测 94 笔已平仓交易：**中位持有 4.7 分钟 = 0.93 根 bar**，
78 笔 reason 是旧代码遗留字符串 `stop_or_target`，16 笔 manual —— **没有任何一笔能区分止损与止盈**。

**改造**：止损宽度应与标签的持有期绑定。若标签是 60 分钟，止损应放在 60 分钟内
价格可能触及的位置（用该品种自身的历史分布算），而不是 ATR 的一个固定倍数。

### 1.4 【实测】离场规则有两个来源，谁先到谁生效

`PaperAccount.mark(prices, bars)` 有两条互斥的判定路径：

```
bar 存在 → evaluate_bar_exit(OHLC 保守规则)  → 按 止损价/目标价 成交
bar 缺失 → 用 marks 里的价格比 stop/target → 按 该价格成交
```

调用点有两个，喂的东西不一样：

| 调用点 | 节奏 | 传入 | 走哪条路径 |
|---|---|---|---|
| `pumps.mark_pump` → `session.mark({sym: mark_price})` | **每 2 秒** | 标记价，**无 bars** | 第二条（价格比较） |
| `decision_loop.evaluate_bar` → `session.mark(prices, bars)` | 每根 bar | 收盘价 + bars | 第一条（OHLC） |
| `feeds_handlers` 的 mark/trade/book 事件 | 事件驱动 | 价格，**无 bars** | 第二条 |

所以 5 分钟 bar 之间，**每 2 秒**就有一个标记价去比 stop/target —— 是它，而不是
OHLC 规则，决定了绝大多数离场。实测 94 笔已平仓交易中 `reason_detail.source` 全部缺失
（旧记录），reason 全是 `stop_or_target`/`manual`，**无法区分**。

这件事和 1.3 的止损宽度叠加起来是致命的：止损宽度 ≈ 0.4%（40bp），而 5m bar 的中位
波动只有 5.6–20bp，标记价每 2 秒动一次。持仓中位只活了 **0.93 根 bar**，
说明止损几乎总是被**标记价的某个瞬时跳动**触发的，而不是被收盘价确认的。

**改造**：
1. 把两条路径显式分开：`mark_for_exit(last_price)` 与 `mark_for_liquidation(mark_price)`
2. 离场只允许一个来源。业界通行做法是用**已收盘 bar** 确认（freqtrade 全用 bar），
   或者用标记价但要求**持续 N 秒**，二者选一，写进配置
3. `reason_detail.source` 必须落库（这次审计拿不到，因为当时没记）

### 1.5 缺失的撮合真实性（逐条对照）

| 机制 | 本系统 | 行业参照 | 影响 |
|---|---|---|---|
| 订单簿逐档吃单 | `execute_depth` 有，仅在 MARKET 单时使用 | hftbacktest / Hummingbot `budget_checker` | 已有，较好 |
| 限价单**队列位置** | **无**。`process()` 只要 `reference <= limit_price` 就全额成交 | hftbacktest 队列模型（RiskAverse/Prob） | 挂单策略收益被系统性高估 |
| 部分成交 | 有（`open_fill` 加权平均） | Hummingbot `InFlightOrder` | 已有 |
| **行情/下单延迟** | **无** | hftbacktest `feed_latency`/`order_entry_latency` | 见 1.1 |
| **bar 内路径** | 保守取值但按止损价成交 | freqtrade 文档明确列出假设 | 见 1.2 |
| 强平用标记价 | 是（mark_price 事件驱动 `session.mark`） | Binance 规则 | 已有 |
| **撮合价用标记价** | `mark_pump` 每 2s 用 mark 喂 `session.mark`，`feeds_handlers` 用 trade/book 喂**同一个** `mark()` | 真实交易所：标记价触发、**最新价成交** | 止损可能被标记价触发并按标记价成交 |
| 维持保证金分档 | 有（`MaintenanceSchedule`） | — | 已有，较好 |
| 资金费 | 有，按品种各自的结算窗口，方向正确 | freqtrade `calculate_funding_fees` | 较好，但标的是 `self.marks`（可能是 trade price 而非 mark） |
| maker/taker 分档费率 | 有 | — | 已有 |
| **冲击成本 `impact_coefficient_bps`** | **默认 0.0**（`.env` 未设） | 平方根律 / Almgren | 定价与单量无关 |
| **挂单逆向选择** | 明确不建模（代码注释承认） | — | 诚实，可接受 |

### 1.6 【实测】权益曲线与绩效指标

`PaperAccount.mark()` 每次被调用就 `equity_curve.append` 一次。而 `mark()` 的调用点有四处
（mark_pump 每 2s、feeds_handlers 每个 mark 事件、bar 循环每根 bar、pumps），
采样频率是**事件驱动**的，不是等间隔的。

`metrics.summary` 把它当等间隔序列，用 `interval_ms` 年化（`periods_per_year = 105120`）：

```
/api/state 实际展示的（store.equity_curve(limit=180)，末尾 180 点）:
    跨度 = 2497 秒 (41.6 分钟)
    annualized_volatility_pct = 220.49%
    sharpe = 1.10

全量持久化曲线 (8627 点, 43.7 小时):
    annualized_volatility_pct = 45495.49%
    sharpe = 6.47
    sortino = 428.66
    max_drawdown_pct = 99.98%
```

**这些数字没有任何意义**。根因是「一个点 ≠ 一根 bar」。
实测持久化采样间隔的中位数是 **1 毫秒**（6,807 次），最大 10 秒以上（1,767 次）。

**改造**：
1. `equity_curve` 改为按 bar 边界降采样（每个 interval 保留最后一个值），并同时存时间戳
2. `summary` 由时间戳推导真实周期数，而不是用配置的 interval
3. 或给 `PaperAccount.mark()` 加 `record=False` 默认值，只在 bar 收盘时记录

### 1.7 【实测】资金费的标的价不是标记价

`PaperAccount.apply_funding`：
```python
amount = self.marks.get(symbol, p.entry) * p.qty * float(rate) * (1 if LONG else -1)
```
`self.marks` 被四个不同来源写入（mark_price / trade price / book mid / bar close）。
Binance 用**标记价**结算资金费。当 `marks` 里是 trade price 时，8 小时结算一次的费用会有偏差。

**改造**：`marks` 拆成 `last_price` 与 `mark_price` 两个字典 ——
资金费用后者，止损触发用前者，强平用后者。

### 1.8 【实测】会计恒等式与部分成交残留

补上本次审计的最后一块（模拟保真度）。

**做对的部分**：`open()` / `open_fill()` / `close()` 三条路径的现金流是自洽的。
实测一笔 1 手 100 的多单：

```
open()   : cash 10000 -> 9999.96   (扣 entry_fee 0.04)
close()  : cash 9999.96 -> 10000.899408   delta = +0.939408
           恒等式 gross_pnl - exit_fee = 0.9798 - 0.040392 = +0.939408  OK
round trip: cash - initial_cash = 0.899408
            sum(trade.pnl)       = 0.899408   OK
            ledger 求和           = 0.899408   OK
```

**重复计费不存在**：`open()` 扣一次 `entry_fee`，`close()` 只在报表里把它汇总
（`total_fees = exit_fee + p.entry_fee`），不再从 `cash` 里扣第二次。三处口径一致。
这一块比多数个人项目做得好。

**部分成交的加权是对的**：0.6 100 后再 0.9 100.5 → qty=1.5、entry=100.30
（与 `(0.6*100+0.9*100.5)/1.5` 完全一致），`entry_fee` 累加；同 symbol 上
**不同 order_id 的成交被正确拒绝**；`order_state.status_for_fill(quantity, filled_quantity)`
也正确（1.0 / 0.5 → PARTIALLY_FILLED）。

**仍缺两件**：

1. **`close()` 是全有全无**（`positions.pop(symbol)`），没有按 fill 分 lot。
   所以**分批止盈 / scale-out 无法表达** —— 而这正是元标签体系里最需要的离场形式。
   三重障碍标签的「上障碍部分减仓」在当前账本上写不出来。
2. **`order.filled_quantity` 与 `position.qty` 从不互相对账**。
   一个 PARTIALLY_FILLED 的订单随后 EXPIRED（`is_open(PARTIALLY_FILLED) = True`），
   仓位留着，但**没有任何记录说明这次建仓从未完成**。
   风控按 `position.qty` 计算敞口，而订单侧认为自己只成交了一部分 ——
   两者在 Hummingbot 的 `InFlightOrder` + `PositionHold` 里是被强制对齐的。

**改造**：
- 仓位记录 `lots: [(price, qty, fee, fill_id)]`，`close()` 支持 `quantity` 参数
- 加 `reconcile_order_position()`：按 `order_id` 分组比对
  `sum(order.filled_quantity)` 与 `position.qty`，不一致就报事件并拒绝新开仓


---

## 2. 交易决策（Decision）的缺陷

### 2.1 【实测·最严重】训练报告的 active 集合 ≠ 线上门槛的 active 集合

**训练侧**（`tabular_model.evaluate`）：
```python
side = np.where(prediction > cost, 1, np.where(prediction < -cost, -1, 0))   # cost = 12bp
```
所以训练报告里的「active」= **预测收益超过 12bp 的样本**。

**线上侧**（`live_models._assemble`）：
```python
edge_bps = abs(expected) * 10000
elif edge_bps < self.min_edge_bps:   # min_edge_bps = 0.5
```
门槛是 **0.5bp**，而且取了 `abs()` —— **符号丢了，成本项完全不存在**。
（`TabularMember.predict` 其实返回了 `expected_net_return = abs(value) - cost`，
但 `_assemble` 只读 `expected_return`，把成本项丢弃了。）

在真实权重上复现两个集合：

```
selection rule                              n     gross bps    net bps
training report: pred > +12bp  (LONG)        77      +82.60     +70.60
training report: pred < -12bp  (SHORT)      192      -13.39     -25.39
LIVE: pred > +0.5bp  (LONG)                1339       -2.36     -14.36
LIVE: pred < -0.5bp  (SHORT)             187182       -1.95     -13.95

训练报告 active = 269 行（占 0.143%）
线上门槛 active = 188,521 行（占 99.94%）   ← 700 倍
```

逐品种也一样（线上门槛下，12 个品种全部为负）：

```
ADAUSDT   15707 笔   -15.01 bp/笔
AVAXUSDT  15688 笔   -14.07 bp/笔
BTCUSDT   15718 笔   -13.59 bp/笔
TRXUSDT   15718 笔   -12.40 bp/笔   ← 最好的品种仍然是负的
XRPUSDT   15717 笔   -13.55 bp/笔
```

在 188,521 个样本上下注：**净 −13.95bp/笔，t = −102.55，命中率 33.06%**。

> **为什么现在看不出来**：当前线上 `feature_degraded` 把所有 bar 都拦掉了（见 2.3），
> 所以从来没有真正下过这些注。**一旦补上数据采集解除 2.3 的死锁，**
> **系统会立刻开始以 −13.95bp/笔 的期望下注。这是本次审计最重要的一条警告。**

**改造（最小改动、最大收益）**：
1. 把 `MODEL_MIN_EDGE_BPS` 从 0.5 提到 **12**（等于训练成本阈值）—— README 已写了这句话，但配置没做
2. 更正确的做法：`_assemble` 改用成员自己的 cost-gated `side`（`FLAT` 当 `|value| < cost`），
   而不是 `abs(expected) > 0.5bp`
3. 加一条断言：把线上门槛函数导出，让它与 `evaluate()` 共用同一个 `select()`

### 2.2 【实测】模型方向退化：99.3% 输出 SHORT

在留出集上：

```
lightgbm 预测分布: mean=+0.0001   std=0.0006
  只有 0.74% 的预测为正 → 其余 99.26% 为负
  pred > +12bp: 77 个      pred < -12bp: 192 个

逐品种预测标准差（bps）:
  ADAUSDT 6.98   DOTUSDT 8.43   AVAXUSDT 8.47   LINKUSDT 8.20   SOLUSDT 7.73
  BTCUSDT 0.27   ETHUSDT 0.44   BNBUSDT 0.47    TRXUSDT 0.24
```

这不是「集成投票」，这是**单边过滤器**。训练报告里 `active_net_edge_bps` 看起来是正的，
是因为它同时包含 77 个 LONG 和 192 个 SHORT，而 **192 个 SHORT 的平均实际收益是 +13.39bp**
（即模型做空，市场却涨了）。整个「正 edge」由那一小撮 LONG 撑着。

**根因**：`mark_basis` 在 11.8% 的行上被钉在 ±2% 截断值上。实测：

```
数据集 mtime: 2026-09-10 16:23   size=578.6 MB
按 1/4 抽样 314,586 行:
  mark_basis      mean=0.000335   min=-0.02   max=0.02   at_clip=11.81%
  funding_rate    mean=0.0000242  min=-0.000365  max=0.000574

唯一持有 5 分钟粒度标记价的 derivatives_detail:
  span = 2026-09-11 22:55 -> 2026-09-12 01:50  (2.92 小时)
```

**数据集是在 `mark_basis` 修复之前构建的**，当时用的是资金费时间轴（8 小时一次）当标记价，
所以 `mark_basis` 实际算的是「距上次结算以来的涨跌幅」。十分之一的数据把一个常数喂给模型，
模型于是学到了一个方向性极强但无意义的偏置（在真实数据集上 `mark_basis` 被截断的比例
实测 11.81%，与另一位审计者独立测得的 13.93% 一致）。

**改造**：重建数据集 → 重新训练 → 重新评估。这一步不可跳过。

### 2.3 【实测】当前实盘 100% FLAT，且是双重死锁

从审计日志实测（`events` + `events_archive`）：

```
order_rejected 12,706 条:
  stale_market_data          11,860  (93.3%)
  missing_book_time             335
  max_positions                 205
  portfolio_ok                  141
  max_symbol_notional           129
  risk_budget_exhausted          29
order_filled 97 条

stale 拒绝的时段分布：09-10 13h 100%、16-23h 92.9-98.9%、09-11 00h 78.6%、
09-11 03h 之后全部为 0%（服务停止后审计里只剩策略行）
```

**93% 的拒绝是 `stale_market_data`**。而 `MarketGuard` 依赖
`runtime.guard.observe(symbol, event_now)`，唯一调用点在 `feeds_handlers.handle_ws_event`。

更糟的是死锁链：

```
数据缺失（15/30 个 v4 特征无源）
   → FeatureSource 标记 degraded
   → FEATURE_DEGRADED_POLICY=block
   → _assemble 返回 FLAT（live_models.py:788，在方向判断之前）
   → SymbolEdge.observe 只在 side ∈ (LONG, SHORT) 时才记录
   → symbol_edge 永远 0 样本
   → require_samples=True → allows() 恒返回 False
   → 即使数据补齐，前 30 个信号也全被拒
```

顺带一个 **gate 顺序 bug**：`live_models.py:788` 的 `feature_degraded` 在 `elif` 链的
**最前面**，所以 `edge_below_floor` 的测量永远不会执行。

### 2.4 门槛体系的「有效性」审计

| 门槛 | 默认值 | 被测量校准过？ | 实际是否生效 |
|---|---|---|---|
| `provenance` | `MODEL_REQUIRE_PROMOTED=0` | 否 | **空操作**（`data/models/` 为空） |
| `model_expired` | `MODEL_MAX_AGE_HOURS=0` | 否 | **空操作** |
| `feature_degraded` | `block` | 否 | **全拦**（见 2.3） |
| `out_of_distribution` | `block` | 部分 | 对 `mark_basis` **空操作**（边界=截断范围）；对 20 个 v4 特征**结构上不存在** |
| `agreement` | `0.6` | 否 | **2 成员时是空操作**（只能取 0/0.5/1.0，而 0.5 已被 `no_directional_edge` 拦掉）；3 成员（含 Chronos）时才有意义 |
| `edge_below_floor` | `0.5bp` | **否** | 实测成员 |pred| 中位 ≈ 1.0bp → **近乎空操作** |
| `min_edge_multiple` | `2.5` | 否 | 见 2.5 |
| `symbol_edge` | `min_t=2.0` | 是 | 在 2.3 的死锁下**恒拒** |

### 2.5 计划几何：门槛与模型边际解耦

```python
expected_move = max(levels['target_distance'], abs(expected_return) * price)
if expected_move < round_trip * self.min_edge_multiple:   # 12bp * 2.5 = 30bp
    return None
```

`target_distance = stop_distance * 1.8`，`stop_distance = max(atr*1.5, price*0.004)`。
对 BTC（`atr_pct` 中位约 0.15%）：`stop` 被 0.4% 下限托住，`target = 0.72%` = 72bp > 30bp → 通过。
对高波动山寨（`atr_pct` = 2%）：`stop = 3%`，`target = 5.4%` → 也通过。

所以这个门槛实际上从来不拦 —— 因为 `levels` 是**几何**决定的，与模型的 edge 无关。
模型只被允许把目标**推远**，永远不能因为边际不足而被拒绝。这是 U 形不对称。

**改造**：门槛应作用在「模型预测的边际」上（`abs(expected_return)*price`），
而不是作用在「止损宽度的 1.8 倍」上。几何决定的是风险，不是收益。

### 2.6 仓位与 edge 完全无关

`risk.size()` 的输入是 `entry`、`stop`、`equity`、`open_notional`、`open_risk`、
`symbol_notional`、`vol_scale`。

**没有 `edge`，没有 `confidence`，没有 `probability_up`。**

实测：一个 0.5bp 的边际和一个 50bp 的边际，会得到**完全相同的仓位**。
`probability_up`（已校准概率）现在被计算、被传输、**被丢弃**（代码注释自己承认了这一点）。

**改造**：仓位改成 Kelly-fraction 形式
```
f* = edge_bps / (stop_distance_bps * k)        # k 由历史胜率/盈亏比校准
quantity ∝ clip(f*, 0, max_risk_per_trade) * equity / stop_distance
```

### 2.7 标签与验证：进步很大，但仍缺四件

**已经做对的**：`label-v2` 从 `bar[i+1].open` 起算，避免了「用产生信号的同一价格成交」；
`dataset_split` 有 label purge（manifest 报 288，与 12 品种 × 12 bar × 2 边界一致）；
`walkforward` 有 purge gap 与 moving-block bootstrap 置信区间。这些都优于大多数个人项目。

**仍缺**：
1. **无三重障碍标签**（triple-barrier）。当前是固定 12 根 bar 的远期收益，
   与实盘「1.5 ATR 止损 / 1.8RR 目标 / 跟踪 / 时间止损」**不是同一个分布**
2. **无样本唯一性权重**。相邻样本共享 12/13 的收益窗口，被当作独立样本。
   实测标签 lag-1 自相关 **0.557**，naive t=14.0 会塌到 t≈5.15（约 3–33 倍有效样本缩水）
3. **无 embargo**。`walkforward` 的 early-stopping 验证集紧贴测试块，没有间隔
4. **无 Deflated Sharpe / PBO**。参数搜索（`horizon`、`MIN_EDGE_MULTIPLE` 等）没有试验次数记录

---

## 3. 缺了什么（机制清单，按 ROI 排序）

| # | 缺失机制 | 参照实现 | 为什么重要 |
|---|---|---|---|
| 1 | **线上/离线门槛同构** | — | 见 2.1，唯一的「立刻在亏钱」级别问题 |
| 2 | **延迟模型（行情/下单分离）** | hftbacktest `feed_latency` / Nautilus `LatencyModel` | 见 1.1 |
| 3 | **三重障碍标签 + 元标签** | López de Prado AFML ch.3/7 | 标签分布必须与被交易的策略同分布 |
| 4 | **样本唯一性权重 + PurgedKFold + embargo** | `timeseriescv` / `mlfinlab` 系列 | 见 2.7 |
| 5 | **Deflated Sharpe / PBO** | `pypbo` / quantstats `probabilistic_sharpe_ratio` | 记录试验次数，防止选到噪声 |
| 6 | **限价单队列位置模型** | hftbacktest 队列模型 | 若要做挂单策略 |
| 7 | **风控 Protections（分品种熔断）** | freqtrade `StoplossGuard/MaxDrawdown/LowProfitPairs/Cooldown` | 最易移植到逐品种门槛体系 |
| 8 | **组合层（波动率目标/相关性/风险平价）** | LEAN `IPortfolioConstructionModel` | `max_correlated_leverage` 默认 0，是关的 |
| 9 | **对账 + 持仓锁（PositionHold）** | Hummingbot `ExecutorOrchestrator` | 有 `account_reconcile` 但不阻塞下单 |
| 10 | **灾难避险指数（turbulence）** | FinRL `add_turbulence` | 系统级回撤领先指标 |
| 11 | **可插拔 Fill/Fee/Slippage/Margin/Latency 模型类** | LEAN 六类模型 | 已有 Fee+Fill，缺 Latency/Settlement |
| 12 | **横截面因子** | Qlib Alpha158 | 当前只有时序特征，没有全市场排名/相对强弱 |
| 13 | **基差/价差交易** | vnpy `SpreadTrading` | 资金费套利这条线完全没有 |

---

## 4. 该拉什么数据

### 4.1 【实测】当前采集的覆盖度

```
derivatives_detail (OI / 多空比 / taker 比):
  rows=432  symbols=13  span=2.92 小时
  30 天 × 5m × 12 品种本应有 103,680 行
  → 当前覆盖 = 一个月窗口的 0.04%

flow 表 (逐分钟主动方失衡 + 强平):
  rows=0
  FLOW_COLLECT_ENABLED=1，!forceOrder@arr 已订阅，
  FlowCollector 在 main.py:341 已构造 —— 表是空的

candles 的订单流列 (trades/quote_volume/taker_buy_volume):
  2,749,163 根 5m K 线中只有 2,792 根有 trades (0.10%)
```

**v4 契约 30 个特征中，15 个（8 positioning + 7 flow）依赖基本没采到的数据。**
这是「过期即永久丢失」的数据 —— Binance `/futures/data/*` 只保留 30 天。

### 4.2 训练该拉什么

**A. 立刻补（免费、且过期不可回补）**

| 数据 | 端点 | 历史深度 | 用途 |
|---|---|---|---|
| **5m 衍生指标归档** | `data.binance.vision` 的 `futures/um/daily/metrics/` | 与 UM 同寿 | **一劳永逸解决 30 天限制** |
| OI 实时轮询 | `/futures/data/openInterestHist` | 30 天 | 回补归档之外的当前窗口 |
| 大户持仓多空比 | `/futures/data/topLongShortPositionRatio` | 30 天 | 比账户数更接近真实仓位 |
| 大户账户多空比 | `/futures/data/topLongShortAccountRatio` | 30 天 | 情绪 |
| 全局多空比 | `/futures/data/globalLongShortAccountRatio` | 30 天 | 散户拥挤度；与大户差 = Smart-Retail Gap |
| Taker 买卖比 | `/futures/data/takerlongshortRatio` | 30 天 | 官方口径订单流 |
| **K 线 CSV 补齐三列** | `futures/um/monthly/klines/` 里本来就有 `taker_buy_base_asset_volume` / `number_of_trades` / `quote_asset_volume` | 2019-09 起 | **零成本**把 279 万根 K 线的 NULL 补上 |
| 逐笔成交 | `futures/um/daily/aggTrades/` | 2019-09 起 | CVD、大单占比、平均单笔额 |
| 最优挂单 | `futures/um/daily/bookTicker/` | 2019-09 起 | 价差、microprice 偏离 |
| 资金费历史 | `/fapi/v1/fundingRate` | 合约上线起 | 已有；需按品种取真实结算周期 |
| 期权 IV / DVOL | Deribit `public/get_volatility_index_data` | 多年，免费无 key | IV-RV 价差、25Δ skew 作 regime |
| 恐贪指数 | alternative.me `api.alternative.me/fng/?limit=0` | 2018 起 | 日频 regime |
| 宏观 | FRED / Stooq CSV | 多年 | DXY/NDX/10Y/VIX 日频 |

**A2. 归档的精确路径与语义（已核验）**

`data.binance.vision` 是一个 S3 桶浏览器，可以用 S3 `ListObjects` 语义程序化枚举：
```
BUCKET  = https://s3-ap-northeast-1.amazonaws.com/data.binance.vision
列目录  = {BUCKET}?delimiter=/&prefix=data/futures/um/daily/klines/BTCUSDT/
路径模板 = /data/futures/um/{daily|monthly}/{type}/{SYMBOL}/[{interval}/]
           {SYMBOL}-{type}[-{interval}]-{YYYY-MM-DD|YYYY-MM}.zip
每个 zip 有配套 .CHECKSUM (sha256)；updates/ 目录是追溯修订的变更日志
```

实测可用（HTTP 探测）：`klines`(1m/5m/…)、`aggTrades`、`trades`、`metrics`、
`indexPriceKlines`、`markPriceKlines`、`premiumIndexKlines`。

三个**必须知道的归档语义**：

1. **`metrics` 只有日频，没有月频**（`monthly/metrics/...` 返回 404）。
   它包含 `sum_open_interest`、`sum_open_interest_value`、
   `count_toptrader_long_short_ratio`、`sum_toptrader_long_short_ratio`、
   `count_long_short_ratio`、`sum_taker_long_short_vol_ratio`，
   **粒度 5 分钟、可回溯数年** —— 这是绕开 `/futures/data/*` 30 天墙的唯一免费途径。
2. **归档文件会被追溯替换**（`updates/` 里有变更日志）。所以「今天建的数据集」
   与「一年前建的数据集」可能不同。要记录你消费过的 checksum。
3. **没有强平归档**（`liquidationSnapshot` 返回 404）。强平只能自己录，或向
   Coinglass / Amberdata / Kaiko / Tardis 购买 —— 而且它们的 Binance 部分同样是
   基于被限流的 `!forceOrder`，绝对量偏小，只能用相对变化。

**A3. 「过期即永久丢失」清单（第一天就要开始录）**

| 数据 | 有免费归档？ | 不录就永久丢失？ |
|---|---|---|
| K 线 / aggTrades / trades / metrics / 各种 mark-index-premium K 线 | **有** | 否，可回补 |
| 5m OI + 多空比 + taker 比（`metrics`） | **有** | 否 |
| `/futures/data/*` 的 5m–1d 实时值 | 只有最近 30 天 | 更细于 5m 的粒度**丢失** |
| **1s–1min 的 OI** | 无 | **永久丢失** |
| **全量增量深度 L2（100ms）** | 无 | **永久丢失** |
| **强平逐笔** | 无 | **永久丢失** |
| **最优挂单逐笔（bookTicker）** | 部分 | 大部分丢失 |
| 资金费结算历史 | 有 | 否 |
| **资金费周期变更史**（`fundingInfo`） | 只有当前值 | 不每日快照则丢失 |
| **杠杆档位 / 过滤器 / 手续费率** | 只有当前值 | 不每日快照则丢失 |
| **ADL 风险评级**（`symbolAdlRisk`，每 30 分钟更新） | 只有当前值 | 丢失 |
| 保险基金余额 | 只有当前值 | 丢失 |
| Deribit 完整期权链（含 bid/ask IV） | 无（DVOL 可取） | 丢失 |
| 宏观 actual-vs-consensus、Google Trends、恐贪、社交 | 无 / 会被重述 | 丢失或受污染 |
| **你自己的订单/成交/延迟/滑点/拒单** | 只有你能产生 | 丢失 |

**D. 实时决策的时效容忍度（必须写进代码，而不是靠自觉）**

| 数据 | 传输 | 节奏 | **时效上限** | 超时的后果 |
|---|---|---|---|---|
| mark / last / index | WS `@markPrice@1s`、`@aggTrade` | 事件驱动 | **< 250 ms** | 强平与资金费算错 |
| 最优买卖价 + 量 | WS `@bookTicker` | 事件驱动 | **< 100 ms** | 价差/滑点估计失真 |
| L2 增量簿（前 20 档） | WS `@depth100ms` + REST 重同步 | 100ms | **< 200 ms** | 簿静默损坏 |
| OI（当前值） | REST `/fapi/v1/openInterest` | 1–10 s | **< 5 s** | 慢于 5m 归档就没有增量信息 |
| 资金费（预测值 + 下次时刻） | REST `/fapi/v1/premiumIndex` | 1–5 s | < 60 s | carry 定价错 |
| 强平 | WS `!forceOrder@arr` | 事件驱动 | **< 500 ms** | 瀑布特征失明 |
| 跨所资金费/OI/基差 | REST bybit/okx/hyperliquid | 1–10 s | **< 5 s** | 拥挤度读数过期 |
| Deribit 期权面 | REST `get_book_summary_by_currency` | 15–60 s | ≤ 5 min | 波动率特征过期 |
| 交易所过滤器/杠杆档位 | REST `exchangeInfo`、`leverageBracket` | 每日 + 告警 | 小时 | 下单被拒 / 仓位算错 |
| 账户状态 | user-data WS + REST | 事件驱动 | **< 1 s** | 风控失效 |
| **时钟偏移** | `/fapi/v1/time` | 每 60 s | **< 50 ms** | 签名请求被拒、特征错位 |

**通用规则**：每个响应持久化 `(本地接收时间, 交易所时间, 原始载荷哈希)`；
每个特征自带 `as_of` 与 `max_staleness`；**任何一个关键特征超龄，就退化为不下单，**
**而不是拿一个过期的向量去交易**。

**E. 本项目特有的四条陷阱（在通用陷阱之外）**

1. `mark_basis` 的历史实现用**资金费时间轴**（8 小时）当标记价时间轴，
   已经污染了当前在用的数据集（见 2.2）。重建时标记价必须走 `markPriceKlines` 或
   `metrics` 的 `sum_open_interest_value / sum_open_interest`，不能复用资金费的下标。
2. `/fapi/v1/fundingInfo` 的 `fundingIntervalHours` 在 2023–2025 之间发生过
   8h → 4h → 1h 的变更。代码里 `funding.py` 的 `RECORDS_PER_DAY = 3` 是硬编码的 8h 假设，
   而 `apply_funding` 已经按品种取窗口 —— 两处口径不一致，年化 carry 会算错。
3. 数据的 **`E`（事件时间）/ `T`（成交时间）/ kline `openTime` / 本地接收时间** 四个时钟
   目前没有区分。跨所 join 必须用保守滞后（接收时间或显式 delta），否则会把
   快的那家交易所的未来信息引进来。
4. `!forceOrder@arr` 每个 symbol **每 1000ms 最多一条**，且只推最大的那条。
   任何「强平总量」类特征都是**下界**，只能用来做相对变化，不能做绝对阈值。

**B. 特征族（当前完全没有的高价值项）**

- **订单流/微观结构**：`trade_imbalance_1m`（用 `m` 字段）、`cvd_z`、`large_trade_ratio`、
  `spread_bps`、`microprice_dev`、`depth_imb_l20`、`ofi_top5`、`amihud_illiq`、`vwap_dev`
- **资金费与基差**：真正的 `basis = mark/index - 1`（当前 `mark_basis` 是滞后动量）、
  `basis_z_7d`、`annualized_funding`、**`mins_to_funding`**、`predicted_funding_rate`
- **持仓量**：`oi_chg_5m/1h/24h`、`oi_price_quadrant`、`oi_notional_z`、`oi_vol_ratio`、`smart_retail_gap`
- **强平**：`liq_long_usd_5m`、`liq_imbalance`、`liq_z_24h`、`cascade_risk = 强平额/OI`、`mins_since_big_liq`
- **波动率状态**：`rv_5m/1h/24h`、`rv_ratio`、`yang_zhang`、`vol_of_vol`、`dvol_chg`、`iv_rv_spread`、`skew_25d`
- **横截面**：`beta_btc_1h`、`resid_mom`（剔除 BTC 的残差动量）、`xs_mom_rank`、`btc_dominance_chg`

**C. 标签**

把固定 12 根 bar 的远期收益换成**三重障碍**：
```
上障碍  = entry * (1 + k_up   * sigma_t)     # sigma_t = 该 bar 的 EWMA 波动
下障碍  = entry * (1 - k_down * sigma_t)
时间障碍 = entry_time + max_holding_ms
label   = 先触发的那个障碍的带符号收益
weight  = 平均唯一性（average uniqueness，AFML 4.3）
```

### 4.3 决策该拉什么（实时）

| Stream | 频率 | 内容 | 现状 |
|---|---|---|---|
| `<sym>@kline_1m` | 250ms | 未收盘 K 线 | 已订阅（当前 5m） |
| `<sym>@markPrice@1s` | 1s | mark、index、funding、nextFundingTime | 已订阅 |
| `<sym>@bookTicker` | 实时 | 最优买卖价+量 | 已订阅（另一条连接） |
| `!markPrice@arr@1s` | 1s | 全市场 mark 数组 | 未订阅（可替代逐订阅，省连接） |
| `<sym>@aggTrade` | 逐笔 | 价格/量/`m` | **已订阅，未落库到 candles** |
| `!forceOrder@arr` | 实时 | 全市场强平 | **已订阅，表是空的** |
| `!miniTicker@arr` | 1s | 全市场 24h 统计 | 未订阅（现用 REST 每 5s） |
| `<sym>@depth100ms` | 100ms | 增量深度 | 未订阅 |

**必须先修的一件事**：`!forceOrder@arr` 与 `aggTrade` 已经在订阅，但 `flow` 表是 0 行，
说明 `FlowCollector.drain()` 这条链断了 —— 要么 drain 没被调用到，要么 drain 之后没有落库。
这是「订了却没用」的第二次复发（第一次是 `aggTrade` 没用上，代码注释里已承认）。

### 4.4 数据质量陷阱（逐条必须处理）

1. `/futures/data/*` 只保留 30 天；`/fapi/v1/openInterest` **无历史**
2. K 线 `closeTime = openTime + interval - 1ms`；特征必须用已收盘 bar
3. **资金费周期不统一**（8h / 4h / 1h）→ 年化必须按品种的间隔算
4. 成交量口径：ticker 的 24h volume 是滚动值，K 线 volume 是区间值，`quoteVolume` 才是 USDT 额
5. 限频：`/fapi` 2400 weight/min/IP，429 指数退避、418 封 IP
6. 合约生命周期：下架/改名、`onboardDate` 之前无数据 → 必须用 point-in-time universe
7. 强平流 `!forceOrder@arr` **不是全量**（部分走 ADL/内部撮合）→ 强平量是**下界**
8. 统一 UTC，禁止本地时区/DST 进桶

---

## 5. 优秀开源项目可借鉴之处

| 项目 | 最值得抄的一点 | 本系统对应 |
|---|---|---|
| **hftbacktest** | L2/L3 重放 + **队列位置模型** + **行情延迟与下单延迟分离** | 延迟模型完全没有；限价单无队列 |
| **NautilusTrader** | 策略只发意图，回测/实盘换 `ExecutionClient`；`LatencyModel` | 已有 `backtest.py --models` 复用同一 `ModelDecision`，方向是对的 |
| **LEAN** | **Fill/Fee/Slippage/Margin/BuyingPower/Settlement 六类可插拔模型** | 已有 `fill_models.py`（Fee+Fill），缺 Latency/Settlement |
| **freqtrade** | ① `calculate_funding_fees` 用 **funding rate × mark price** 向量化区间查询；② **lookahead 分析器**（对每笔交易截断数据重跑，diff 指标）；③ `DelistFilter` + `check_delisting_time`；④ protections（`StoplossGuard/MaxDrawdown/LowProfitPairs/Cooldown`）；⑤ `VolatilityFilter`/`SpreadFilter` pairlist | 资金费已较好；**lookahead 分析器值得直接移植**；protections 可补 |
| **Hummingbot** | `InFlightOrder` 订单 FSM + 幂等 `clientOrderId` + `BudgetChecker` + `PositionHold` | `order_state.py` 已有 FSM；缺 PositionHold |
| **vnpy** | `SpreadTrading` 价差/基差交易应用；`RiskManager` 规则插件化 | 基差交易完全没有 |
| **cryptofeed** | 归一化 L2/成交/资金费/持仓量/强平流 | 已有部分 |
| **Qlib** | Alpha158/Alpha360 表达式因子引擎 + `RollingGen` 走步 | 只有时序特征，无横截面因子库 |
| **vectorbt / quantstats** | `Splitter.rolling_split` 参数热力图；`probabilistic_sharpe_ratio` | 无参数高原检验 |
| **FinRL** | `add_turbulence` 崩盘避险指数 | 无系统级风险指数 |
| **Jesse** | 诚实的蜡烛级回测 + 研究环境 | — |

（详表见 `docs/quant_oss_landscape_report.md`。）

**已在系统里、且比多数开源项目更好的部分（不要动）**：
`margin.py` 的交易所分档维持保证金表、纸面强平写零并记 shortfall、
`apply_funding` 的逐品种结算窗口、`order_state` 的 FSM、
`dataset_split` 的 label purge、`walkforward` 的 moving-block bootstrap。

---

## 6. 改造路线图

### P0 —— 立刻（不改这些，后面全是白做）

> 状态：**1–7 已全部完成**（本轮分层重构一并落地）。每一项都有对应的回归测试。

1. ✅ **修门槛同构**：`_assemble` 现在算 `net_bps = abs(expected)*10000 - cost_bps`，与训练
   的 `side = pred > cost ? LONG : pred < -cost ? SHORT : FLAT` 逐行等价。解析验证：在
   4,006 个点上两条规则**零分歧**。`tests/test_cost_contract.py`
2. ✅ **修 gate 顺序**：`if/elif` 链改成「全部求值、再判定」，`blocked_by` 收集所有原因。
   此前 `feature_degraded` 短路了 `edge_below_floor`，导致审计无法回答「边际地板本来会挡掉
   多少 bar」——正是区分数据问题与模型问题所需的那个数。`tests/test_cost_contract.py`
3. ✅ **修 `flow` 落库**：链路本身是通的（实测：`observe` → `drain` → `record_flow` 写出 1 行）。
   表为空的唯一原因是**服务自该采集器上线后没有跑过**（库里最新事件是 2026-09-12）。
   现已用 `record=False` 把 tick 与 bar 边界分开，重启一次即可看到数据流入。
4. ✅ **修权益指标**：`mark()` 新增 `record` 参数，**只有 bar 边界记录采样点**；2 秒的 mark
   pump 与每个 feed 事件都传 `record=False`。新增测试证明：同一价格路径下，每 bar 采 1 个点
   与采 200 个点**得到完全相同的波动率与 Sharpe**。`tests/test_equity_and_marks.py`
5. ✅ **修 2.5 的门槛作用对象**：`min_edge_multiple` 现在作用在**模型自己的预期位移**上。
   此前比较的是 `max(几何目标, 模型预期)`——而几何目标几乎总是更大，于是这个门槛量的是
   止损/止盈比（一个常数），不是模型边际，因此**恒为通过**。
6. ✅ **统一离场来源**：新增 `exit_on_tick` 开关，**默认 `False`**（bar 确认，freqtrade 的
   做法，也是回测唯一能复现的规则）。`reason_detail.source` 现在必填（`'bar'` / `'tick'`）。
   强平不受影响——偿付能力不是策略选择。`tests/test_exit_source.py`（6 项）
7. ✅ **拆开 `marks`**：新增 `mark_prices` 字典。资金费、强平、`equity`、`available_margin`
   全部改用标记价；`marks`（最新成交价）只负责止损/止盈触发。快照往返与平仓清理都有测试。
   `tests/test_equity_and_marks.py`

### P0.5 —— 重构过程中新发现并已修复

8. ✅ **`/metrics` 对每一次抓取都返回 500**（实测复现，非本次重构引入：备份中的代码一字不差）。
   `web.metrics` 用 `content_type='text/plain; version=0.0.4; charset=utf-8'` 构造响应，
   而 aiohttp 会抛 `ValueError("charset must not be in content_type argument")`——charset
   必须作为独立参数。这是整个系统里**最安静的一种故障**：不触发告警、不影响面板，唯一症状
   是一张没人看的空图。实测：修复前 500，修复后 200 + 103 行。`tests/test_metrics_endpoint.py`
9. ✅ **`core/` 值对象归位**：`ContractSpec` 与 `MaintenanceTier` 从 `trading/` 移到 `core/`。
   行情层为了用它们 import 了下单层，是把环藏在函数内 import 后面的根源。
10. ✅ **删除死代码**：`MarketDataConnector` / `ExecutionConnector`（零引用的 Protocol）、
    `LedgerEntry`、`PERIODS`、`STAGES`、`UNFILLED_TERMINAL_STATUSES`、`is_terminal`、
    `candle_order_flow`、`net_cashflow`，以及 16 处未使用的 import。
    `directional_outcomes` 没有删——它被内联重复在 `fit_directional` 里，改为真正接上，
    两条规则合一。复查：**未引用模块级名字 0 个**。

### P1 —— 下周

11. **重建数据集**（`mark_basis` 代码已修，数据集还是旧的）→ 重训 → 重评
12. ✅ **补 K 线三列**：从 `data.binance.vision` 的 `futures/um/monthly/klines/` 回补
    `taker_buy_volume` / `quote_volume` / `trades` —— 代码已做并实测跑通，见「本轮已完成（P1 #12、#13）」
13. ✅ **接 `futures/um/daily/metrics/`**，把 OI/多空比的历史从 2.92 小时提到数年
    （注意：`metrics` **只有日频归档**，没有月频；粒度 5 分钟）—— 已做，见同节
14. **开始录「过期即丢失」的数据**（见 4.2 A3）：`!forceOrder@arr` 落库、
   `/fapi/v1/openInterest` 1–10s 轮询、`@depth100ms` 增量簿 —— **今天不录，明天补不回来**
15. **每日快照会漂移的静态数据**：`exchangeInfo`（含 `onboardDate`/`deliveryDate`）、
    `leverageBracket`、`fundingInfo`（`fundingIntervalHours`）、`insuranceBalance`、
    `symbolAdlRisk`（每 30 分钟）
16. ✅ **仓位分 lot + `close(quantity)`**（见 1.8）—— 已做，见下

### P2 —— 一个月

17. ✅ **延迟模型**：`FillModel.entry_latency_ms` —— 已做，见下
18. ✅ **三重障碍标签 + 唯一性权重** —— 已做，见下
18b. ✅ **walkforward 的 fit/验证集 purge** —— 已做，见下。原先测的是 **0 bar** 空隙
    而标签向前伸 12 bar，早停轮数是在「已经背下来的行」上选的
19. ✅ **Deflated Sharpe / PBO** 进训练报告 —— 已做，见下
20. ✅ **仓位接 edge 与 `probability_up`** —— 已做，见下

### 本轮已完成（P1 #12、#13、P2 #18b）

这一轮的关键前提是**网络通了**。但通了之后才发现，`data.binance.vision` 本身的可靠性
远低于它的名声，而它不可靠的方式恰好会**静默丢数据**。三件事按发现的顺序记下来。

#### 发现一：CDN 会对**存在**的归档返回 404，且持续数分钟

同一个 URL（2024 年发布、之后不可变的归档），curl 和 Python 都在 200 与 404 之间反复：

```
20 次连续请求全部 200  →  几分钟后 12/12 全部 404  →  再几分钟后又是 200
```

两个客户端在同一时刻结论一致，所以不是请求的问题。这直接决定了回补代码的写法：
**一个 404 不能当作「归档不存在」**。若采信第一次 404，回补会跳过整月、并报告成功 ——
这是这个任务能犯的最坏的错（它的全部意义就是补洞）。因此：

- 404/403 走指数退避重试（累计约 65s）后才认定为缺失；
- 连接失败**永远不等于**缺失，直接抛 `ArchiveError`（见 `test_a_connection_failure_raises_instead_of_reporting_absence`）；
- 驱动脚本里，缺失**不写入 done 列表**，只记 `unresolved` 供下一轮重试。

这一条不是理论推演：第一版驱动脚本确实把 `BTCUSDT:2025-09` 判成 `MISS` 并永久标记完成，
而几分钟后同一文件用 `fetch` 一次就拿到了 388,679 字节。那个假记录已清除。

#### 发现二：四个 CDN 边缘节点里有一个是死的，而 Python 只试第一个

```
13.35.190.101  -> 200  388,679 bytes  0.4s
13.35.190.114  -> 200  388,679 bytes  0.3s
13.35.190.120  -> TimeoutError        8.0s   ← 拒绝连接
13.35.190.23   -> 200  388,679 bytes  0.3s
```

`urllib` 每次只取解析出的**第一个**地址，于是约四分之一的请求会挂在一个永远不会应答的
节点上，超时信息与请求本身毫无关系。curl 之所以看着正常，是因为它自己会遍历地址表 ——
这正是「shell 里好好的、Python 里卡死」的来源。

修法是逐地址尝试，但**必须保留 TLS SNI 与 Host 为真实域名**：把 IP 写进 URL 再交给
`urllib` 会得到 `SSLV3_ALERT_HANDSHAKE_FAILURE`（没有证书匹配裸 IP）。所以这里没有用
改 URL 的办法，而是自己建 socket、`wrap_socket(server_hostname=host)`、再手工发请求。
另外记住最后一次成功的地址并优先使用，否则每个文件都要先等一次死节点超时。

#### 发现三：`metrics` 只有日频归档（这一点原审计已记对）

月频 `metrics` 返回 404，日频正常。所以回补 OI 历史是**每天一个文件**：12 个品种 × 一年
约 4,400 次下载，每份约 289 个 5 分钟采样。

#### P1 #12 已经实测跑通

回补前 279 万行 K 线里，三列各有 **99.9% 是 NULL**（2,780,535 / 2,784,287）—— 凡是
order-flow 采集器存在之前写入的行都没有值。而 `taker_imbalance`、`taker_imbalance_z`、
`trade_count_z`、`avg_trade_size_z`、`quote_volume_ratio` 这 5 个特征**正是**从这三列算的，
所以这 5 个名字在整段历史上都建不出来，模型只能训练 30 个契约特征里的 10 个。

写入路径先在离线用合成归档验证过（不依赖网络）：

```
区间内 NULL 行 before: 50/50
values filled: 150  (期望 150)
区间内 NULL 行 after : 0
stored: (1757487600000, 96.87285, 24059857.6812, 1234)
expected taker: 96.87285  quote: 24059857.6812   ← 完全一致
```

真实归档跑通后，这 5 个特征第一次算出了非零值：

```
filled range: 2025-09-10 07:00 .. 2025-09-30 23:55 (5,964 rows)
taker_imbalance      0.0692      taker_imbalance_z   0.2749
trade_count_z       -0.9694      avg_trade_size_z   -0.6245
quote_volume_ratio   0.4454      order_flow_bars     48
```

回补只写 NULL：语句是 `UPDATE ... WHERE ... AND <col> IS NULL`，**已经有值的行不会被选中**，
所以服务正在写的库和这个任务可以同时跑，重复执行也是幂等的。

#### P1 #13 已经实测跑通

`derivatives_detail` 从 **860 行增至 105,977 行**（16 个品种），每份日归档 288 个采样、
重复写入 0 行（`INSERT OR IGNORE`）。8 个 positioning 特征随即可以计算：

```
oi_change_5m   -0.000391   oi_change_1h        -0.003558
oi_z           -1.976943   oi_price_quadrant    0.0
long_short_ratio_z -0.8271 smart_retail_gap     0.2378
global_long_short_ratio 1.1119   taker_buy_sell_ratio_z 0.3367
```

有两列**故意留空**：`taker_buy_volume` / `taker_sell_volume` 不在归档里（来自只有 30 天的
`/futures/data/takerlongshortRatio`）。写 0 会变成「没有主动买卖」这一更强的断言，而归档
并没有这么说 —— 所以留 NULL。

#### P2 #18b：walkforward 的 fit / 验证集之间补上 purge

原先的记录说「walkforward 的验证集仍紧贴测试块」，**实测下来不准确**。测的是真实数据集
（1,258,344 行、104,874 个时间戳）：

```
fold 3: last_train=idx 17465  first_test=idx 17478  gap=13.0 bars   ← train/test 已 purge
```

真正紧贴的是**另一条边界**：`fit` 与早停用的 `valid` 之间是 **0 bar**，而标签向前伸 12 bar。
也就是说早停轮数是在「模型刚刚拟合过的那些行」上挑的，选出来的轮数部分来自记忆而非泛化。
两者都在「训练」这个筐里，所以格外容易漏掉。

修法与 train/test 完全同构：按**时间戳**（不是行下标）往回退 `label_span` 个 bar 切。
用行下标是错的 —— 一根 bar 有 12 个品种的行，按下标切会把同一根 bar 的品种劈到两边。
gap 用**最大**标签跨度而不是中位数：一条越界的标签就足以泄漏。

```
label span = 12 bars（与名义 horizon 一致）
fold 2/3/4: fit 最后一条标签结束于 valid 开始前 1.0 bar（目标 ≥13）
            fit 与 valid 的时间戳交集 = 0
            valid→test 仍是 13.0 bars
```

代价是每个 fold 少约 12 行训练数据（188,496 → 188,484），可以忽略。

新增 `label_span_bars` 进报告，让读者能看到实际用了多大的 gap。

#### 顺带修掉：重构让整个看板变成空白页

`/` 和 `/assets/*` **全部返回 404** —— 页面打不开。服务本身是好的：进程起来了、
`/health` 返回 200、`/metrics` 正常，只有静态资源全丢。

根因是一次重构的**路径算术没有跟着走**：

```python
WEB_ROOT = Path(__file__).resolve().parent.parent / 'web'   # 旧
```

文件原本在 `app/web.py`，那时 `parent.parent` 是项目根，正确。搬进 `app/web/web.py`
之后多了一层，同一个表达式就指向了 `app/web/` —— 那里什么都不发布。资源实际在项目根的
`web/`。

它**不报错**，只是安静地 404：进程正常、健康检查正常、日志正常，浏览器里一片空白，
也没有任何线索指向路径。**没有任何测试碰过根路径**，所以重构验收时它整个漏掉了。

修法不是改成 `parents[3]`（那下次再搬一层还会错），而是从包根解析：

```python
WEB_ROOT = Path(__file__).resolve().parents[2] / 'web'
```

并加了一条**方向性**的测试：`WEB_ROOT` 不得落在 `app/` 内。另外一条测试直接从
`index.html` 里解析它引用的 `/assets/...` 再逐个查存在性 —— 这样往页面里加脚本却忘了发
文件，会在测试里失败而不是在浏览器里。

浏览器实测（Playwright + Edge）：页面渲染、K 线画布 3461 个像素、符号切到 ETHUSDT 正常、
`errors []`（无 JS 报错）、移动端无横向溢出。

#### 顺带修掉：回补进度被并发进程互相覆盖

两个阶段（klines 与 metrics）本该并行跑，但第一版共用一个 `data/backfill_state.json`，
于是后写的进程把先写的进度整个覆盖掉。实测：metrics 跑了 **366 个文件全部成功**，而
`metrics_done` 是**空的** —— 每次重启都会把这 366 个文件重下一遍。

这不是崩溃，是**静默重做**：日志里全是 `OK`，看起来一切正常。所以修法有两层：
状态文件按阶段分开（`data/backfill_state/{klines,metrics}.json`），并且**原子写入**
（写 `.part` 再 `os.replace`）—— 一个写了一半的 JSON 读回来是空对象，那会把已完成的记录
全部丢掉，比崩溃更糟。已从日志里把 366 条记录抢救回 state。
### 本轮已完成（P1 #16 / P2 #17、#20）

这三项改的是「模拟是否诚实」，不依赖网络，因此先做。每一处都附了实测数字。

**P1 #16 分批止盈 + 订单/仓位对账**

- `Position` 增加 `lots: [(price, qty, fee, fill_id, at)]`，加权均价与 `entry_fee`
  改为 lot 列表的**推论**而不是第二份实现，两者不可能再对不上。
- `close(symbol, price, reason, timestamp, reason_detail, quantity=None)`：
  `quantity=None` 平掉全部（旧行为，逐字节保留）；小于持仓量则按 **FIFO** 消耗 lot，
  只实现这一部分盈亏，仓位留着。每个 lot 带着**它自己那一笔**的手续费，所以分批
  离场收的是真实份额的费用，不是混合均价摊派；资金费按同一比例切分，剩余部分留在
  未平的仓位上。实测：3 手在 110 分三次各平 1 手，`sum(pnl)` 与一次性平仓**完全相等**
  （这一条是分批止盈能用于回测的前提）。
- 恢复 `lots` 为空的旧快照时，用均价合成一个 lot，老快照不会变得无法平仓。

**P1 #16 顺带挖出两个真 bug**（都属于「检查存在但从不生效」这一类）：

1. `position_quantities()` 读 `position.quantity`，而 `Position` 的字段叫 `qty`，
   于是**每个品种都返回 0**。后果不是报表里数字不对，而是 `compare_positions` 一直
   拿成交台账去比一组零 —— 这个检查从来没有生效过。现在两种拼写都接受。
2. 更根本：被比较的两边根本不是一个东西。`signed_fills(account.trades)` 用的是
   `trades`，而 `trades` 只装**已平仓的往返**，一个品种只有在仓位消失之后才会出现，
   且净额为零。拿它去比「当前持仓」，对一个未平仓的仓位永远报告匹配。
   改为用**订单台账**（`broker.orders` 的 `filled_quantity` 按 `order_id` 聚合）
   与持仓对账，新增 `order_position_mismatches()` 与 finding
   `position_order_mismatch`。这正好覆盖审计 1.8 描述的场景：一笔 PARTIALLY_FILLED
   的建仓单随后 EXPIRED，订单侧认为自己只成交了一部分，仓位却留着，
   风控按仓位算敞口而订单侧不知道这次建仓从未完成。实测 `filled=0.4 / held=1.0`
   → `difference=0.6`，报出；对齐时为空。

**P2 #20 仓位接 edge**

审计原文：0.5bp 与 50bp 的边际会得到**完全相同**的仓位，`probability_up` 被算出来、
被传输、**被丢弃**。现在的形式是：

```
edge >= reference   -> 1.0   （满额风险预算，即改动前的行为）
edge == 0           -> 0.0   （无边际则不下单）
break-even < edge < reference -> 线性
```

为什么不是直接用 Kelly：Kelly `f* = p - (1-p)/b` 对一个 0.5% 的单笔上限是巨大的 ——
40% 胜率配 1.8 盈亏比给出 `f* = 0.067`，是上限的 13 倍。用上限归一化会让**所有**
真实边际都饱和到 1.0，等于没有缩放（第一版就是这么写的，实测每一档都是 1.000）。
现在用**参考边际**归一化，Kelly 仍然计算并记录，因为它回答另一个问题：这笔交易
值不值得做 —— 跌破盈亏平衡点时 Kelly ≤ 0，缩放直接归零。

实测（入场 100 / 止损 98 / 目标 104，即 2:1，盈亏平衡胜率 1/3）：

| 输入 | 数量 | 缩放 | 隐含概率 | Kelly |
|---|---|---|---|---|
| 0bp | 0 | 0.00 | 0.3333 | 0.0 |
| 0.5bp | 0.5 | 0.02 | 0.3342 | 0.00125 |
| 12.5bp | 12.5 | 0.50 | 0.3542 | 0.03125 |
| 25bp | 25 | 1.00 | 0.3750 | 0.0625 |
| 50bp | 25 | 1.00 | 0.4167 | 0.125 |

三条硬性质，都有测试钉住：

- **只能缩小仓位**：`max_risk` 仍是天花板，任何边际下 `quantity <= 基线`。
  所有既有上限、限额、测试的含义不变。
- **两种说法必须一致**：`edge_bps=0` 与 `probability_up=盈亏平衡点` 描述同一笔交易，
  必须给出同一个缩放。第一版的换算写错了，前者给 0.0、后者给 0.84 —— 现在两者都是
  0.0，且隐含概率相同。
- **`SIZING_REFERENCE_EDGE_BPS=0` 可整体关闭**，逐字节回到改动前。

一处实现细节值得记下：`approve()` 原先只认关键字参数，而实盘的 candidate 字典里
**本来就带着** `edge_bps` / `probability_up`（`plan()` 是 `{**signal, ...}`）。
要求调用方再拆一次参数，就意味着漏一次就静默退回固定仓位 —— 正是要修的那个 bug。
现在 `size()` 在关键字为空时回落到读 plan，实测端到端 0.5bp→0.02、25bp→1.00。

还有一个只有跑起来才会发现的浮点问题：恰好处于盈亏平衡点时，Kelly 算出的是
`5e-17` 而不是 `0`（`break_even = loss/span` 两次除法不落在同一个浮点数上），
于是「无边际」会提交一个 `2e-14` 手的订单，而 `binding` 显示 `risk` 而不是
`risk_budget_exhausted`。现在小于 `1e-9` 直接归零。

**P2 #17 延迟模型**

`FillModel` 原来只有两项：`spread_bps`（穿价差）与 `impact_coefficient_bps`
（市场冲击）。两项描述的都是**成交那一刻**的盘口，没有一项描述订单在途期间
发生的移动 —— 而那个移动是**系统性不利**的：订单之所以成交，是因为变得可成交，
也就是价格朝它走过来了。

新增 `entry_latency_ms`（默认 **0**，即完全不改变任何既有结果）与
`latency_adverse_share`（默认 1.0，保守读法），按扩散缩放：

```
latency_bps = sigma_per_bar_bps * sqrt(latency_ms / bar_ms) * adverse_share
```

实测（`spread_bps=2`，买入）：

| 场景 | 滑点 bps | 成交价 |
|---|---|---|
| 无延迟 | 2.000 | 50010.00 |
| 500ms，50bp/bar | 4.041 | 50020.21 |
| 2000ms，50bp/bar | 6.082 | 50030.41 |
| 2000ms，200bp/bar | 18.330 | 50091.65 |
| 2000ms，200bp/bar，adverse 0.5 | 10.165 | 50050.82 |

延迟 4 倍 → 成本 2 倍（扩散，非线性漂移）；挂单成交不付延迟；
无波动率测量则不计费（读作「没有估计」，不是「波动率为零」）。
默认值改由 `ENTRY_LATENCY_MS`（现为 350ms，零售 HTTP 客户端的诚实取值）驱动，
设为 0 即回到旧行为。

**顺带修的第三处**：`execute_depth()`（走盘口阶梯的成交路径）原来**完全不套用
成本模型** —— 直接返回阶梯均价。于是同一个订单走「深度路径」比走隔壁「报价路径」
更便宜。现在它也过 `FillModel`，但**只收延迟**：阶梯本身已经体现了价差与冲击，
再加一遍就是重复计费（第一版正是如此，把一条有测试钉住的成交价从 101.33 推到了
101.35）。`slip_bps` 因此新增 `include_spread` / `include_impact` 两个开关。

另外，波动率必须由 **strategy 层**算好传进来（`trading` 不能反向 import
`strategy`）—— 第一版把 `_realized_volatility` 直接 import 进了 `execution.py`，
被 `tests/test_architecture.py` 当场拦下。现在由 `decision_loop` 与
`feeds_handlers` 传入 `_bar_volatility_bps()`。

### 本轮已完成（P2 #18、#19）

这一轮改的是「训练出来的东西是否可信」。三项都依赖数据但不依赖网络，所以能做。

**P2 #18a 三重障碍标签**

旧标签是「下一根开盘 → 12 根之后收盘」的固定远期收益。审计的原话是：**它描述的是
一个没人跑的策略** —— 实盘有止损、止盈、移动止损和时间止损，碰到哪个就按哪个离场。
拿固定远期收益训出来的模型，回答的是执行层从没问过的问题。

现在按 Lopez de Prado 的三重障碍：从每根 bar 设上（止盈）、下（止损）、垂直（时间）
三道障碍，标签是**先被碰到的那一道**及其实际收益。障碍距离直接来自
`exit_policy.plan_levels()` —— **实盘调用的同一个函数**，不是一份常量的拷贝。

一处不是简化的设计决定：**上下障碍对称**（都用风险单位）。标签是在模型还没给出方向
之前就造好的，如果按策略的 1.8R 目标放上面、1R 止损放下面，上障碍就比下障碍远 1.8 倍，
标签会**因为与市场无关的原因而系统性偏多**，模型学会的正是这个偏差。

实测（BTCUSDT 5m，2787 行）：

| 障碍 | 行数 | 占比 |
|---|---|---|
| vertical（时间止损） | 1719 | 61.7% |
| lower（止损） | 586 | 21.0% |
| upper（止盈） | 482 | 17.3% |

与旧标签的相关系数 0.730 —— 有关但不等同，正是「不是同一个分布」的量化版本。

**P2 #18b 唯一性权重**

相邻样本共享 12/13 的收益窗口，被当成独立样本。实测 BTC 5m 的标签 lag-1 自相关：
固定远期 **0.9162**，三重障碍 **0.8207**。权重归一化到均值 1（否则加权拟合会连学习率
一起悄悄改掉），有效样本数 **2168 / 2787（77.8%）** —— 也就是朴素 t 值大约虚高
√(1/0.778) ≈ 13%。

接进 `train_tabular` 时有一个细节：`label_weights()` 在没有区间信息时返回 **None**
而不是全 1 的数组。显式传均匀权重与不传权重在 LightGBM 里**不是同一条代码路径**，
会让「标签改造前建的数据集」与之后的跑出不同结果 —— 传 None 则与改动前逐位一致。

**顺带挖出一个性能 bug（会直接把真实数据集跑死）**

`sample_weights` 第一版把记录的毫秒时间戳直接喂给并发扫描，于是它**按毫秒分配数组**：
587 行跨越 1.76 亿毫秒，耗时 **134 秒**、占用近 1GB；真实数据集有 120 万行，根本跑不完。
现在先转成 **bar 序号**（时间戳在行集里的稠密排名）+ `bars_held`。同一个输入：**0.005 秒**。

这个 bug 阴险的地方在于**权重算出来是对的**，只是慢到不可用 —— 所以专门加了计时回归测试。

**P2 #19 Deflated Sharpe / PBO**

审计原文：`horizon` 与 `MIN_EDGE_MULTIPLE` 被调过，但**没有记录试验次数**，
这让 Sharpe 这个数字**无法解释**（而不是仅仅偏乐观）。新增 `app/models/validation.py`：

- **DSR**：在「这是 N 次试验里最好的一个」的前提下，Sharpe 真实大于 0 的概率。
  零技巧下 N 次试验的期望最大值 `E[max] ≈ sqrt(2 ln N)`：1000 次试验光靠噪声就能
  得到可观的 Sharpe。
- **PBO**：组合切分中，样本内最优配置在样本外落到中位数以下的比例。选择毫无信息时
  趋于 0.5。

默认基准的选择是关键，第一版写错了：`variance_of_trials` 默认取 1.0 时，
基准对任何超过百次的搜索都是 ~2.5 个 Sharpe 单位，**所有策略的概率都归零**，
指标变得无用而不是保守。改为默认取零技巧离散度 `1/T` 后，z 值化简为
`sharpe*sqrt(T-1) - E[max]`，即**夏普的 t 统计量减去期望最大值**，物理意义清楚，
且不需要调用方提供任何拿不到的量。

实测（年化 Sharpe 1.5，换算到 5m bar 为 0.00463/bar）：

| bar 数 | 1 次试验 | 100 次试验 |
|---|---|---|
| 5,000 | 0.628 | 0.014 |
| 20,000 | 0.744 | 0.030 |
| 100,000 | 0.928 | 0.143 |
| 400,000 | 0.998 | 0.654 |

读法：**一个年化 1.5 的策略，要在 100 次试验的搜索后仍然站得住，需要约 40 万根 5m bar**
（约两年）。这解释了为什么小样本上的调参结果不可复现。

在真实数据集上验证了它**能区分噪声与信号**：纯噪声预测器（60,986 个成交样本）DSR = 0.000；
接近神谕的预测器 DSR = 1.000。

裁决是**建议性**的：`survives` 会写进 manifest，但不会阻止训练 —— 一个模型可能因为
Sharpe 捕捉不到的理由值得上线，静默拒绝训练比给一条警告更糟。
证据不足时返回 `None` 而不是一个看起来通过的数字，因为**静默给默认值比没有指标更糟**：
操作者就再也不会去看它了。

**一处诚实的残留**：PurgedKFold 与 embargo 仍未做。`dataset_split` 有 label purge
（manifest 报 288 行，与 12 品种 × 12 bar × 2 边界一致），但 `walkforward` 的
early-stopping 验证集仍然紧贴测试块，没有间隔。

### P3 —— 之后

21. 限价单队列模型（若要做挂单策略）
22. 横截面因子（Qlib Alpha158 的 crypto 版）
23. 组合层（波动率目标 / 相关性上限 / 风险平价）
24. freqtrade protections 移植（分品种熔断）
25. 跨所资金费/基差（Bybit / OKX / Hyperliquid 免费 REST）：携带因子，
    参考文献 Schmeling, Schrimpf & Todorov (2023) *Crypto carry*, BIS WP 1087
26. Deribit DVOL / 25Δ skew（免费无 key）：变异风险溢价，参考文献
    Alexander & Imeraj (2021) *The Bitcoin VIX and Its Variance Risk Premium*, JAI

---

## 附：复现脚本

审计过程中生成的脚本已全部删除（验证完即弃）。需要复现时，文中每一条实测结论都给了
\`file:line\` 与可直接执行的命令；分层重构后模块路径有变动，正文与 README 中的路径
均已同步更新。

分层契约本身是可执行的，见 \`tests/test_architecture.py\`；本轮修复的回归测试见
\`tests/test_cost_contract.py\`、\`tests/test_equity_and_marks.py\`、
\`tests/test_exit_source.py\`、\`tests/test_metrics_endpoint.py\`。
