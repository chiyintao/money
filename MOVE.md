# 搬运指南（MOVE.md）

把整个文件夹复制到另一台机器 / 另一个路径，让它照常跑起来。

好消息是**代码本身已经是路径无关的**：没有任何硬编码绝对路径，
`data_dir` 默认是相对路径 `data`，`start_paper.bat` 用 `%~dp0` 定位自己。
所以搬运的问题不在代码，而在**哪些数据该带、哪些设置是本机特有的**。

---

## 一、先决定带什么

运行自检（**只报告，不改动**）：

```bash
python -m scripts.clean_data
```

输出把每一项分成三类：

| 类别 | 大小 | 说明 |
|---|---|---|
| **必须搬** | ~46 MB | `app/` `tests/` `scripts/` `web/` `docs/` 配置与启动脚本、`data/models/`、线上候选 |
| **建议搬** | ~1.3 GB | `data/research.sqlite3` —— K 线/资金费/持仓量历史，重建要**数小时** |
| **可再生** | ~2.5 GB | 训练数据集、Chronos 权重、归档 zip 缓存、日志 |

### 两种搬法

**A. 全量搬（3.8 GB）** —— 复制整个文件夹，什么都是现成的。

**B. 精简搬（~1.4 GB）** —— 先删掉可再生的部分：

```bash
python -m scripts.clean_data --drop-safe   # 删除"可再生"那一类
```

到新机器上补回来：

```bash
python scripts/download_models.py          # Chronos 权重（456 MB）
```

数据集不用手动补：**第一次训练时自动重建**，而且现在并行构建，
15 个品种约 2 分钟（优化前是 16.5 分钟）。

> 如果连 `data/research.sqlite3` 也不带（比如新机器要重新回补），
> 那首次运行需要先跑回补脚本把 K 线拉下来，这一步是**小时级**的。

---

## 二、必须先做的两件事

### 1. 关掉本机的代理设置（最容易踩的坑）

`.env` 末尾有三行：

```
BINANCE_WS_PROXY=http://127.0.0.1:7890
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890
```

这是**开发机特有的**。原机器上 `fstream.binance.com`（行情 WebSocket）直连会 25 秒超时，
必须走本地代理；而新机器上大概率没有这个代理，
**留着这三行会让所有网络请求失败**，表现为行情流为 0、下单被拒为 `missing_book_time`。

搬到新机器后：

- 新机器能直连 Binance → **把这三行注释掉或删掉**
- 新机器也需要代理 → 改成你自己的代理地址

`.env.example` 里这三行默认是**注释掉的**，就是为这个原因。

### 2. 重建 Python 环境

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-models.txt
```

`start_paper.bat` 会在缺依赖时自动装 `requirements.txt`，但模型依赖要手动装。

---

## 三、搬完之后验证

```bash
# 1. 代码完整（应该 906 通过）
python -m pytest tests -q

# 2. 数据与模型都在
python -m scripts.clean_data

# 3. 网络通不通（先确认这一步，再启动服务）
python -c "import urllib.request; print(urllib.request.urlopen('https://fapi.binance.com/fapi/v1/time', timeout=15).status)"

# 4. 启动
start_paper.bat
```

启动后打开 <http://127.0.0.1:8101>，看 `/health`：

```json
{"name": "market_data", "status": "ok", "detail": "live"}      ← 必须是 ok
{"name": "feed",        "status": "ok", "detail": "streaming"} ← 必须是 ok
```

如果这两项是 `down` / `no_live_events`，看下一步。

---

## 四、排障对照表

| 现象 | 原因 | 处理 |
|---|---|---|
| `/health` 里 `market_data: down`，`connector_health.connected=false` | 行情 WebSocket 连不上 | 确认网络；需要代理则设 `BINANCE_WS_PROXY` |
| 订单全被拒为 `missing_book_time` | 同上：没有订单簿就没有成交价 | 同上 |
| 所有 REST 请求 418 / 超时 | 代理设错，或 Binance 封了 IP | 检查代理；被封则**停掉服务等封禁到期**（继续请求会刷新计时） |
| 选币列表为空、`trained` 长度为 0 | 特征空间没验证通过 | 看 `/api/models` 的 `feature_space.verified` 是否为 true |
| `training_dataset_missing` | 数据集被删了 | 跑一次训练，会自动重建 |
| 端口 8101 被占 | 旧进程还在 | `start_paper.bat` 会自动杀掉占用者 |

---

## 五、这份配置里有哪些是"为了让演示跑起来"而放宽的

搬运不会改变这些，但换机器后如果发现它**频繁开仓且持续亏损**，原因在这里。

为了让模拟真正运转，以下参数被有意放宽（详见 `docs/simulation-and-decision-audit.md` §0.5.19）：

| 参数 | 严格值 | 当前值 |
|---|---|---|
| `MIN_EDGE_MULTIPLE` | 2.5 | 1.0 |
| `MODEL_MIN_EDGE_BPS` | 0.5 | −1.5 |
| `MODEL_MIN_AGREEMENT` | 0.6 | 0.3 |
| `SYMBOL_EDGE_ENABLED` | 1 | 0 |
| `MODEL_OOD_POLICY` | block | warn |
| `FEATURE_DEGRADED_POLICY` | block | warn |

其中只有 `ENTRY_ORDER_TYPE=limit`（成本 12bp→4bp）是**技术上正确**的：
挂单不吃价差、只付 maker 费率，成本模型本就该随执行方式定价。

**其余都是明知故犯地放宽。** 实测模型没有 edge（走查 −10.33bp，方向准确率 49.2%），
恢复严格值就回到"只在有正期望时才开仓"——代价是它基本不开仓。

---

## 六、目录结构速查

```
money/
├─ app/                    服务本体（分层：core→storage→market→features→trading→backtest→models→strategy→ops→web）
├─ tests/                  906 个测试
├─ scripts/                运维脚本（回补、下载模型、清理数据、前端检查）
├─ web/                    前端资源
├─ docs/                   审计与设计文档
├─ data/
│  ├─ research.sqlite3     K 线/资金费/持仓量/事件 —— 核心数据，建议搬
│  ├─ models/              晋级模型
│  ├─ research_v3/
│  │  ├─ candidates/       线上候选（服务实际加载这里）
│  │  ├─ candidates_*/     分层训练产物
│  │  └─ training_dataset_*.jsonl   可再生
│  ├─ pretrained/          Chronos 权重，可再生
│  └─ archive_cache/       Binance 归档缓存，可再生
├─ .env                    本机配置（含代理）
├─ .env.example            可移植模板
├─ MOVE.md                 本文档
└─ start_paper.bat         启动
```
