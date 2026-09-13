// Model training: universe selection, progress and out-of-sample results.

import {esc, emptyRow, num, dt} from "../format.js";
import {set, html} from "../dom.js";

const STAGE_LABELS = {
  universe: "识别币种池", history: "抓取历史 K 线", dataset: "构建训练集",
  walk_forward: "滚动验证", train: "训练模型", done: "完成",
};

const TIER_LABELS = {mainstream: "热门正规币", speculative: "快钱山寨币",
                     excluded: "已排除"};

export function trainingShell() {
  return `
<section id="training" class="view">
  <div class="panel">
    <div class="panel-head">
      <div><h2>模型训练</h2><p>抓取历史 → 构建训练集 → 滚动验证 → 训练候选模型</p></div>
      <button id="startTraining">开始训练</button>
    </div>
    <div class="train-controls">
      <label>币种池<select id="trainTier">
        <option value="mainstream">热门正规币</option>
        <option value="speculative">快钱山寨币</option>
      </select></label>
      <label>K 线周期<select id="trainInterval">
        <option value="5m">5m</option><option value="15m">15m</option>
        <option value="1h">1h</option><option value="4h">4h</option>
      </select></label>
      <label>历史天数<input id="trainDays" type="number" min="30" max="1500" value="365"></label>
      <label>验证折数<input id="trainFolds" type="number" min="4" max="30" value="12"></label>
      <label>成本 (bp)<input id="trainCost" type="number" min="0" max="100" value="12"></label>
      <label>币数上限<input id="trainMaxSymbols" type="number" min="1" max="40" value="15"></label>
      <label>前瞻根数<input id="trainHorizon" type="number" min="1" max="96" value="12"></label>
    </div>
    <div id="trainEstimate" class="train-estimate"></div>
    <div id="trainStatus" class="train-status" role="status">尚未运行训练</div>
    <div id="trainVerdict" class="train-verdict"></div>
  </div>
  <div class="panel">
    <div class="panel-head"><h2>币种池分类</h2><span id="universeCounts"></span></div>
    <div class="table-wrap"><table><thead><tr>
      <th>币种</th><th>分层</th><th>24h 成交额</th><th>日振幅</th><th>上市天数</th><th>判定依据</th>
    </tr></thead><tbody id="universeRows"></tbody></table></div>
  </div>
  <div class="panel">
    <div class="panel-head"><h2>滚动验证</h2><span>每折都是独立的样本外区间</span></div>
    <div class="table-wrap"><table><thead><tr>
      <th>折</th><th>测试起点</th><th>样本数</th><th>实际交易</th><th>方向准确率</th><th>成本后边际</th>
    </tr></thead><tbody id="walkForwardRows"></tbody></table></div>
  </div>
  <div class="panel">
    <div class="panel-head"><h2>训练记录</h2><span>每次运行的样本外成绩</span></div>
    <div class="table-wrap"><table><thead><tr>
      <th>时间</th><th>币种池</th><th>币数</th><th>交易数</th><th>边际</th><th>P(边际&gt;0)</th><th>结论</th>
    </tr></thead><tbody id="trainHistoryRows"></tbody></table></div>
  </div>
  <div class="panel">
    <div class="panel-head"><h2>运行日志</h2></div>
    <pre id="trainLogs" class="train-logs"></pre>
  </div>
</section>`;
}

const BARS_PER_DAY = {"5m": 288, "15m": 96, "1h": 24, "4h": 6};
const PAGE_SIZE = 1500;
const SECONDS_PER_PAGE = 4.8;   // measured through the proxy, six symbols in flight
const CONCURRENCY = 6;

// A year of 5m bars is 71 requests per symbol, about a quarter of an hour for fifteen
// symbols. Saying so up front is the difference between "working" and "stuck".
export function estimateDownload(interval, days, symbols) {
  const bars = (BARS_PER_DAY[interval] || 288) * Number(days || 0);
  const pages = Math.max(1, Math.ceil(bars / PAGE_SIZE));
  const waves = Math.max(1, Math.ceil(Number(symbols || 1) / CONCURRENCY));
  const seconds = Math.round(waves * pages * SECONDS_PER_PAGE);
  return seconds >= 120 ? "约 " + Math.round(seconds / 60) + " 分钟" : "约 " + seconds + " 秒";
}

export function renderEstimate() {
  const read = (id, fallback) => {
    const node = document.getElementById(id);
    const value = node ? Number(node.value) : NaN;
    return Number.isFinite(value) ? value : fallback;
  };
  const intervalNode = document.getElementById("trainInterval");
  const interval = intervalNode ? intervalNode.value : "5m";
  const days = read("trainDays", 365);
  const symbols = read("trainMaxSymbols", 15);
  set("trainEstimate", "首次抓取预估 " + estimateDownload(interval, days, symbols)
      + "（已有历史自动跳过；" + interval + " × " + days + " 天 × 最多 " + symbols + " 个币）");
}

function money(value) {
  if (value == null) return "—";
  const millions = Number(value) / 1e6;
  return millions >= 1000 ? (millions / 1000).toFixed(2) + "B" : millions.toFixed(1) + "M";
}

export function renderUniverse(data) {
  const counts = data.counts || {};
  set("universeCounts", ["热门正规币 " + (counts.mainstream || 0),
                         "快钱山寨币 " + (counts.speculative || 0),
                         "已排除 " + (counts.excluded || 0)].join(" · "));
  const shown = (data.symbols || []).filter(item => item.tier !== "excluded").slice(0, 60);
  html("universeRows", shown.map(item => "<tr>"
    + "<td><b>" + esc(item.symbol) + "</b></td>"
    + '<td><span class="status-badge ' + (item.tier === "mainstream" ? "good" : "warn")
    + '">' + esc(TIER_LABELS[item.tier] || item.tier) + "</span></td>"
    + "<td>" + money(item.quote_volume) + "</td>"
    + "<td>" + (item.daily_range_pct == null ? "—" : num(item.daily_range_pct, 1) + "%") + "</td>"
    + "<td>" + num(item.age_days, 0) + "</td>"
    + '<td class="muted">' + esc((item.reasons || []).join("; ")) + "</td>"
    + "</tr>").join("") || emptyRow(6, "暂无行情数据"));
}

export function renderTraining(state) {
  // Fall back to the most recent finished run so the per-fold detail and the verdict are
  // still on screen after a restart, when nothing is in flight.
  const job = (state && state.current)
    || (state && state.history && state.history.length ? state.history[0] : null);
  const running = !!(state && state.running);
  const button = document.getElementById("startTraining");
  if (button) {
    button.disabled = running;
    button.textContent = running ? "训练中…" : "开始训练";
  }

  if (!job) {
    set("trainStatus", "尚未运行训练");
    html("trainVerdict", "");
    html("walkForwardRows", emptyRow(6, "尚无验证结果"));
    set("trainLogs", "");
  } else {
    const stages = ["universe", "history", "dataset", "walk_forward", "train"];
    const at = stages.indexOf(job.stage);
    const label = STAGE_LABELS[job.stage] || job.stage;
    set("trainStatus", [
      "运行 " + job.run_id,
      "币种池 " + (TIER_LABELS[job.tier] || job.tier),
      "阶段 " + (at >= 0 ? (at + 1) + "/" + stages.length + " " + label : label),
      job.status === "running" ? "进行中" : job.status === "failed" ? "失败" : "已完成",
    ].join(" · "));

    html("trainVerdict", (job.verdict || []).map(line =>
      '<div class="train-verdict-line">' + esc(line) + "</div>").join(""));

    const folds = (job.walk_forward && job.walk_forward.per_fold) || [];
    html("walkForwardRows", folds.map(fold => "<tr>"
      + "<td>" + (fold.fold + 1) + "</td>"
      + "<td>" + esc(dt(fold.test_start)) + "</td>"
      + "<td>" + num(fold.test_rows, 0) + "</td>"
      + "<td>" + num(fold.active_samples, 0) + "</td>"
      + "<td>" + num(fold.directional_accuracy_pct, 2) + "%</td>"
      + '<td class="' + (fold.active_net_edge_bps > 0 ? "up" : "down") + '">'
      + num(fold.active_net_edge_bps, 2) + " bp</td></tr>").join("")
      || emptyRow(6, "尚无验证结果"));

    set("trainLogs", (job.logs || []).join("\n"));
  }
  renderHistory(state && state.history);
}

function renderHistory(history) {
  const rows = (history || []).map(run => {
    const aggregate = (run.walk_forward && run.walk_forward.aggregate) || {};
    const boot = (run.walk_forward && run.walk_forward.bootstrap) || {};
    const edge = aggregate.net_edge_bps;
    return "<tr>"
      + "<td>" + esc(dt(run.finished_at || run.started_at)) + "</td>"
      + "<td>" + esc(TIER_LABELS[run.tier] || run.tier) + "</td>"
      + "<td>" + num((run.symbols || []).length, 0) + "</td>"
      + "<td>" + num(aggregate.active_samples, 0) + "</td>"
      + '<td class="' + (edge > 0 ? "up" : "down") + '">'
      + (edge == null ? "—" : num(edge, 2) + " bp") + "</td>"
      + "<td>" + (boot.prob_positive == null ? "—" : num(boot.prob_positive, 2)) + "</td>"
      + '<td class="muted">' + esc((run.verdict || [])[0] || run.status) + "</td>"
      + "</tr>";
  });
  html("trainHistoryRows", rows.join("") || emptyRow(7, "暂无训练记录"));
}
