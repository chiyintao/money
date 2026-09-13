// Model center and system health views.

import {esc, num} from "./format.js";
import {emptyRow} from "./format.js";
import {set, html} from "./dom.js";

const value = (v, d = 2) => v === null || v === undefined ? "—" : num(v, d);
const MODE_LABELS = {production: "生产模型", candidate: "真实候选权重", unavailable: "模型不可用", loading: "加载中"};

export function operationsShell() {
  return `
<section id="models" class="view">
  <div class="ops-summary">
    <div><span>决策模式</span><strong id="decisionMode">真实模型</strong></div>
    <div><span>生产模型</span><strong id="productionModel">未启用</strong></div>
    <div><span>预测队列</span><strong id="modelQueue">—</strong></div>
    <div><span>模型接口</span><strong id="modelConnection">等待数据</strong></div>
  </div>
  <div class="panel-head"><h2>真实模型决策信号</h2><button id="refreshModels">刷新模型</button></div>
  <div id="modelMessage" role="status"></div>
  <div class="table-wrap"><table><thead><tr>
    <th>合约</th><th>方向</th><th>模型收益 (bp)</th><th>一致性</th><th>投票</th><th>来源</th><th>K 线时间</th>
  </tr></thead><tbody id="decisionRows"></tbody></table></div>
  <div class="panel-head"><h2>已加载权重</h2></div>
  <div class="table-wrap"><table><thead><tr>
    <th>模型</th><th>版本</th><th>状态</th><th>训练成本阈值</th><th>测试准确率</th><th>活跃样本</th><th>校验</th>
  </tr></thead><tbody id="weightRows"></tbody></table></div>
  <div class="panel-head"><h2>影子预测</h2></div>
  <div class="table-wrap"><table><thead><tr>
    <th>合约 / 模型</th><th>状态</th><th>信号</th><th>预测收益</th><th>成本后预期</th><th>数据时间</th><th>推理耗时</th>
  </tr></thead><tbody id="modelRows"></tbody></table></div>
</section>
<section id="system" class="view">
  <div class="panel-head"><h2>运行健康</h2><span>PAPER ONLY</span></div>
  <div id="healthGrid" class="health-grid"></div>
  <div class="panel-head"><h2>行情新鲜度</h2></div>
  <div class="table-wrap"><table><thead><tr>
    <th>合约</th><th>标记价延迟</th><th>开仓数据状态</th>
  </tr></thead><tbody id="freshnessRows"></tbody></table></div>
</section>`;
}

function decisionRow(signal) {
  const votes = Object.entries(signal.votes || {})
    .map(([name, v]) => esc(name) + " " + value(v * 10000, 3)).join(" · ");
  const when = signal.bar_time ? new Date(signal.bar_time).toLocaleString() : "—";
  return "<tr>"
    + "<td><b>" + esc(signal.symbol) + "</b><small>" + esc(String(signal.model_mode || "")) + "</small></td>"
    + "<td>" + esc(signal.side || "—") + "</td>"
    + "<td>" + value(signal.edge_bps) + "</td>"
    + "<td>" + value(signal.agreement, 2) + "</td>"
    + '<td class="muted">' + (votes || "—") + "</td>"
    + "<td>" + esc(signal.source || "—") + "<small>" + esc((signal.reason_codes || []).join(" · ")) + "</small></td>"
    + "<td>" + esc(when) + "</td>"
    + "</tr>";
}

function weightRow(member) {
  const healthy = member.status === "production";
  return "<tr><td><b>" + esc(member.name) + "</b></td>"
    + "<td>" + esc(member.version) + "</td>"
    + '<td><span class="status-badge ' + (healthy ? "good" : "bad") + '">' + esc(member.status) + "</span></td>"
    + "<td>" + value(member.cost_bps, 1) + " bp</td>"
    + "<td>" + value(member.test?.directional_accuracy_pct) + "%</td>"
    + "<td>" + value(member.test?.active_samples, 0) + "</td>"
    + '<td class="muted">' + esc(String(member.sha256 || "").slice(0, 10)) + "</td></tr>";
}

function chronosRow(chronos) {
  const state = chronos.error ? "异常" : chronos.loaded ? "已加载" : "待加载";
  return "<tr><td><b>chronos-2</b></td>"
    + "<td>" + esc(chronos.device) + "</td>"
    + '<td><span class="status-badge ' + (chronos.error ? "bad" : "good") + '">' + state + "</span></td>"
    + "<td>—</td><td>—</td>"
    + "<td>" + value(chronos.forecasts, 0) + " 次预测</td>"
    + '<td class="muted">h' + value(chronos.horizon, 0) + "</td></tr>";
}

function shadowRows(result) {
  const rows = [];
  const stale = Date.now() - Number(result.bar_time) > 120000;
  for (const [name, prediction] of Object.entries(result.predictions || {})) {
    const raw = prediction.expected_return ?? prediction.median_return;
    rows.push("<tr>"
      + "<td><b>" + esc(result.symbol) + "</b><small>" + esc(name) + "</small></td>"
      + '<td><span class="status-badge ' + (prediction.error || stale ? "bad" : "good") + '">'
      + (prediction.error ? "异常" : stale ? "数据过期" : "研究") + "</span></td>"
      + "<td>" + esc(prediction.side || "分位预测") + "</td>"
      + "<td>" + value(raw == null ? null : raw * 100, 4) + "%</td>"
      + "<td>" + value(prediction.expected_net_return == null ? null : prediction.expected_net_return * 100, 4) + "%</td>"
      + "<td>" + esc(result.bar_time ? new Date(result.bar_time).toLocaleString() : "—") + "</td>"
      + "<td>" + value(result.latency_ms) + " ms</td></tr>");
  }
  for (const [name, message] of Object.entries(result.load_errors || {})) {
    rows.push("<tr><td>" + esc(result.symbol) + " / " + esc(name) + '</td><td colspan="6" class="down">' + esc(message) + "</td></tr>");
  }
  return rows;
}

export function renderModels(data, error = "") {
  const decision = data.decision || {};
  const featureSpace = decision.feature_space || {};
  set("modelConnection", error ? "连接失败" : (decision.members || []).length + " 个真实权重");
  set("decisionMode", error ? "未知" : (decision.label || MODE_LABELS[decision.mode] || decision.mode || "等待后端数据"));
  set("productionModel", data.production?.version || (decision.mode === "candidate" ? "候选权重（未晋级）" : "未启用"));
  set("modelQueue", value(data.queued, 0));
  set("modelMessage", error || [
    decision.rule_fallback ? "已开启规则回退：真实模型不可用时退回 EMA/RSI 基线" : "规则回退关闭：真实模型不可用时不下单",
    "越界输入策略：" + (decision.ood_policy === "block" ? "拦截" : "仅告警"),
    "训练特征域：" + (featureSpace.verified ? "已按数据集校验" : "未校验"),
  ].join(" · "));

  const weightRows = (decision.members || []).map(weightRow);
  if (decision.chronos?.enabled) weightRows.push(chronosRow(decision.chronos));
  for (const [name, message] of Object.entries(decision.load_errors || {})) {
    weightRows.push("<tr><td>" + esc(name) + '</td><td colspan="6" class="down">' + esc(message) + "</td></tr>");
  }
  const shadow = (data.models || []).flatMap(shadowRows);

  html("decisionRows", (decision.recent || []).map(decisionRow).join("") || emptyRow(7, "真实模型尚未产生决策信号"));
  html("weightRows", weightRows.join("") || emptyRow(7, "未加载到真实模型权重"));
  html("modelRows", shadow.join("") || emptyRow(7, "暂无影子预测记录"));
}

export function renderHealth(state) {
  const connector = state.connector_health || {};
  const persistence = state.persistence || {};
  const cells = [
    ["行情连接", connector.connected === undefined ? "未知" : connector.connected ? "已连接" : "未连接"],
    ["行情事件", value(state.ws_events, 0)],
    ["重连次数", value(connector.reconnects ?? state.ws_reconnects, 0)],
    ["风控", state.risk_state?.halted ? "已熔断" : "未熔断"],
    ["账户审计", state.account_audit ? JSON.stringify(state.account_audit) : "未知"],
    ["持久化", Object.keys(persistence).length ? JSON.stringify(persistence) : "暂无状态"],
  ];
  html("healthGrid", cells.map(([label, text]) =>
    "<div><span>" + esc(label) + "</span><strong>" + esc(text) + "</strong></div>").join(""));

  const fresh = Object.entries(state.mark_event_age_ms || {}).map(([symbol, age]) => {
    const recent = age >= 0 && age <= 5000;
    return "<tr><td>" + esc(symbol) + "</td><td>" + value(age, 0) + ' ms</td><td class="'
      + (recent ? "up" : "down") + '">' + (recent ? "标记价近期" : "标记价过期") + "</td></tr>";
  });
  html("freshnessRows", fresh.join("") || emptyRow(3, "暂无标记价事件"));
}
