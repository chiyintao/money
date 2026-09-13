// Top-level render orchestration.
//
// This module used to hold a single 3,944-character render() that computed every
// figure and patched every element inline. Views now own their own rendering; this
// file only decides what to update and in what order.

import {renderTerminal} from "./terminal.js";
import {drawSeries} from "./charts.js";
import {renderHealth} from "./operations.js";
import {html, set} from "./dom.js";
import {esc, dt, duration, money, num, pct, tone, sideBadge} from "./format.js";
import {positionRow} from "./views/positions.js";
import {renderTradeRows} from "./views/trades.js";
import {renderOrderRows} from "./views/orders.js";
import {renderMarketRows} from "./views/markets.js";
import {renderAuditRows} from "./views/audit.js";
import {renderAnalytics, renderBalances, renderRisk} from "./views/balance.js";
import {renderSession, renderRiskProfile, renderExitPolicy} from "./views/session.js";

const emptyRow = (cols, text) => '<tr><td colspan="' + cols + '" class="empty"><b>暂无数据</b><span>' + text + '</span></td></tr>';

export function render(state) {
  // The risk panel is drawn from the separately fetched catalogue, which is the only
  // place that knows the presets and which one is actually in force.
  if (window.__riskCatalogue) renderRiskProfile(window.__riskCatalogue);
  if (window.__exitCatalogue) renderExitPolicy(window.__exitCatalogue);
  renderTerminal(state);
  renderHealth(state);
  window.__paperState = state;

  const metrics = state.metrics || {};
  const positions = state.positions_detail || [];

  renderBalances(state);
  renderRisk(state);
  renderAnalytics(state);
  renderSession(state);

  set("lastUpdate", state.updated ? new Date(state.updated).toLocaleTimeString("zh-CN", {hour12: false}) : "--");
  set("freshText", Object.keys(state.mark_event_age_ms || {}).length + " 个实时行情源");
  set("positionCount", positions.length);
  set("positionSummary", positions.length + " 个持仓 · " + money(state.total_notional) + " 名义价值");

  const overview = positions.length ? positions.map(x => positionRow(x)).join("") : emptyRow(11, "模拟策略开仓后会显示实时风险");
  html("overviewPositions", overview);

  const detailed = positions.length ? positions.map(x => positionRow(x, true)).join("") : emptyRow(12, "当前没有合约持仓");
  html("positionRows", detailed);

  const trades = renderTradeRows(state, state.all_time_trades);
  set("tradeCount", trades.count);
  html("tradeRows", trades.body);
  set("ledgerSummary", trades.count + " 笔本会话成交 · 净盈亏 " + money(state.realized_pnl, 4));

  const orders = renderOrderRows(state);
  set("openOrderCount", orders.openCount);
  html("openOrderRows", orders.openBody);
  html("orderHistoryRows", orders.historyBody);

  const markets = renderMarketRows(state);
  html("marketRows", markets.body);
  html("marketStrip", markets.strip);

  const audit = renderAuditRows(state);
  html("historyRows", audit.historyBody);
  set("eventCount", audit.eventCount);
  html("eventRows", audit.eventBody);

  const values = (state.equity_curve || []).map(x => Number(x.total ?? x));
  drawSeries(document.getElementById("overviewChart"), values, "#f0b90b", true);
  drawSeries(document.getElementById("analyticsChart"), values, "#f0b90b", true);
}


export function showSessionReview(index) {
  const x = (window.__paperState?.simulation_history || [])[index];
  if (!x) return;
  document.getElementById("drawerContent").innerHTML = sessionReview(x);
  openDrawer();
}

function sessionReview(x) {
  const label = esc(x.training_label || "flat");
  const stats = (x.symbol_stats || []).map(s => `
    <div>
      <b>${esc(s.symbol)}</b>
      <span>${num(s.trades, 0)} 笔 · 胜率 ${pct(s.win_rate_pct)}</span>
      <strong class="${tone(s.net_pnl)}">${money(s.net_pnl, 4)}</strong>
    </div>`).join("") || `<p class="review-note">暂无品种级交易数据</p>`;
  return `
    <div class="drawer-head">
      <span>模拟训练复盘</span>
      <h2>会话 ${esc(String(x.session_id).slice(0, 12))}</h2>
      <p>${dt(x.started_at)} → ${dt(x.ended_at)} · ${duration(x.duration_ms)}</p>
    </div>
    <div class="drawer-pnl ${tone(x.net_pnl)}">
      <span>会话净盈亏</span>
      <strong>${money(x.net_pnl, 4)}</strong>
      <em>${pct(x.return_pct || 0)}</em>
    </div>
    <dl class="detail-grid">
      ${pair("训练标签", label)}
      ${pair("结束原因", esc(x.end_reason || "--"))}
      ${pair("交易数", `${num(x.trades, 0)} · ${num(x.wins, 0)} 胜 / ${num(x.losses, 0)} 负`)}
      ${pair("胜率", pct(x.win_rate_pct))}
      ${pair("利润因子", num(x.profit_factor, 3))}
      ${pair("最大回撤", pct(x.max_drawdown_pct))}
      ${pair("最佳 / 最差交易", `${money(x.best_trade, 4)} / ${money(x.worst_trade, 4)}`)}
      ${pair("平均交易盈亏", money(x.avg_trade_pnl, 4))}
      ${pair("手续费", money(x.total_fees, 4))}
      ${pair("资金费", money(x.total_funding, 4))}
      ${pair("平均持仓时长", duration(x.average_holding_ms))}
      ${pair("策略决策数", num(x.decision_count, 0))}
      ${pair("交易品种", esc((x.symbols || []).join(", ") || "--"))}
    </dl>
    <h3>品种表现</h3>
    <div class="symbol-review-list">${stats}</div>
    <h3>训练结论</h3>
    <p class="review-note">${conclusion(x.training_label)}</p>`;
}

function conclusion(label) {
  if (label === "profitable") return "该会话产生正收益，可重点分析入场信号、持仓时长和盈利交易的共同特征。";
  if (label === "loss") return "该会话产生亏损，建议重点检查信号置信度、止损距离、手续费拖累和连续亏损段。";
  return "该会话接近持平，可用于比较不同参数下的交易频率与成本。";
}

export function showTradeDetail(index) {
  const x = (window.__paperState?.recent_trades || [])[index];
  if (!x) return;
  document.getElementById("drawerContent").innerHTML = `
    <div class="drawer-head">
      <span>成交详情</span>
      <h2>${esc(x.symbol)} ${sideBadge(x.side)}</h2>
      <p>${dt(x.exit_time || x.timestamp)}</p>
    </div>
    <div class="drawer-pnl ${tone(x.pnl)}">
      <span>净盈亏</span>
      <strong>${money(x.pnl, 4)}</strong>
      <em>${pct(x.pnl_pct)}</em>
    </div>
    <dl class="detail-grid">
      ${pair("订单号", `<span class="mono">${esc(x.order_id || "--")}</span>`)}
      ${pair("交易编号", `<span class="mono">${esc(x.trade_id || "--")}</span>`)}
      ${pair("开仓时间", dt(x.entry_time))}
      ${pair("平仓时间", dt(x.exit_time || x.timestamp))}
      ${pair("持仓时长", duration(x.holding_ms))}
      ${pair("杠杆", `${num(x.leverage, 0)}x`)}
      ${pair("开仓价格", num(x.entry_price || x.entry, 8))}
      ${pair("平仓价格", num(x.exit_price || x.exit, 8))}
      ${pair("成交数量", num(x.qty, 8))}
      ${pair("开仓名义价值", money(x.entry_notional, 4))}
      ${pair("占用保证金", money(x.margin_used, 4))}
      ${pair("毛盈亏", money(x.gross_pnl, 4))}
      ${pair("开仓手续费", money(x.entry_fee, 6))}
      ${pair("平仓手续费", money(x.exit_fee, 6))}
      ${pair("资金费", money(x.funding, 6))}
      ${pair("平仓原因", esc(x.reason || "--"))}
      ${pair("止损价格", stopCell(x, "stop"))}
      ${pair("止盈价格", stopCell(x, "target"))}
    </dl>`;
  openDrawer();
}

// The exit policy moves the stop: breakeven_at_rr raises it to the entry once the
// trade is 0.6R in profit and the trailing rule follows the best price after that.
// Showing only the final level made almost every trade read as "stop equals entry",
// which looks like a bug and is not one. Both levels are shown and the moved one is
// labelled, so a raise reads as a decision rather than as a wrong number.
function stopCell(x, kind) {
  const final = kind === "stop" ? x.stop_price : x.take_profit_price;
  const initial = kind === "stop" ? x.initial_stop : x.initial_target;
  const shown = num(final, 8);
  if (initial === undefined || initial === null || !isFinite(initial) || initial <= 0) {
    return shown;
  }
  if (Math.abs(initial - final) < 1e-12) {
    return shown;
  }
  const moved = kind === "stop" ? "已移保本/跟踪" : "已调整";
  return shown + ' <span class="level-note">(原 ' + num(initial, 8) + " · " + moved + ")</span>";
}

function pair(label, value) {
  return `<div><dt>${label}</dt><dd>${value}</dd></div>`;
}

function openDrawer() {
  document.getElementById("tradeDrawer").classList.add("open");
  document.getElementById("drawerBackdrop").classList.add("open");
}
