// Session history and event audit views.

import { dt, duration, esc, money, num, pct, tone } from "../format.js";
import { emptyRow } from "../format.js";

function sessionRow(x, index) {
  const label = esc(x.training_label || "flat");
  return "<tr>"
    + '<td class="mono">' + esc(String(x.session_id).slice(0, 12)) + "</td>"
    + '<td><span class="training-tag ' + label + '">' + esc(x.status) + "</span></td>"
    + "<td>" + dt(x.started_at) + "<small>" + duration(x.duration_ms) + "</small></td>"
    + "<td>" + money(x.initial_cash) + "<small>" + num(x.leverage, 0) + "x · " + esc((x.symbols || []).slice(0, 3).join(", ")) + "</small></td>"
    + '<td class="' + tone(x.net_pnl) + '"><b>' + money(x.net_pnl, 4) + "</b><small>均值 " + money(x.avg_trade_pnl, 4) + "</small></td>"
    + '<td class="' + tone(x.return_pct || x.net_pnl) + '">' + pct(x.return_pct || ((x.net_pnl || 0) / (x.initial_cash || 1) * 100)) + "</td>"
    + "<td>" + pct(x.win_rate_pct) + "<small>" + num(x.wins, 0) + " 胜 / " + num(x.losses, 0) + " 负</small></td>"
    + "<td>" + num(x.profit_factor, 2) + "</td>"
    + "<td>" + pct(x.max_drawdown_pct) + "</td>"
    + "<td>" + money(x.total_fees, 4) + "<small>" + money(x.total_funding, 4) + "</small></td>"
    + '<td><span class="training-tag ' + label + '">' + label + "</span></td>"
    + '<td><button class="session-review-btn" data-session-index="' + index + '">复盘</button></td>'
    + "</tr>";
}

function eventRow(x) {
  return "<tr><td>" + dt(x.event_time) + "</td>"
    + '<td><span class="event-type">' + esc(x.type) + "</span></td>"
    + '<td class="event-payload">' + esc(JSON.stringify(x.payload || {})) + "</td></tr>";
}

export function renderAuditRows(state) {
  const sessions = state.simulation_history || [];
  const events = state.events || [];
  return {
    historyBody: sessions.map(sessionRow).join("") || emptyRow(12, "启动模拟会话后将在这里留档"),
    eventCount: events.length + " 条事件",
    eventBody: events.map(eventRow).join("") || emptyRow(3, "暂无审计事件"),
  };
}
