// Open orders and order history views.

import { dt, esc, num, sideBadge } from "../format.js";
import { emptyRow } from "../format.js";
import { filterTrades } from "./trades.js";

function openRow(x) {
  return "<tr>"
    + "<td>" + dt(x.created_at) + "</td>"
    + '<td><b class="symbol">' + esc(x.symbol) + "</b></td>"
    + "<td>" + sideBadge(x.side) + "</td>"
    + "<td>" + esc(x.order_type) + "</td>"
    + "<td>" + num(x.limit_price, 6) + "</td>"
    + "<td>" + num(x.quantity, 6) + "<small>" + num(x.filled_quantity, 6) + "</small></td>"
    + "<td>" + (x.reduce_only ? "是" : "否") + "</td>"
    + '<td><span class="order-status open">' + esc(x.status) + "</span></td>"
    + '<td class="mono">' + esc(String(x.order_id).slice(0, 14)) + "</td>"
    + '<td><button class="cancel-btn" data-id="' + esc(x.order_id) + '">撤单</button></td>'
    + "</tr>";
}

function historyRow(x) {
  return "<tr>"
    + "<td>" + dt(x.created_at) + "</td>"
    + "<td>" + dt(x.updated_at) + "</td>"
    + '<td><b class="symbol">' + esc(x.symbol) + "</b></td>"
    + "<td>" + sideBadge(x.side) + "</td>"
    + "<td>" + esc(x.order_type) + "</td>"
    + "<td>" + num(x.limit_price, 6) + "</td>"
    + "<td>" + num(x.quantity, 6) + "<small>" + num(x.filled_quantity, 6) + "</small></td>"
    + '<td><span class="order-status ' + String(x.status).toLowerCase() + '">' + esc(x.status) + "</span></td>"
    + '<td class="mono">' + esc(String(x.order_id).slice(0, 16)) + "</td>"
    + "</tr>";
}

export function renderOrderRows(state) {
  const open = filterTrades(state.open_orders || []);
  const history = filterTrades([...(state.order_history || [])].sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0)));
  return {
    openCount: (state.open_orders || []).length,
    openBody: open.length ? open.map(openRow).join("") : emptyRow(10, "当前没有等待成交的委托"),
    historyBody: history.length ? history.map(historyRow).join("") : emptyRow(9, "委托创建后会保留全部状态记录"),
  };
}
