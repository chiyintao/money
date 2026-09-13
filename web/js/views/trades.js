// Trade ledger view.

import { dt, duration, esc, money, num, pct, sideBadge, tone } from "../format.js";
import { value } from "../dom.js";
import { emptyRow } from "../format.js";

export function filterTrades(rows) {
  const query = value("ledgerSearch").toUpperCase();
  const side = value("ledgerSide");
  return rows.filter(x =>
    (!query || String(x.symbol || "").includes(query) || String(x.order_id || "").toUpperCase().includes(query))
    && (!side || x.side === side || x.side === (side === "LONG" ? "BUY" : "SELL")));
}

function row(x, index) {
  return "<tr>"
    + "<td>" + dt(x.exit_time || x.timestamp) + "</td>"
    + '<td><b class="symbol">' + esc(x.symbol) + "</b><small>" + esc(String(x.order_id || "").slice(0, 10)) + "</small></td>"
    + "<td>" + sideBadge(x.side) + "</td>"
    + "<td>" + num(x.qty, 6) + "</td>"
    + "<td>" + num(x.entry_price || x.entry, 6) + "<small>" + num(x.exit_price || x.exit, 6) + "</small></td>"
    + '<td class="' + tone(x.pnl) + '"><b>' + money(x.pnl, 4) + "</b><small>毛利 " + money(x.gross_pnl, 4) + "</small></td>"
    + '<td class="' + tone(x.pnl_pct) + '">' + pct(x.pnl_pct) + "</td>"
    + "<td>" + money(x.fees, 4) + "</td>"
    + "<td>" + money(x.funding, 4) + "</td>"
    + "<td>" + duration(x.holding_ms) + "</td>"
    + "<td>" + esc(x.reason || "--") + "</td>"
    + '<td><button class="detail-btn" data-trade-index="' + index + '">详情</button></td>'
    + "</tr>";
}

export function renderTradeRows(state, allTrades) {
  const source = state.recent_trades || [];
  const rows = filterTrades(source);
  const body = rows.length
    ? rows.map(x => row(x, source.indexOf(x))).join("")
    : emptyRow(12, "完成一笔平仓后会显示完整成交明细");
  return { count: source.length, body, allCount: (allTrades || []).length };
}
