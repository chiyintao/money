// Market table and top market strip.

import { compact, esc, num, pct, sideBadge, tone } from "../format.js";
import { value } from "../dom.js";
import { emptyRow } from "../format.js";

export function sortMarkets(state) {
  const query = value("marketSearch").toUpperCase();
  const key = value("marketSort", "change");
  return [...(state.all || [])]
    .filter(x => String(x.symbol).includes(query))
    .sort((a, b) => key === "symbol"
      ? String(a.symbol).localeCompare(String(b.symbol))
      : Number(b[key] || 0) - Number(a[key] || 0));
}

function row(x) {
  const prediction = x.expected_return == null ? "" : " <small>" + num(x.expected_return * 10000, 2) + "bp</small>";
  return "<tr>"
    + '<td><b class="symbol">' + esc(x.symbol) + "</b><small>永续</small></td>"
    + "<td>" + num(x.price, 6) + "</td>"
    + '<td class="' + tone(x.change) + '">' + pct(x.change) + "</td>"
    + "<td>" + compact(x.volume) + "</td>"
    + "<td>" + pct(Number(x.funding_rate || 0) * 100) + "</td>"
    + "<td>" + sideBadge(x.side || "FLAT") + ' <span class="muted">' + num(x.confidence, 2) + "</span>" + prediction + "</td>"
    + '<td class="muted">' + esc(x.updated || "--") + "</td>"
    + "</tr>";
}

function stripItem(x) {
  return "<div><b>" + esc(x.symbol.replace("USDT", "")) + " / USDT</b>"
    + "<strong>" + num(x.price, 6) + "</strong>"
    + '<span class="' + tone(x.change) + '">' + pct(x.change) + "</span></div>";
}

export function renderMarketRows(state) {
  const rows = sortMarkets(state);
  return {
    body: rows.slice(0, 250).map(row).join("") || emptyRow(7, "没有匹配的合约"),
    strip: rows.slice(0, 5).map(stripItem).join(""),
  };
}
