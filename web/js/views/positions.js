// Position row template shared by the overview and trade tables.

import { esc, money, num, pct, tone, dt } from "../format.js";
import { sideBadge } from "../format.js";

const CLOSE_BUTTON = symbol => '<td><button class="close-position-btn" data-symbol="' + esc(symbol) + '">平仓</button></td>';

export function positionRow(x, detail = false) {
  const pnl = Number(x.unrealized_pnl || 0);
  const margin = Number(x.margin || 0);
  const roi = margin ? (pnl / margin) * 100 : 0;
  return "<tr>"
    + '<td><b class="symbol">' + esc(x.symbol) + "</b><small>永续</small></td>"
    + "<td>" + sideBadge(x.side) + ' <span class="lev">' + num(x.leverage || 0, 0) + "x</span></td>"
    + "<td>" + num(x.qty, 6) + "</td>"
    + (detail ? "<td>" + money(x.notional) + "</td>" : "")
    + "<td>" + num(x.entry, 6) + (detail ? "<small>" + num(x.mark, 6) + "</small>" : "") + "</td>"
    + (!detail ? "<td>" + num(x.mark, 6) + "</td>" : "")
    + '<td class="' + tone(pnl) + '"><b>' + money(pnl, 4) + "</b></td>"
    + '<td class="' + tone(roi) + '">' + pct(roi) + "</td>"
    + "<td>" + money(x.margin) + "</td>"
    + "<td>" + num(x.liquidation_price, 6) + "</td>"
    + "<td>" + num(x.target, 6) + "<small>" + num(x.stop, 6) + "</small></td>"
    + (detail ? "<td>" + dt(x.opened_at) + "</td>" : "")
    + CLOSE_BUTTON(x.symbol)
    + "</tr>";
}
