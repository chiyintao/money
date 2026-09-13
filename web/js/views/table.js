// Shared table rendering. Every list view used to inline its own row template and
// its own empty-state fallback; they are unified here.

import { esc, emptyRow } from "../format.js";

export function renderRows(rows, columns, emptyText) {
  if (!rows.length) return emptyRow(columns, emptyText);
  return rows.join("");
}

export function cell(value, cls) {
  return cls ? '<td class="' + cls + '">' + value + "</td>" : "<td>" + value + "</td>";
}

export function symbolCell(symbol, note = "永续") {
  return '<td><b class="symbol">' + esc(symbol) + "</b><small>" + esc(note) + "</small></td>";
}

export function sub(main, note) {
  return main + "<small>" + note + "</small>";
}
