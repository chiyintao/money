// Pure formatting helpers. Kept free of DOM access so they can be unit tested.

export const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[c]));

export const num = (v, d = 2) => Number(v || 0).toLocaleString("zh-CN", {
  minimumFractionDigits: d, maximumFractionDigits: d,
});

export const compact = v => Number(v || 0).toLocaleString("zh-CN", {
  notation: "compact", maximumFractionDigits: 2,
});

export const pct = v => num(v, 2) + "%";

export const dt = v => v ? new Date(Number(v)).toLocaleString("zh-CN", { hour12: false }) : "--";

export const money = (v, d = 2) => num(v, d) + " USDT";

export const tone = v => Number(v) >= 0 ? "up" : "down";

export const signedPct = v => (Number(v) >= 0 ? "+" : "") + pct(v);

export function duration(ms) {
  if (!ms) return "--";
  const s = Math.floor(ms / 1000);
  if (s < 60) return s + "秒";
  if (s < 3600) return Math.floor(s / 60) + "分 " + (s % 60) + "秒";
  if (s < 86400) return Math.floor(s / 3600) + "时 " + Math.floor((s % 3600) / 60) + "分";
  return Math.floor(s / 86400) + "天 " + Math.floor((s % 86400) / 3600) + "时";
}

export const sideBadge = side => '<span class="side '
  + (side === "LONG" || side === "BUY" ? "long" : "short") + '">' + esc(side) + "</span>";

export const emptyRow = (cols, text) => '<tr><td colspan="' + cols + '" class="empty"><b>暂无数据</b><span>'
  + text + "</span></td></tr>";

export const SESSION_STATUS = {
  idle: "未启动", running: "运行中", paused: "已暂停", ended: "已结束", liquidated: "已强平",
};
