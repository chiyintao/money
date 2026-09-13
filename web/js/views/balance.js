// Account balance and risk strip.

import { money, num, pct, signedPct, tone } from "../format.js";
import { set, width } from "../dom.js";

// Lifetime PnL as a return, not a dollar sum.
//
// The sessions in the trade table were started against different capitals -- the form
// lets the operator choose, and the history holds both 100 and 10,000 -- so adding their
// dollar PnL reported one number over accounts of wildly different size. A percentage of
// each account's own capital is the only form in which those rounds are comparable.
function lifetimeText(state) {
  const pctValue = state.lifetime_return_pct;
  const n = state.lifetime_trades || 0;
  if (pctValue === null || pctValue === undefined || !n) return "历史累计 --";
  return "历史累计 " + signedPct(pctValue) + "（" + n + " 笔）";
}

export function renderBalances(state) {
  set("equity", money(state.equity));
  set("cash", money(state.cash));
  set("availableMargin", money(state.available_margin));
  set("unrealizedPnl", money(state.unrealized_pnl, 4), tone(state.unrealized_pnl));
  set("realizedPnl", money(state.realized_pnl, 4), tone(state.realized_pnl));
  set("realizedPnlNote", (state.session_trades || 0) + " 笔本会话 · " + lifetimeText(state));
  set("marginBalance", money(state.equity));
  set("returnPct", signedPct(state.return_pct), tone(state.return_pct));
  set("marginUsage", pct(state.margin_usage_pct));
  set("totalNotional", money(state.total_notional));
  set("usedMargin", money(state.used_margin));
  set("maintenanceMargin", money(state.maintenance_margin));
  set("leverage", num(state.gross_leverage, 2) + "x");
  set("tradeEquity", money(state.equity));
  set("tradePnl", money(state.total_pnl, 4), tone(state.total_pnl));
  set("tradeAvailable", money(state.available_margin));
  set("tradeNotional", money(state.total_notional));
  set("tradeFees", money(state.total_fees, 4));
  set("tradeWinRate", pct(state.win_rate_pct));
}

export function renderRisk(state) {
  const risk = state.risk_state || {};
  set("riskStatus", risk.halted ? "已熔断" : "正常", "status-badge " + (risk.halted ? "bad" : "good"));
  set("riskSubtitle", risk.halted
    ? "日亏损保护：权益 " + money(risk.halt_equity) + " ≤ 阈值 " + money(risk.halt_threshold)
    : "Cross 保证金 · 日亏损上限 " + pct(Number(risk.max_daily_loss || 0) * 100));
  width("marginUsageBar", state.margin_usage_pct);
}

export function renderAnalytics(state) {
  const metrics = state.metrics || {};
  set("analyticsTrades", metrics.trades || 0);
  set("analyticsWin", pct(state.win_rate_pct));
  set("analyticsDrawdown", pct(metrics.max_drawdown_pct));
  set("analyticsProfitFactor", num(metrics.profit_factor, 2));
  set("metricSummary", (metrics.trades || 0) + " 笔交易 · 最大回撤 " + pct(metrics.max_drawdown_pct)
    + " · " + lifetimeText(state));
}
