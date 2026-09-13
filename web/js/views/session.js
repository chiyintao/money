// Session control bar.

import { SESSION_STATUS, dt, esc, money, num, pct } from "../format.js";
import { disabled, set, html } from "../dom.js";

// Sizing is returned as fractions of equity; percentages are what the operator reasons in.
const asPct = value => value === null || value === undefined ? "--" : pct(value * 100);

/**
 * Fill a profile <select> from the catalogue, preserving the current choice.
 *
 * The options come from the backend so the presets are defined in exactly one place.
 * Rebuilding the list on every render would also reset the user's in-progress choice,
 * so the selection is restored explicitly.
 */
export function fillProfileSelect(id, catalogue) {
  const node = document.getElementById(id);
  if (!node || !catalogue?.profiles) return;
  const wanted = node.value || catalogue.default;
  if (node.options.length !== catalogue.profiles.length) {
    // Label and key figures only. An <option> cannot wrap, so putting the full
    // description here made the select 542px wide and forced the page to scroll
    // sideways on phones. The description belongs in the readout, which can wrap.
    node.innerHTML = catalogue.profiles
      .map(p => '<option value="' + esc(p.name) + '">' + esc(p.label)
        + " · 单笔 " + (p.max_risk_per_trade * 100) + "% · 敞口 " + p.target_exposure + "x</option>")
      .join("");
  }
  node.value = catalogue.profiles.some(p => p.name === wanted) ? wanted : catalogue.default;
}

// The name the panel last rendered. The live selector is only forced back to it when it
// genuinely changes: assigning on every frame reverted the operator's choice before they
// could press the switch button, which looked like the switch was simply broken.
let renderedProfileName = null;
let renderedExitName = null;

export function fillPolicySelect(id, catalogue) {
  const node = document.getElementById(id);
  if (!node || !catalogue?.policies) return;
  const wanted = node.value || catalogue.default;
  if (node.options.length !== catalogue.policies.length) {
    // Short labels only: an <option> cannot wrap, so a full description here widens the
    // select past the viewport on phones.
    node.innerHTML = catalogue.policies
      .map(p => '<option value="' + esc(p.name) + '">' + esc(p.label)
        + " · " + p.stop_atr_multiple + "ATR · " + p.target_rr + "R</option>")
      .join("");
  }
  node.value = catalogue.policies.some(p => p.name === wanted) ? wanted : catalogue.default;
}

export function renderExitPolicy(catalogue) {
  if (!catalogue?.active) return;
  const policy = catalogue.active;
  const shown = catalogue.name === "custom" ? "自定义" : (policy.label || policy.name);
  set("sessionExitName", shown);
  const trail = policy.trail_atr_multiple ? policy.trail_atr_multiple + "ATR 跟踪" : "不跟踪";
  const be = policy.breakeven_at_rr ? policy.breakeven_at_rr + "R 保本" : "不保本";
  const timeStop = policy.time_stop_bars ? policy.time_stop_bars + " 根K线强平" : "无时间止损";
  set("sessionExitReadout", "止损 " + policy.stop_atr_multiple + "ATR · 盈亏比 " + policy.target_rr
    + " · " + be + " · " + trail + " · " + timeStop);
  if (catalogue.name !== renderedExitName) {
    renderedExitName = catalogue.name;
    const live = document.getElementById("liveExitSelect");
    if (live && catalogue.name && catalogue.name !== "custom") live.value = catalogue.name;
  }
}

export function renderRiskProfile(catalogue) {
  if (!catalogue?.active) return;
  const profile = catalogue.active;
  const shown = catalogue.name === "custom" ? "自定义" : (profile.label || profile.name);
  set("activeProfile", shown);
  set("profileRisk", asPct(profile.max_risk_per_trade));
  set("profilePortfolioRisk", asPct(profile.max_portfolio_risk));
  set("profileExposure", num(profile.target_exposure, 1) + "x");
  set("profilePositions", num(profile.max_positions, 0));
  // The session bar carries the same figures, because that is the page where a session
  // is configured and where the switch has to be reachable.
  set("sessionProfileName", shown);
  set("sessionProfileReadout", "单笔 " + asPct(profile.max_risk_per_trade)
    + " · 组合 " + asPct(profile.max_portfolio_risk)
    + " · 敞口 " + num(profile.target_exposure, 1) + "x");
  if (catalogue.name !== renderedProfileName) {
    renderedProfileName = catalogue.name;
    const live = document.getElementById("liveProfileSelect");
    if (live && catalogue.name && catalogue.name !== "custom") live.value = catalogue.name;
  }
}

export function renderSession(state) {
  const sim = state.simulation || {};
  const status = sim.status || "idle";
  set("sessionState", SESSION_STATUS[status] || status, "session-state " + status);
  set("sessionTitle", sim.session_id ? "会话 " + String(sim.session_id).slice(0, 10) : "模拟交易会话");
  set("sessionMeta", sim.started_at
    ? dt(sim.started_at) + " · " + num(sim.leverage, 0) + "x · " + (sim.selected_symbols || []).join(", ")
    : "配置资金与策略范围后启动");
  disabled("startSession", status === "running" || status === "paused");
  disabled("pauseSession", status !== "running");
  disabled("resumeSession", status !== "paused");
  disabled("endSession", !["running", "paused"].includes(status));
}
